import cv2
import mediapipe as mp
import numpy as np
import torch

#––– Setup –––
mp_holistic    = mp.solutions.holistic
mp_drawing     = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(
    static_image_mode=False,       # live feed
    model_complexity=1,            # 0,1,2→lighter↔heavier face model
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally for a selfie-view & convert BGR→RGB
    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with Holistic
    results = holistic.process(rgb)

    # Draw face mesh
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=results.face_landmarks,
            connections=mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_tesselation_style()
        )
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=results.face_landmarks,
            connections=mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_contours_style()
        )

    # Draw pose (including arms)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=results.pose_landmarks,
            connections=mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
              .get_default_pose_landmarks_style()
        )

    # Draw left & right hands
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2),
        )
    if results.right_hand_landmarks:
        L = results.pose_landmarks.landmark  # list of 33 NormalizedLandmark
        print(type(L))

        # 1) node‐features: shape (N,3)
        coords = np.stack([[lm.x, lm.y, lm.z] for lm in L], axis=0)
        coords_torch = torch.from_numpy(coords).float()  # FloatTensor[N,3]

        # 2) edge‐index: shape (2, E)
        #    POSE_CONNECTIONS is a set of (u,v) pairs
        # NOTE: The adjacency list is always defined the same way (so the adjacency list is not something we have to learn)
        edges = np.array(list(mp.solutions.holistic.POSE_CONNECTIONS))  # (E,2)
        # for GCN libs we often want shape (2,E):
        edge_index = torch.from_numpy(edges.T).long()  # LongTensor[2, E]

        # 3) (optional) build adjacency matrix
        N = coords.shape[0]
        adj = np.zeros((N, N), dtype=np.float32)
        for u, v in edges:
            adj[u, v] = adj[v, u] = 1.0
        adj_torch = torch.from_numpy(adj)  # [N, N]

        print(adj_torch)
        print(coords_torch)

        # Now coords_torch, edge_index (and/or adj_torch) *are* your graph as tensors.
    mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2),
        )

    # Show the result
    cv2.imshow('MediaPipe Holistic', frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
