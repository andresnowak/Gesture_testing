import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from enum import Enum


# NOTE: We use the normalized landmarks (normalized based on height and width of the image)

class CoordinateSystem(Enum):
    """Enum for different coordinate systems"""
    ORIGINAL = "original"  # Raw MediaPipe normalized coordinates [0,1]
    SHOULDER_CENTERED = "shoulder_centered"  # Unified shoulder-centered coordinates

class VideoPreprocessor:
    def __init__(self,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 include_pose: bool = True,
                 include_face: bool = True,
                 include_hands: bool = True):
        """
        Initialize MediaPipe Holistic model

        Args:
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
            include_pose: Whether to include pose landmarks in extraction
            include_face: Whether to include face landmarks in extraction
            include_hands: Whether to include hand landmarks in extraction
        """
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize Holistic model
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Body part selection flags
        self.include_pose = include_pose
        self.include_face = include_face
        self.include_hands = include_hands

        # Define landmark indices for each component
        self.pose_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.face_landmarks = [132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 367, 288, 435, 361]
        self.hand_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    def build_adjacency_matrices(self) -> Dict[str, np.ndarray]:
        """
        Build adjacency matrices for each body part based on MediaPipe connections.
        Returns dictionary with adjacency matrices for pose, face, and hands.
        Matrices are built based on selected landmark indices only.
        """
        adjacency_matrices = {}

        # Build pose adjacency matrix
        if self.include_pose:
            pose_size = len(self.pose_landmarks)
            pose_adj = np.zeros((pose_size, pose_size), dtype=np.float32)

            # Create mapping from original index to filtered index
            pose_idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(self.pose_landmarks)}

            # Add edges from POSE_CONNECTIONS
            for connection in self.mp_holistic.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                # Only add connection if both landmarks are in our selected set
                if start_idx in pose_idx_map and end_idx in pose_idx_map:
                    new_start = pose_idx_map[start_idx]
                    new_end = pose_idx_map[end_idx]
                    pose_adj[new_start, new_end] = 1.0
                    pose_adj[new_end, new_start] = 1.0  # Undirected graph

            adjacency_matrices['pose'] = pose_adj

        # Build face adjacency matrix
        if self.include_face:
            face_size = len(self.face_landmarks)
            face_adj = np.zeros((face_size, face_size), dtype=np.float32)

            # Create mapping from original index to filtered index
            face_idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(self.face_landmarks)}

            # Add edges from FACEMESH_CONTOURS
            for connection in self.mp_holistic.FACEMESH_CONTOURS:
                start_idx, end_idx = connection
                # Only add connection if both landmarks are in our selected set
                if start_idx in face_idx_map and end_idx in face_idx_map:
                    new_start = face_idx_map[start_idx]
                    new_end = face_idx_map[end_idx]
                    face_adj[new_start, new_end] = 1.0
                    face_adj[new_end, new_start] = 1.0  # Undirected graph

            adjacency_matrices['face'] = face_adj

        # Build hand adjacency matrices (same structure for both hands)
        if self.include_hands:
            hand_size = len(self.hand_landmarks)
            hand_adj = np.zeros((hand_size, hand_size), dtype=np.float32)

            # Create mapping from original index to filtered index
            hand_idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(self.hand_landmarks)}

            # Add edges from HAND_CONNECTIONS
            for connection in self.mp_holistic.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                # Only add connection if both landmarks are in our selected set
                if start_idx in hand_idx_map and end_idx in hand_idx_map:
                    new_start = hand_idx_map[start_idx]
                    new_end = hand_idx_map[end_idx]
                    hand_adj[new_start, new_end] = 1.0
                    hand_adj[new_end, new_start] = 1.0  # Undirected graph

            adjacency_matrices['left_hand'] = hand_adj
            adjacency_matrices['right_hand'] = hand_adj.copy()  # Same structure for both hands

        return adjacency_matrices

    def build_unified_adjacency_matrix(self) -> np.ndarray:
        """
        Build a unified adjacency matrix for the shoulder-centered coordinate system.
        This includes connections within each body part AND connections between body parts:
        - Hands to pose via wrists (hand wrist idx 0 connects to pose wrist idx 15/16)
        - Face to pose via nose (face nose idx 0 connects to pose nose idx 0)
        """
        # Define inter-body-part connections (in original MediaPipe indices)
        # HANDS_POSE_CONNECTIONS = frozenset([(0, 15), (0, 16)])  # (hand_wrist, left_pose_wrist), (hand_wrist, right_pose_wrist)
        # FACE_POSE_CONNECTIONS = frozenset([(0, 0)])  # (face_nose, pose_nose)

        # Calculate total size
        total_landmarks = 0
        if self.include_pose:
            total_landmarks += len(self.pose_landmarks)
        if self.include_face:
            total_landmarks += len(self.face_landmarks)
        if self.include_hands:
            total_landmarks += 2 * len(self.hand_landmarks)

        # Initialize unified adjacency matrix
        unified_adj = np.zeros((total_landmarks, total_landmarks), dtype=np.float32)

        # Get individual adjacency matrices
        adj_matrices = self.build_adjacency_matrices()

        # Create index mappings from original MediaPipe indices to filtered indices
        if self.include_pose:
            pose_idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(self.pose_landmarks)}
        if self.include_face:
            face_idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(self.face_landmarks)}
        if self.include_hands:
            hand_idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(self.hand_landmarks)}

        # Fill in the unified matrix block by block
        current_idx = 0
        indices_pos = {}

        if self.include_pose:
            pose_size = len(self.pose_landmarks)
            unified_adj[current_idx:current_idx+pose_size, current_idx:current_idx+pose_size] = adj_matrices['pose']
            indices_pos["pose"] = (current_idx, current_idx + pose_size)
            current_idx += pose_size

        if self.include_face:
            face_size = len(self.face_landmarks)
            unified_adj[current_idx:current_idx+face_size, current_idx:current_idx+face_size] = adj_matrices['face']
            indices_pos["face"] = (current_idx, current_idx + face_size)
            current_idx += face_size

        if self.include_hands:
            hand_size = len(self.hand_landmarks)
            # Left hand (MediaPipe left = actually right hand in mirrored video)
            unified_adj[current_idx:current_idx+hand_size, current_idx:current_idx+hand_size] = adj_matrices['left_hand']
            indices_pos["left_hand"] = (current_idx, current_idx + hand_size)
            current_idx += hand_size

            # Right hand (MediaPipe right = actually left hand in mirrored video)
            unified_adj[current_idx:current_idx+hand_size, current_idx:current_idx+hand_size] = adj_matrices['right_hand']
            indices_pos["right_hand"] = (current_idx, current_idx + hand_size)

        # Add inter-body-part connections
        # Connect hands to pose (wrists)
        if self.include_hands and self.include_pose:
            # Check if wrist landmarks exist in filtered sets
            hand_wrist_orig = 0  # Hand wrist is always index 0 in MediaPipe
            left_wrist_pose_orig = 15  # Left wrist in pose
            right_wrist_pose_orig = 16  # Right wrist in pose

            if hand_wrist_orig in hand_idx_map:
                # Connect left hand wrist to pose right wrist (because of mirroring)
                left_hand_wrist_unified = indices_pos["left_hand"][0] + hand_idx_map[hand_wrist_orig]
                pose_right_wrist_unified = indices_pos["pose"][0] + pose_idx_map[right_wrist_pose_orig]
                unified_adj[left_hand_wrist_unified, pose_right_wrist_unified] = 1.0
                unified_adj[pose_right_wrist_unified, left_hand_wrist_unified] = 1.0

                # Connect right hand wrist to pose left wrist (because of mirroring)
                right_hand_wrist_unified = indices_pos["right_hand"][0] + hand_idx_map[hand_wrist_orig]
                pose_left_wrist_unified = indices_pos["pose"][0] + pose_idx_map[left_wrist_pose_orig]
                unified_adj[right_hand_wrist_unified, pose_left_wrist_unified] = 1.0
                unified_adj[pose_left_wrist_unified, right_hand_wrist_unified] = 1.0

        # Connect face to pose (nose)
        if self.include_face and self.include_pose:
            face_nose_orig = 1  # Nose is index 0 in both
            pose_nose_orig = 0

            face_nose_unified = indices_pos["face"][0] + face_idx_map[face_nose_orig]
            pose_nose_unified = indices_pos["pose"][0] + pose_idx_map[pose_nose_orig]
            unified_adj[face_nose_unified, pose_nose_unified] = 1.0
            unified_adj[pose_nose_unified, face_nose_unified] = 1.0

        return unified_adj

    def calculate_visibility_score(self, landmark, margin: float = 0.1) -> float:
        """
        Returns visibility score [0,1] based on how far landmark is from boundaries.
        Useful for detecting when hands/faces are leaving the frame.

        Args:
            landmark: MediaPipe landmark object with x, y attributes (normalized [0,1])
            margin: how close to edge before we consider it "leaving"

        Returns:
            float: visibility score between 0.0 (outside/at edge) and 1.0 (well inside frame)
        """
        x, y = landmark.x, landmark.y

        # If clearly outside frame
        if x < 0 or x > 1 or y < 0 or y > 1:
            return 0.0

        # Calculate distance from nearest edge
        dist_to_edge = min(x, 1-x, y, 1-y)

        # Normalize: 0 at boundary, 1.0 at margin distance inward
        visibility = min(dist_to_edge / margin, 1.0)

        return visibility

    def unify_normalized_landmarks_to_shoulder_center(self, results) -> Optional[Dict]:
        """
        Transform all MediaPipe Holistic landmarks to shoulder-centered coordinates.
        Shoulder center = midpoint between left shoulder (11) and right shoulder (12)

        Uses normalized coordinates [0,1] and transforms all body parts to a unified
        coordinate system centered at the shoulders.

        Note: z-coordinates have different scales but are combined using displacement vectors:
          - Pose z: relative to hips
          - Hand z: relative to wrist
          - Face z: relative to head center
        """
        if not results.pose_landmarks:
            return None

        pose_norm = results.pose_landmarks.landmark

        # Calculate shoulder center (NEW ORIGIN)
        left_shoulder = pose_norm[11]
        right_shoulder = pose_norm[12]

        shoulder_center = {
            'x': (left_shoulder.x + right_shoulder.x) / 2,
            'y': (left_shoulder.y + right_shoulder.y) / 2,
            'z': (left_shoulder.z + right_shoulder.z) / 2
        }

        unified_data = {
            'pose': [],
            'face': [],
            'left_hand': [],
            'right_hand': []
        }

        # Transform pose to shoulder-centered
        for landmark in pose_norm:
            unified_data['pose'].append({
                'x': landmark.x - shoulder_center['x'],
                'y': landmark.y - shoulder_center['y'],
                'z': landmark.z - shoulder_center['z'],
                'visibility': landmark.visibility
            })

        # Get shoulder-centered anchor points
        nose_shoulder_centered = unified_data['pose'][0]
        left_wrist_shoulder_centered = unified_data['pose'][15]
        right_wrist_shoulder_centered = unified_data['pose'][16]

        # Transform face using nose as anchor
        if results.face_landmarks:
            face_norm = results.face_landmarks.landmark
            nose_face = face_norm[1]  # Nose tip in face mesh

            for landmark in face_norm:
                # Calculate displacement vector from landmark to nose in face coordinate system
                offset_x = landmark.x - nose_face.x
                offset_y = landmark.y - nose_face.y
                offset_z = landmark.z - nose_face.z

                # Apply displacement to shoulder-centered nose position
                unified_data['face'].append({
                    'x': nose_shoulder_centered['x'] + offset_x,
                    'y': nose_shoulder_centered['y'] + offset_y,
                    'z': nose_shoulder_centered['z'] + offset_z
                })

        # Transform left hand using wrist as anchor
        if results.left_hand_landmarks:
            left_hand_norm = results.left_hand_landmarks.landmark
            left_wrist_hand = left_hand_norm[0]

            for landmark in left_hand_norm:
                # Calculate displacement vector in hand's coordinate system
                offset_x = landmark.x - left_wrist_hand.x
                offset_y = landmark.y - left_wrist_hand.y
                offset_z = landmark.z - left_wrist_hand.z

                # Apply displacement to shoulder-centered wrist position
                unified_data['left_hand'].append({
                    'x': left_wrist_shoulder_centered['x'] + offset_x,
                    'y': left_wrist_shoulder_centered['y'] + offset_y,
                    'z': left_wrist_shoulder_centered['z'] + offset_z
                })

        # Transform right hand using wrist as anchor
        if results.right_hand_landmarks:
            right_hand_norm = results.right_hand_landmarks.landmark
            right_wrist_hand = right_hand_norm[0]

            for landmark in right_hand_norm:
                # Calculate displacement vector in hand's coordinate system
                offset_x = landmark.x - right_wrist_hand.x
                offset_y = landmark.y - right_wrist_hand.y
                offset_z = landmark.z - right_wrist_hand.z

                # Apply displacement to shoulder-centered wrist position
                unified_data['right_hand'].append({
                    'x': right_wrist_shoulder_centered['x'] + offset_x,
                    'y': right_wrist_shoulder_centered['y'] + offset_y,
                    'z': right_wrist_shoulder_centered['z'] + offset_z
                })

        return unified_data

    def extract_landmarks(self, results) -> Dict[str, np.ndarray]:
        """
        Extract ORIGINAL landmarks from MediaPipe Holistic (normalized coordinates [0,1])
        Returns dictionary with pose, face, left_hand, right_hand landmarks (based on include flags)
        Each landmark has [x, y, z, visibility] format
        """
        landmarks_data = {}

        # Pose landmarks
        if self.include_pose:
            if results.pose_landmarks:
                pose_landmarks = []
                for index, landmark in enumerate(results.pose_landmarks.landmark):
                    if index in self.pose_landmarks:
                        pose_landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                landmarks_data['pose'] = np.array(pose_landmarks)
            else:
                landmarks_data['pose'] = np.zeros((len(self.pose_landmarks), 4))

        # Face landmarks
        if self.include_face:
            if results.face_landmarks:
                face_landmarks = []
                for index, landmark in enumerate(results.face_landmarks.landmark):
                    if index in self.face_landmarks:
                        visibility = self.calculate_visibility_score(landmark)
                        face_landmarks.append([landmark.x, landmark.y, landmark.z, visibility])
                landmarks_data['face'] = np.array(face_landmarks)
            else:
                landmarks_data['face'] = np.zeros((len(self.face_landmarks), 4))

        # Hand landmarks
        if self.include_hands:
            # Left hand landmarks
            if results.left_hand_landmarks:
                left_hand_landmarks = []
                for index, landmark in enumerate(results.left_hand_landmarks.landmark):
                    if index in self.hand_landmarks:
                        visibility = self.calculate_visibility_score(landmark)
                        left_hand_landmarks.append([landmark.x, landmark.y, landmark.z, visibility])
                landmarks_data['left_hand'] = np.array(left_hand_landmarks)
            else:
                landmarks_data['left_hand'] = np.zeros((len(self.hand_landmarks), 4))

            # Right hand landmarks
            if results.right_hand_landmarks:
                right_hand_landmarks = []
                for index, landmark in enumerate(results.right_hand_landmarks.landmark):
                    if index in self.hand_landmarks:
                        visibility = self.calculate_visibility_score(landmark)
                        right_hand_landmarks.append([landmark.x, landmark.y, landmark.z, visibility])
                landmarks_data['right_hand'] = np.array(right_hand_landmarks)
            else:
                landmarks_data['right_hand'] = np.zeros((len(self.hand_landmarks), 4))

        return landmarks_data

    def extract_unified_landmarks(self, results) -> Optional[np.ndarray]:
        """
        Extract SHOULDER-CENTERED unified landmarks from MediaPipe Holistic
        Returns a single concatenated array of SELECTED landmarks in shoulder-centered coordinates
        Shape depends on which body parts are included (based on include flags)
        Each landmark has [x, y, z, visibility] format
        """
        unified_data = self.unify_normalized_landmarks_to_shoulder_center(results)

        if unified_data is None:
            return None

        landmarks_to_concatenate = []

        # Convert pose to numpy array - FILTER selected landmarks only
        if self.include_pose:
            if unified_data['pose']:
                pose_array = []
                for idx in self.pose_landmarks:
                    lm = unified_data['pose'][idx]
                    pose_array.append([lm['x'], lm['y'], lm['z'], lm.get('visibility', 1.0)])
                pose_landmarks = np.array(pose_array)
            else:
                pose_landmarks = np.zeros((len(self.pose_landmarks), 4))
            landmarks_to_concatenate.append(pose_landmarks)

        # Convert face to numpy array - FILTER selected landmarks only
        if self.include_face:
            if unified_data['face'] and results.face_landmarks:
                face_array = []
                for idx in self.face_landmarks:
                    lm = unified_data['face'][idx]
                    # Calculate visibility from original normalized coordinates
                    visibility = self.calculate_visibility_score(results.face_landmarks.landmark[idx])
                    face_array.append([lm['x'], lm['y'], lm['z'], visibility])
                face_landmarks = np.array(face_array)
            else:
                face_landmarks = np.zeros((len(self.face_landmarks), 4))
            landmarks_to_concatenate.append(face_landmarks)

        # Convert hands to numpy arrays - FILTER selected landmarks only
        if self.include_hands:
            # Left hand
            if unified_data['left_hand'] and results.left_hand_landmarks:
                left_hand_array = []
                for idx in self.hand_landmarks:
                    lm = unified_data['left_hand'][idx]
                    # Calculate visibility from original normalized coordinates
                    visibility = self.calculate_visibility_score(results.left_hand_landmarks.landmark[idx])
                    left_hand_array.append([lm['x'], lm['y'], lm['z'], visibility])
                left_hand_landmarks = np.array(left_hand_array)
            else:
                left_hand_landmarks = np.zeros((len(self.hand_landmarks), 4))
            landmarks_to_concatenate.append(left_hand_landmarks)

            # Right hand
            if unified_data['right_hand'] and results.right_hand_landmarks:
                right_hand_array = []
                for idx in self.hand_landmarks:
                    lm = unified_data['right_hand'][idx]
                    # Calculate visibility from original normalized coordinates
                    visibility = self.calculate_visibility_score(results.right_hand_landmarks.landmark[idx])
                    right_hand_array.append([lm['x'], lm['y'], lm['z'], visibility])
                right_hand_landmarks = np.array(right_hand_array)
            else:
                right_hand_landmarks = np.zeros((len(self.hand_landmarks), 4))
            landmarks_to_concatenate.append(right_hand_landmarks)

        # Concatenate selected landmarks into a single unified graph
        if landmarks_to_concatenate:
            unified_graph = np.concatenate(landmarks_to_concatenate, axis=0)
        else:
            # If no body parts selected, return empty array
            unified_graph = np.zeros((0, 4))

        return unified_graph

    def draw_landmarks_on_frame(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Draw all holistic landmarks on the frame
        """
        annotated_frame = frame.copy()

        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # Draw face landmarks
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )

        # Draw left hand landmarks
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
            )

        # Draw right hand landmarks
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
            )

        return annotated_frame

    def process_video(self,
                     video_path: str,
                     output_dir: str,
                     save_annotated_video: bool = True,
                     coordinate_systems: List[CoordinateSystem] = [CoordinateSystem.ORIGINAL]) -> Dict:
        """
        Process a single video file

        Args:
            video_path: Path to input video
            output_dir: Directory to save outputs
            save_annotated_video: Whether to save annotated video with landmarks drawn
            coordinate_systems: List of coordinate systems to save. Options:
                - CoordinateSystem.ORIGINAL: Raw MediaPipe normalized coordinates [0,1]
                - CoordinateSystem.SHOULDER_CENTERED: Unified shoulder-centered coordinates

        Example:
            # Save only original coordinates
            process_video(..., coordinate_systems=[CoordinateSystem.ORIGINAL])

            # Save only shoulder-centered
            process_video(..., coordinate_systems=[CoordinateSystem.SHOULDER_CENTERED])

            # Save both
            process_video(..., coordinate_systems=[CoordinateSystem.ORIGINAL,
                                                   CoordinateSystem.SHOULDER_CENTERED])
        """
        video_name = Path(video_path).stem
        gloss = video_name.split("-")[-1].split(" ")[0].replace("seed", "")
        output_dir_gloss = os.path.join(output_dir, gloss)
        os.makedirs(output_dir_gloss, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing video: {video_name}")
        print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}, Frames: {total_frames}")
        print(f"Saving coordinate systems: {[cs.value for cs in coordinate_systems]}")

        # Setup output video writer if needed
        if save_annotated_video:
            output_video_path = os.path.join(output_dir_gloss, f"{video_name}_annotated.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        # Storage for different coordinate systems
        original_landmarks = [] if CoordinateSystem.ORIGINAL in coordinate_systems else None
        shoulder_centered_landmarks = [] if CoordinateSystem.SHOULDER_CENTERED in coordinate_systems else None

        frame_count = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            # Process with MediaPipe Holistic
            results = self.holistic.process(rgb_frame)

            # Extract landmarks based on requested coordinate systems
            if CoordinateSystem.ORIGINAL in coordinate_systems:
                frame_landmarks = self.extract_landmarks(results)
                original_landmarks.append(frame_landmarks)

            if CoordinateSystem.SHOULDER_CENTERED in coordinate_systems:
                unified_landmarks = self.extract_unified_landmarks(results)
                shoulder_centered_landmarks.append(unified_landmarks)

            # Draw landmarks and save annotated video
            if save_annotated_video:
                rgb_frame.flags.writeable = True
                annotated_frame = self.draw_landmarks_on_frame(
                    cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR), results
                )
                out_video.write(annotated_frame)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")

        cap.release()
        if save_annotated_video:
            out_video.release()
            print(f"Annotated video saved to: {output_video_path}")

        # Save datasets for each coordinate system
        if CoordinateSystem.ORIGINAL in coordinate_systems:
            self.save_landmarks_dataset(
                original_landmarks,
                video_name,
                output_dir_gloss,
                coordinate_system="original"
            )

        if CoordinateSystem.SHOULDER_CENTERED in coordinate_systems:
            self.save_landmarks_dataset(
                shoulder_centered_landmarks,
                video_name,
                output_dir_gloss,
                coordinate_system="shoulder_centered"
            )

        return {
            'video_name': video_name,
            'total_frames': frame_count,
            'original_landmarks': original_landmarks,
            'shoulder_centered_landmarks': shoulder_centered_landmarks,
            'fps': fps,
            'resolution': (frame_width, frame_height)
        }

    def save_landmarks_dataset(self,
                               landmarks_data: List,
                               video_name: str,
                               output_dir: str,
                               coordinate_system: str = "original"):
        """
        Save landmarks dataset in NPZ format

        Args:
            landmarks_data: List of frame landmarks
                - For "original": List of Dict with separate pose, face, left_hand, right_hand
                - For "shoulder_centered": List of np.ndarray with unified graph
            video_name: Name of the video
            output_dir: Output directory
            coordinate_system: "original" or "shoulder_centered"
        """
        num_frames = len(landmarks_data)
        suffix = f"_{coordinate_system}" if coordinate_system != "original" else ""

        if coordinate_system == "original":
            # Handle separate graphs (original format)
            pose_size = len(self.pose_landmarks)
            face_size = len(self.face_landmarks)
            hand_size = len(self.hand_landmarks)

            npz_arrays = {}
            included_parts = []

            # Build and add adjacency matrices
            adjacency_matrices = self.build_adjacency_matrices()
            for key, adj_matrix in adjacency_matrices.items():
                npz_arrays[f'adj_{key}'] = adj_matrix

            # Process each body part if included
            if self.include_pose:
                pose_data = []
                for frame_landmarks in landmarks_data:
                    if frame_landmarks is not None and 'pose' in frame_landmarks:
                        pose_data.append(frame_landmarks['pose'])
                    else:
                        pose_data.append(np.zeros((pose_size, 4)))
                npz_arrays['pose'] = np.array(pose_data)
                npz_arrays['pose_landmarks'] = np.array(self.pose_landmarks)
                included_parts.append('pose')

            if self.include_face:
                face_data = []
                for frame_landmarks in landmarks_data:
                    if frame_landmarks is not None and 'face' in frame_landmarks:
                        face_data.append(frame_landmarks['face'])
                    else:
                        face_data.append(np.zeros((face_size, 4)))
                npz_arrays['face'] = np.array(face_data)
                npz_arrays['face_landmarks'] = np.array(self.face_landmarks)
                included_parts.append('face')

            if self.include_hands:
                left_hand_data = []
                right_hand_data = []
                for frame_landmarks in landmarks_data:
                    if frame_landmarks is not None:
                        left_hand_data.append(frame_landmarks.get('left_hand', np.zeros((hand_size, 4))))
                        right_hand_data.append(frame_landmarks.get('right_hand', np.zeros((hand_size, 4))))
                    else:
                        left_hand_data.append(np.zeros((hand_size, 4)))
                        right_hand_data.append(np.zeros((hand_size, 4)))
                npz_arrays['left_hand'] = np.array(left_hand_data)
                npz_arrays['right_hand'] = np.array(right_hand_data)
                npz_arrays['hand_landmarks'] = np.array(self.hand_landmarks)
                included_parts.extend(['left_hand', 'right_hand'])

            # Add metadata as arrays
            npz_arrays['num_frames'] = np.array([num_frames])
            npz_arrays['coordinate_system'] = np.array([coordinate_system], dtype='U20')
            npz_arrays['included_parts'] = np.array(included_parts, dtype='U20')

            # Save as compressed numpy arrays
            np.savez_compressed(
                os.path.join(output_dir, f"{video_name}{suffix}_landmarks.npz"),
                **npz_arrays
            )

        else:  # shoulder_centered - unified graph
            # Handle unified graph format
            unified_data = []

            # Calculate total landmarks from selected body parts
            total_landmarks = 0
            if self.include_pose:
                total_landmarks += len(self.pose_landmarks)
            if self.include_face:
                total_landmarks += len(self.face_landmarks)
            if self.include_hands:
                total_landmarks += 2 * len(self.hand_landmarks)  # left + right

            for frame_landmarks in landmarks_data:
                if frame_landmarks is not None:
                    unified_data.append(frame_landmarks)
                else:
                    # If no landmarks, create zero array with correct size
                    unified_data.append(np.zeros((total_landmarks, 4)))

            unified_data = np.array(unified_data)  # Shape: (num_frames, total_landmarks, 4)

            # Build unified adjacency matrix
            unified_adj_matrix = self.build_unified_adjacency_matrix()

            # Build landmark structure metadata
            included_parts = []
            landmark_indices = []
            landmark_starts = []
            landmark_ends = []
            landmark_counts = []
            current_idx = 0

            if self.include_pose:
                pose_count = len(self.pose_landmarks)
                included_parts.append('pose')
                landmark_indices.extend(self.pose_landmarks)
                landmark_starts.append(current_idx)
                landmark_ends.append(current_idx + pose_count)
                landmark_counts.append(pose_count)
                current_idx += pose_count

            if self.include_face:
                face_count = len(self.face_landmarks)
                included_parts.append('face')
                landmark_indices.extend(self.face_landmarks)
                landmark_starts.append(current_idx)
                landmark_ends.append(current_idx + face_count)
                landmark_counts.append(face_count)
                current_idx += face_count

            if self.include_hands:
                hand_count = len(self.hand_landmarks)
                included_parts.extend(['left_hand', 'right_hand'])
                landmark_indices.extend(self.hand_landmarks)
                landmark_starts.append(current_idx)
                landmark_ends.append(current_idx + hand_count)
                landmark_counts.append(hand_count)
                current_idx += hand_count

                landmark_indices.extend(self.hand_landmarks)
                landmark_starts.append(current_idx)
                landmark_ends.append(current_idx + hand_count)
                landmark_counts.append(hand_count)
                current_idx += hand_count

            # Save as compressed numpy array with all metadata
            np.savez_compressed(
                os.path.join(output_dir, f"{video_name}{suffix}_landmarks.npz"),
                unified_graph=unified_data,
                adjacency_matrix=unified_adj_matrix,
                num_frames=np.array([num_frames]),
                coordinate_system=np.array([coordinate_system], dtype='U20'),
                total_landmarks=np.array([total_landmarks]),
                included_parts=np.array(included_parts, dtype='U20'),
                landmark_indices=np.array(landmark_indices),
                landmark_starts=np.array(landmark_starts),
                landmark_ends=np.array(landmark_ends),
                landmark_counts=np.array(landmark_counts)
            )

        print(f"Dataset ({coordinate_system}) saved for {video_name} with {num_frames} frames")

    def process_folder(self,
                      input_folder: str,
                      output_folder: str,
                      video_extensions: List[str] = ['.mp4', '.avi', '.mov', '.mkv'],
                      coordinate_systems: List[CoordinateSystem] = [CoordinateSystem.ORIGINAL]):
        """
        Process all videos in a folder

        Args:
            input_folder: Input folder containing videos
            output_folder: Output folder for processed data
            video_extensions: List of video file extensions to process
            coordinate_systems: List of coordinate systems to save
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all video files
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(input_path.glob(f"*{ext}")))
            video_files.extend(list(input_path.glob(f"*{ext.upper()}")))

        if not video_files:
            print(f"No video files found in {input_folder}")
            return

        print(f"Found {len(video_files)} video files")
        print(f"Coordinate systems to save: {[cs.value for cs in coordinate_systems]}")

        # Process each video
        for i, video_path in enumerate(video_files):
            print(f"\n--- Processing video {i+1}/{len(video_files)} ---")
            self.process_video(
                str(video_path),
                str(output_path),
                coordinate_systems=coordinate_systems
            )

        print(f"\nAll videos processed! Output saved to: {output_folder}")


# Example usage
if __name__ == "__main__":
    # Configuration
    input_folder = "data/asl_videos"  # Change this to your input folder path
    output_folder = "data/asl_info"  # Change this to your output folder path

    # Create preprocessor instance
    # Example 1: Include all body parts (default)
    preprocessor = VideoPreprocessor(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        include_pose=True,
        include_face=True,
        include_hands=True
    )

    # Process all videos in the folder
    preprocessor.process_folder(input_folder, output_folder, coordinate_systems=[CoordinateSystem.SHOULDER_CENTERED])
