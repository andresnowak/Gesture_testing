def unify_all_normalized_landmarks_to_shoulder_center(pose_results, hand_results, face_results=None):
    """
    Transform all MediaPipe landmarks to shoulder-centered coordinates using NORMALIZED landmarks.
    Shoulder center = midpoint between left shoulder (11) and right shoulder (12)

    x and y coordinates are normalized to [0, 1] based on the image width and height

    IMPORTANT LIMITATIONS:
    - Coordinates are in normalized [0,1] space, NOT meters
    - z-coordinates have different origins (but because they are relative to something in their coordinate system we can still combine them with displacement vectors):
      * Pose z: relative to hips
      * Hand z: relative to wrist
      * Face z: relative to head center
    - z-values are NOT directly comparable across different body parts
    - Distance from camera affects the normalized coordinates
    - Use world landmarks if you need true 3D metric coordinates (but face doesn't include this)
    """
    if not pose_results.pose_landmarks:
        return None

    pose_norm = pose_results.pose_landmarks.landmark

    # Calculate shoulder center in NORMALIZED space (NEW ORIGIN)
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

    # Transform pose to shoulder-centered (normalized space)
    # NOTE: z is relative to the hips position
    for landmark in pose_norm:
        unified_data['pose'].append({
            'x': landmark.x - shoulder_center['x'],
            'y': landmark.y - shoulder_center['y'],
            'z': landmark.z - shoulder_center['z']
        })

    # Get shoulder-centered anchor points
    nose_shoulder_centered = unified_data['pose'][0]
    left_wrist_shoulder_centered = unified_data['pose'][15]
    right_wrist_shoulder_centered = unified_data['pose'][16]

    # Transform face using nose as anchor
    if face_results and face_results.multi_face_landmarks:
        face_norm = face_results.multi_face_landmarks[0].landmark  # First face
        nose_face = face_norm[1]  # Nose tip in face mesh

        for landmark in face_norm:
            # Get face landmark relative to face nose (get the displacement vector from landmark to nose)
            # NOTE: z here is relative to HEAD CENTER
            offset_x = landmark.x - nose_face.x
            offset_y = landmark.y - nose_face.y
            offset_z = landmark.z - nose_face.z

            # Add the offset to the shoulder-centered nose position
            unified_data['face'].append({
                'x': nose_shoulder_centered['x'] + offset_x,
                'y': nose_shoulder_centered['y'] + offset_y,
                'z': nose_shoulder_centered['z'] + offset_z
            })

    # Transform left and right hand using wrist as anchor
    if hand_results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            handedness = hand_results.multi_handedness[idx].classification[0].label

            if handedness == "Left":
                left_hand_norm = hand_landmarks.landmark
                left_wrist_hand = left_hand_norm[0]

                for landmark in left_hand_norm:
                    # Calculate displacement vector in hand's normalized space
                    # NOTE: z here is relative to WRIST
                    offset_x = landmark.x - left_wrist_hand.x
                    offset_y = landmark.y - left_wrist_hand.y
                    offset_z = landmark.z - left_wrist_hand.z

                    # Apply offset to shoulder-centered wrist position
                    unified_data['left_hand'].append({
                        'x': left_wrist_shoulder_centered['x'] + offset_x,
                        'y': left_wrist_shoulder_centered['y'] + offset_y,
                        'z': left_wrist_shoulder_centered['z'] + offset_z
                    })

            if handedness == "Right":
                right_hand_norm = hand_landmarks.landmark
                right_wrist_hand = right_hand_norm[0]

                for landmark in right_hand_norm:
                    # Calculate displacement vector in hand's normalized space

                    offset_x = landmark.x - right_wrist_hand.x
                    offset_y = landmark.y - right_wrist_hand.y
                    offset_z = landmark.z - right_wrist_hand.z

                    # Apply offset to shoulder-centered wrist position
                    unified_data['right_hand'].append({
                        'x': right_wrist_shoulder_centered['x'] + offset_x,
                        'y': right_wrist_shoulder_centered['y'] + offset_y,
                        'z': right_wrist_shoulder_centered['z'] + offset_z
                    })


    return unified_data

def unify_all_world_landmarks_to_shoulder_center(pose_results, hand_results):
    """
    Transform all MediaPipe Holistic landmarks to shoulder-centered coordinates.
    Shoulder center = midpoint between left shoulder (11) and right shoulder (12)

    NOTE: Face doesn't include world landmarks
    """
    if not pose_results.pose_world_landmarks or not hand_results.pose_world_landmarks:
        return None

    pose_world = pose_results.pose_world_landmarks.landmark

    # Calculate shoulder center (NEW ORIGIN)
    left_shoulder = pose_world[11]
    right_shoulder = pose_world[12]

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
    for landmark in pose_world:
        unified_data['pose'].append({
            'x': landmark.x - shoulder_center['x'],
            'y': landmark.y - shoulder_center['y'],
            'z': landmark.z - shoulder_center['z']
        })

    # Get shoulder-centered anchor points
    left_wrist_shoulder_centered = unified_data['pose'][15]
    right_wrist_shoulder_centered = unified_data['pose'][16]


    # Transform left hand using wrist as anchor
    if hand_results.left_hand_world_landmarks:
        left_hand_world = hand_results.left_hand_world_landmarks.landmark
        left_wrist_hand = left_hand_world[0]

        for landmark in left_hand_world:
            # First we calculate a displacement vector to the wrist (from the hands coordinate system)
            offset_x = landmark.x - left_wrist_hand.x
            offset_y = landmark.y - left_wrist_hand.y
            offset_z = landmark.z - left_wrist_hand.z

            # Then using this displacement we now use it as an offset to calculate the coordinate of the point in the pose coordinate system

            unified_data['left_hand'].append({
                'x': left_wrist_shoulder_centered['x'] + offset_x,
                'y': left_wrist_shoulder_centered['y'] + offset_y,
                'z': left_wrist_shoulder_centered['z'] + offset_z
            })

    # Transform right hand using wrist as anchor
    if hand_results.right_hand_world_landmarks:
        right_hand_world = hand_results.right_hand_world_landmarks.landmark
        right_wrist_hand = right_hand_world[0]

        for landmark in right_hand_world:
            offset_x = landmark.x - right_wrist_hand.x
            offset_y = landmark.y - right_wrist_hand.y
            offset_z = landmark.z - right_wrist_hand.z

            unified_data['right_hand'].append({
                'x': right_wrist_shoulder_centered['x'] + offset_x,
                'y': right_wrist_shoulder_centered['y'] + offset_y,
                'z': right_wrist_shoulder_centered['z'] + offset_z
            })

    return unified_data
