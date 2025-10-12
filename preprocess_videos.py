import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import pickle
from enum import Enum


# NOTE: We use the normalized landmarks (normalized based on height and width of the image)

class CoordinateSystem(Enum):
    """Enum for different coordinate systems"""
    ORIGINAL = "original"  # Raw MediaPipe normalized coordinates [0,1]
    SHOULDER_CENTERED = "shoulder_centered"  # Unified shoulder-centered coordinates

class VideoPreprocessor:
    def __init__(self,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Holistic model

        Args:
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
        """
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize Holistic model
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Define landmark indices for each component
        self.pose_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        self.face_landmarks = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 367, 288, 435, 361, 401, 323, 366, 454]
        self.hand_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

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
        Returns dictionary with pose, face, left_hand, right_hand landmarks
        Each landmark has [x, y, z, visibility] format
        """
        landmarks_data = {}

        # Pose landmarks
        if results.pose_landmarks:
            pose_landmarks = []
            for index, landmark in enumerate(results.pose_landmarks.landmark):
                if index in self.pose_landmarks:
                    pose_landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
            landmarks_data['pose'] = np.array(pose_landmarks)
        else:
            landmarks_data['pose'] = np.zeros((len(self.pose_landmarks), 4))

        # Face landmarks
        if results.face_landmarks:
            face_landmarks = []
            for index, landmark in enumerate(results.face_landmarks.landmark):
                if index in self.face_landmarks:
                    visibility = self.calculate_visibility_score(landmark)
                    face_landmarks.append([landmark.x, landmark.y, landmark.z, visibility])
            landmarks_data['face'] = np.array(face_landmarks)
        else:
            landmarks_data['face'] = np.zeros((len(self.face_landmarks), 4))

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

    def extract_unified_landmarks(self, results) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract SHOULDER-CENTERED unified landmarks from MediaPipe Holistic
        Returns dictionary with pose, face, left_hand, right_hand in shoulder-centered coordinates
        """
        unified_data = self.unify_normalized_landmarks_to_shoulder_center(results)

        if unified_data is None:
            return None

        unified_landmarks = {}

        # Convert pose to numpy array
        if unified_data['pose']:
            pose_array = []
            for lm in unified_data['pose']:
                pose_array.append([lm['x'], lm['y'], lm['z'], lm.get('visibility', 1.0)])
            unified_landmarks['pose'] = np.array(pose_array)
        else:
            unified_landmarks['pose'] = np.zeros((33, 4))

        # Convert face to numpy array (use original landmarks for visibility calculation)
        if unified_data['face'] and results.face_landmarks:
            face_array = []
            for i, lm in enumerate(unified_data['face']):
                # Calculate visibility from original normalized coordinates
                visibility = self.calculate_visibility_score(results.face_landmarks.landmark[i])
                face_array.append([lm['x'], lm['y'], lm['z'], visibility])
            unified_landmarks['face'] = np.array(face_array)
        else:
            unified_landmarks['face'] = np.zeros((468, 4))

        # Convert left hand to numpy array (use original landmarks for visibility calculation)
        if unified_data['left_hand'] and results.left_hand_landmarks:
            left_hand_array = []
            for i, lm in enumerate(unified_data['left_hand']):
                # Calculate visibility from original normalized coordinates
                visibility = self.calculate_visibility_score(results.left_hand_landmarks.landmark[i])
                left_hand_array.append([lm['x'], lm['y'], lm['z'], visibility])
            unified_landmarks['left_hand'] = np.array(left_hand_array)
        else:
            unified_landmarks['left_hand'] = np.zeros((21, 4))

        # Convert right hand to numpy array (use original landmarks for visibility calculation)
        if unified_data['right_hand'] and results.right_hand_landmarks:
            right_hand_array = []
            for i, lm in enumerate(unified_data['right_hand']):
                # Calculate visibility from original normalized coordinates
                visibility = self.calculate_visibility_score(results.right_hand_landmarks.landmark[i])
                right_hand_array.append([lm['x'], lm['y'], lm['z'], visibility])
            unified_landmarks['right_hand'] = np.array(right_hand_array)
        else:
            unified_landmarks['right_hand'] = np.zeros((21, 4))

        return unified_landmarks

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
                               landmarks_data: List[Dict],
                               video_name: str,
                               output_dir: str,
                               coordinate_system: str = "original"):
        """
        Save landmarks dataset in multiple formats

        Args:
            landmarks_data: List of frame landmarks
            video_name: Name of the video
            output_dir: Output directory
            coordinate_system: "original" or "shoulder_centered"
        """
        num_frames = len(landmarks_data)

        # Determine array sizes based on coordinate system
        if coordinate_system == "original":
            pose_size = len(self.pose_landmarks)
            face_size = len(self.face_landmarks)
            hand_size = len(self.hand_landmarks)
        else:  # shoulder_centered
            pose_size = 33
            face_size = 468
            hand_size = 21

        # Initialize arrays for each component
        pose_data = []
        face_data = []
        left_hand_data = []
        right_hand_data = []

        # Fill arrays
        for frame_landmarks in landmarks_data:
            if frame_landmarks is not None:
                pose_data.append(frame_landmarks.get('pose', np.zeros((pose_size, 4))))
                face_data.append(frame_landmarks.get('face', np.zeros((face_size, 4))))
                left_hand_data.append(frame_landmarks.get('left_hand', np.zeros((hand_size, 4))))
                right_hand_data.append(frame_landmarks.get('right_hand', np.zeros((hand_size, 4))))
            else:
                pose_data.append(np.zeros((pose_size, 4)))
                face_data.append(np.zeros((face_size, 4)))
                left_hand_data.append(np.zeros((hand_size, 4)))
                right_hand_data.append(np.zeros((hand_size, 4)))

        pose_data = np.array(pose_data)
        face_data = np.array(face_data)
        left_hand_data = np.array(left_hand_data)
        right_hand_data = np.array(right_hand_data)

        # Create filename suffix
        suffix = f"_{coordinate_system}" if coordinate_system != "original" else ""

        # Save as numpy arrays
        np.savez_compressed(
            os.path.join(output_dir, f"{video_name}{suffix}_landmarks.npz"),
            pose=pose_data,
            face=face_data,
            left_hand=left_hand_data,
            right_hand=right_hand_data
        )

        # Save as pickle
        dataset = {
            'pose': pose_data,
            'face': face_data,
            'left_hand': left_hand_data,
            'right_hand': right_hand_data,
            'metadata': {
                'num_frames': num_frames,
                'coordinate_system': coordinate_system,
                'pose_landmarks': self.pose_landmarks if coordinate_system == "original" else list(range(33)),
                'face_landmarks': self.face_landmarks if coordinate_system == "original" else list(range(468)),
                'hand_landmarks': self.hand_landmarks if coordinate_system == "original" else list(range(21)),
            }
        }

        with open(os.path.join(output_dir, f"{video_name}{suffix}_dataset.pkl"), 'wb') as f:
            pickle.dump(dataset, f)

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
    preprocessor = VideoPreprocessor(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Process all videos in the folder
    preprocessor.process_folder(input_folder, output_folder, coordinate_systems=[CoordinateSystem.SHOULDER_CENTERED])
