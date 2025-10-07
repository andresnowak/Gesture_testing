import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import pickle


class VideoPreprocessor:
    def __init__(self,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Holistic model
        """
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Define landmark counts for each component (25, 25, 21)
        self.pose_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22] # fmt: skip , [23, 24]
        self.face_landmarks = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 367, 288, 435, 361, 401, 323, 366, 454] # fmt: skip
        self.hand_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] # fmt: skip

    def extract_landmarks(self, results) -> Dict[str, np.ndarray]:
        """
        Extract landmarks from MediaPipe results
        Returns dictionary with pose, face, left_hand, right_hand landmarks
        Each landmark has [x, y, z, visibility] format
        """
        landmarks_data = {}

        # Pose landmarks (33 points)
        if results.pose_landmarks:
            pose_landmarks = []
            for index, landmark in enumerate(results.pose_landmarks.landmark):
                if index in self.pose_landmarks:
                    pose_landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
            landmarks_data['pose'] = np.array(pose_landmarks)
        else:
            landmarks_data['pose'] = np.zeros((len(self.pose_landmarks), 4))

        # Face landmarks (468 points)
        if results.face_landmarks:
            face_landmarks = []
            for index, landmark in enumerate(results.face_landmarks.landmark):
                if index in self.face_landmarks:
                    face_landmarks.append([landmark.x, landmark.y, landmark.z, 1.0])  # Face doesn't have visibility
            landmarks_data['face'] = np.array(face_landmarks)
        else:
            landmarks_data['face'] = np.zeros((len(self.face_landmarks), 4))

        # Left hand landmarks (21 points)
        if results.left_hand_landmarks:
            left_hand_landmarks = []
            for index, landmark in enumerate(results.left_hand_landmarks.landmark):
                if index in self.hand_landmarks:
                    left_hand_landmarks.append([landmark.x, landmark.y, landmark.z, 1.0])  # Hands don't have visibility
            landmarks_data['left_hand'] = np.array(left_hand_landmarks)
        else:
            landmarks_data['left_hand'] = np.zeros((len(self.hand_landmarks), 4))

        # Right hand landmarks (21 points)
        if results.right_hand_landmarks:
            right_hand_landmarks = []
            for index, landmark in enumerate(results.right_hand_landmarks.landmark):
                if index in self.hand_landmarks:
                    right_hand_landmarks.append([landmark.x, landmark.y, landmark.z, 1.0])  # Hands don't have visibility
            landmarks_data['right_hand'] = np.array(right_hand_landmarks)
        else:
            landmarks_data['right_hand'] = np.zeros((len(self.hand_landmarks), 4))

        return landmarks_data

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
                     save_dataset: bool = True) -> Dict:
        """
        Process a single video file
        """
        video_name = Path(video_path).stem
        gloss = video_name.split("-")[-1].split(" ")[0]

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

        # Setup output video writer if needed
        if save_annotated_video:
            output_video_path = os.path.join(output_dir_gloss, f"{video_name}_annotated.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        # Storage for landmarks data
        video_landmarks = []
        frame_count = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            # Process with MediaPipe
            results = self.holistic.process(rgb_frame)

            # Extract landmarks
            frame_landmarks = self.extract_landmarks(results)
            video_landmarks.append(frame_landmarks)

            # Draw landmarks and save annotated video
            if save_annotated_video:
                rgb_frame.flags.writeable = True
                annotated_frame = self.draw_landmarks_on_frame(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR), results)
                out_video.write(annotated_frame)

            frame_count += 1
            if frame_count % 30 == 0:  # Progress update every 30 frames
                print(f"Processed {frame_count}/{total_frames} frames")

        cap.release()
        if save_annotated_video:
            out_video.release()
            print(f"Annotated video saved to: {output_video_path}")

        # Save dataset
        if save_dataset:
            self.save_landmarks_dataset(video_landmarks, video_name, output_dir_gloss)

        return {
            'video_name': video_name,
            'total_frames': frame_count,
            'landmarks_data': video_landmarks,
            'fps': fps,
            'resolution': (frame_width, frame_height)
        }

    def save_landmarks_dataset(self, landmarks_data: List[Dict], video_name: str, output_dir: str):
        """
        Save landmarks dataset in multiple formats
        """
        # Create structured numpy arrays
        num_frames = len(landmarks_data)

        # Initialize arrays for each component
        pose_data = np.zeros((num_frames, len(self.pose_landmarks), 4))
        face_data = np.zeros((num_frames, len(self.face_landmarks), 4))
        left_hand_data = np.zeros((num_frames, len(self.hand_landmarks), 4))
        right_hand_data = np.zeros((num_frames, len(self.hand_landmarks), 4))

        # Fill arrays
        for i, frame_landmarks in enumerate(landmarks_data):
            pose_data[i] = frame_landmarks['pose']
            face_data[i] = frame_landmarks['face']
            left_hand_data[i] = frame_landmarks['left_hand']
            right_hand_data[i] = frame_landmarks['right_hand']

        # Save as numpy arrays
        np.savez_compressed(
            os.path.join(output_dir, f"{video_name}_landmarks.npz"),
            pose=pose_data,
            face=face_data,
            left_hand=left_hand_data,
            right_hand=right_hand_data
        )

        # Save as pickle for easy loading
        dataset = {
            'pose': pose_data,
            'face': face_data,
            'left_hand': left_hand_data,
            'right_hand': right_hand_data,
            'metadata': {
                'num_frames': num_frames,
                'pose_landmarks_count': len(self.pose_landmarks),
                'face_landmarks_count': len(self.face_landmarks),
                'hand_landmarks_count': len(self.hand_landmarks),
            }
        }

        with open(os.path.join(output_dir, f"{video_name}_dataset.pkl"), 'wb') as f:
            pickle.dump(dataset, f)

        print(f"Dataset saved for {video_name} with {num_frames} frames")

    def process_folder(self,
                      input_folder: str,
                      output_folder: str,
                      video_extensions: List[str] = ['.mp4', '.avi', '.mov', '.mkv']):
        """
        Process all videos in a folder
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

        # Process each video
        for i, video_path in enumerate(video_files):
            print(f"\n--- Processing video {i+1}/{len(video_files)} ---")
            self.process_video(str(video_path), str(output_path))

        print(f"\nAll videos processed! Output saved to: {output_folder}")


def main():
    # Configuration
    input_folder = "data/asl_videos"  # Change this to your input folder path
    output_folder = "data/asl_info"  # Change this to your output folder path

    # Create preprocessor instance
    preprocessor = VideoPreprocessor(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Process all videos in the folder
    preprocessor.process_folder(input_folder, output_folder)


if __name__ == '__main__':
    main()
