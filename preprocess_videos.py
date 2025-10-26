#!/usr/bin/env python3
"""
Video preprocessing script with command-line arguments.
Processes sign language videos using MediaPipe Holistic to extract landmarks.
"""

import argparse
import sys
import pandas as pd
from pathlib import Path
from src.data_preprocessing.video_preprocessor import VideoPreprocessor, CoordinateSystem


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Process sign language videos to extract pose/face/hand landmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output paths
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to folder containing input videos",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to folder where processed data will be saved",
    )

    # Body part selection
    parser.add_argument(
        "--include_pose",
        action="store_true",
        default=True,
        help="Include pose landmarks in extraction",
    )
    parser.add_argument(
        "--no_pose",
        action="store_false",
        dest="include_pose",
        help="Exclude pose landmarks from extraction",
    )
    parser.add_argument(
        "--include_face",
        action="store_true",
        default=False,
        help="Include face landmarks in extraction",
    )
    parser.add_argument(
        "--no_face",
        action="store_false",
        dest="include_face",
        help="Exclude face landmarks from extraction",
    )
    parser.add_argument(
        "--include_hands",
        action="store_true",
        default=True,
        help="Include hand landmarks in extraction",
    )
    parser.add_argument(
        "--no_hands",
        action="store_false",
        dest="include_hands",
        help="Exclude hand landmarks from extraction",
    )

    # MediaPipe confidence thresholds
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence for MediaPipe (0.0 to 1.0)",
    )
    parser.add_argument(
        "--min_tracking_confidence",
        type=float,
        default=0.5,
        help="Minimum tracking confidence for MediaPipe (0.0 to 1.0)",
    )

    # Coordinate systems
    parser.add_argument(
        "--coordinate_systems",
        type=str,
        nargs="+",
        choices=["original", "shoulder_centered"],
        default=["shoulder_centered"],
        help="Coordinate systems to save. Options: 'original' (raw MediaPipe normalized [0,1]) or 'shoulder_centered' (unified shoulder-centered coordinates)",
    )

    # Video extensions
    parser.add_argument(
        "--video_extensions",
        type=str,
        nargs="+",
        default=[".mp4", ".avi", ".mov", ".mkv"],
        help="Video file extensions to process",
    )

    # Output options
    parser.add_argument(
        "--save_annotated_video",
        action="store_true",
        default=False,
        help="Save annotated videos with landmarks drawn (can be slow and use disk space)",
    )

    # Dataset info JSON
    parser.add_argument(
        "--dataset_json",
        type=str,
        required=True,
        help="Path to dataset JSON file to map video IDs to gloss names (e.g., data/WLASL_v0.3.json)",
    )

    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()

    # Validate input paths
    input_path = Path(args.input_folder)
    if not input_path.exists():
        print(f"Error: Input folder does not exist: {args.input_folder}")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load gloss mapping from JSON
    df_wlasl = pd.read_json(args.dataset_json)
    df_exploded = df_wlasl.explode("instances")
    df_wlasl_info = pd.concat(
        [df_exploded["gloss"], df_exploded["instances"].apply(pd.Series)], axis=1
    )
    gloss_mapping = dict(zip(df_wlasl_info["video_id"].astype(str), df_wlasl_info["gloss"]))
    print(f"Loaded gloss mapping for {len(gloss_mapping)} videos from {args.dataset_json}")

    # Convert coordinate system strings to enum
    coordinate_systems = []
    for cs_str in args.coordinate_systems:
        if cs_str == "original":
            coordinate_systems.append(CoordinateSystem.ORIGINAL)
        elif cs_str == "shoulder_centered":
            coordinate_systems.append(CoordinateSystem.SHOULDER_CENTERED)

    # Create VideoPreprocessor instance
    preprocessor = VideoPreprocessor(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        include_pose=args.include_pose,
        include_face=args.include_face,
        include_hands=args.include_hands,
    )

    # Process all videos in the folder
    try:
        preprocessor.process_folder(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            video_extensions=args.video_extensions,
            coordinate_systems=coordinate_systems,
            gloss_mapping=gloss_mapping,
        )
        print("\n" + "=" * 80)
        print("Processing completed successfully!")
        print("=" * 80)
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
