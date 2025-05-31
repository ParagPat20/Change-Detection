import cv2
import numpy as np
from pathlib import Path
import os

def create_change_detection_video(input_dir: str = "change_detection_results",
                                output_file: str = "change_detection_video.mp4",
                                fps: int = 2,
                                duration_per_frame: float = 2.0):
    """Create a video from change detection output images.
    
    Args:
        input_dir: Directory containing the change detection output images
        output_file: Name of the output video file
        fps: Frames per second for the video
        duration_per_frame: How long to show each frame in seconds
    """
    print(f"Creating video from images in: {input_dir}")
    
    # Get all PNG files from the input directory
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
        
    image_files = sorted(input_path.glob("changes_*.png"))
    if not image_files:
        print(f"Error: No change detection images found in {input_dir}")
        return
        
    print(f"Found {len(image_files)} images")
    
    # Read the first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        print(f"Error: Could not read first image: {image_files[0]}")
        return
        
    height, width = first_image.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"Error: Could not create video file: {output_file}")
        return
        
    # Process each image
    for i, image_file in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {image_file.name}")
        
        # Read image
        frame = cv2.imread(str(image_file))
        if frame is None:
            print(f"Warning: Could not read image: {image_file}")
            continue
            
        # Add frame number and timestamp
        cv2.putText(frame, f"Frame {i+1}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write frame multiple times to control duration
        for _ in range(int(fps * duration_per_frame)):
            video_writer.write(frame)
            
    # Release video writer
    video_writer.release()
    print(f"\nVideo saved to: {output_file}")
    
    # Print video information
    video_path = Path(output_file)
    if video_path.exists():
        size_mb = video_path.stat().st_size / (1024 * 1024)
        print(f"Video size: {size_mb:.1f} MB")
        print(f"Video dimensions: {width}x{height}")
        print(f"Total duration: {len(image_files) * duration_per_frame:.1f} seconds")

def main():
    print("Change Detection Video Creator")
    print("-----------------------------")
    
    # Get input parameters
    input_dir = input("\nEnter input directory (default: change_detection_results): ").strip()
    if not input_dir:
        input_dir = "change_detection_results"
        
    output_file = input("Enter output video file (default: change_detection_video.mp4): ").strip()
    if not output_file:
        output_file = "change_detection_video.mp4"
        
    try:
        fps = int(input("Enter frames per second (default: 2): ").strip() or "2")
        duration = float(input("Enter duration per frame in seconds (default: 2.0): ").strip() or "2.0")
    except ValueError:
        print("Error: Invalid input for fps or duration")
        return
        
    # Create video
    create_change_detection_video(input_dir, output_file, fps, duration)

if __name__ == "__main__":
    main() 