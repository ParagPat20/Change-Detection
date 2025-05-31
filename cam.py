import cv2
import numpy as np
from datetime import datetime
from drone_image import DroneImageManager, DroneImageMetadata
import time
import os
from pathlib import Path

def initialize_camera(camera_id: int = 1, warmup_frames: int = 30) -> tuple[cv2.VideoCapture, bool]:
    """Initialize camera and wait for it to warm up.
    
    Args:
        camera_id: Camera device ID
        warmup_frames: Number of frames to wait for camera to stabilize
        
    Returns:
        Tuple of (camera object, success boolean)
    """
    print("Initializing camera...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return None, False
        
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Full HD width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Full HD height
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
    
    # Wait for camera to warm up
    print("Waiting for camera to warm up...")
    for i in range(warmup_frames):
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read from camera during warm-up")
            cap.release()
            return None, False
        time.sleep(0.1)  # Small delay between frames
        
    print("Camera initialized successfully!")
    return cap, True

def capture_image(cap: cv2.VideoCapture) -> tuple[np.ndarray, bool]:
    """Capture an image from the initialized camera.
    
    Args:
        cap: Initialized camera object
        
    Returns:
        Tuple of (image array, success boolean)
    """
    if cap is None:
        return None, False
        
    # Capture multiple frames and use the last one
    for _ in range(3):  # Capture 3 frames to ensure we get a good one
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            return None, False
        time.sleep(0.1)  # Small delay between captures
        
    return frame, True

def capture_batch(cap: cv2.VideoCapture, num_images: int, delay: float = 2.0) -> list[tuple[np.ndarray, datetime]]:
    """Capture a batch of images with timestamps.
    
    Args:
        cap: Initialized camera object
        num_images: Number of images to capture
        delay: Delay between captures in seconds
        
    Returns:
        List of tuples containing (image array, timestamp)
    """
    images = []
    for i in range(num_images):
        print(f"\nCapturing image {i+1}/{num_images}...")
        frame, success = capture_image(cap)
        if success:
            timestamp = datetime.utcnow()
            images.append((frame, timestamp))
            if i < num_images - 1:  # Don't wait after the last image
                time.sleep(delay)
    return images

def main():
    # Initialize camera and image manager
    camera_id = 1  # Change this if your camera has a different ID
    manager = DroneImageManager()
    
    # Initialize camera
    cap, success = initialize_camera(camera_id)
    if not success:
        return
        
    print("\nDrone Image Capture System")
    print("-------------------------")
    print("1. Capture batch of images")
    print("2. Exit")
    
    try:
        while True:
            choice = input("\nEnter your choice (1-2): ").strip()
            
            if choice == "2":
                break
            elif choice == "1":
                try:
                    # Get session ID
                    session_id = input("\nEnter session ID (e.g., test1, dev2): ").strip()
                    if not session_id:
                        print("Error: Session ID cannot be empty")
                        continue
                        
                    # Get number of images to capture
                    num_images = int(input("Enter number of images to capture: "))
                    if num_images <= 0:
                        print("Error: Number of images must be positive")
                        continue
                        
                    # Get delay between captures
                    delay = float(input("Enter delay between captures (seconds): "))
                    if delay < 0:
                        print("Error: Delay must be non-negative")
                        continue
                        
                    # Capture batch of images
                    images = capture_batch(cap, num_images, delay)
                    
                    if not images:
                        print("No images were captured successfully")
                        continue
                        
                    # Get metadata for the batch
                    print("\nEnter metadata for all images:")
                    try:
                        latitude = float(input("Latitude (-90 to 90): "))
                        longitude = float(input("Longitude (-180 to 180): "))
                        altitude = float(input("Altitude (meters): "))
                        yaw = float(input("Yaw (0 to 360 degrees): "))
                    except ValueError as e:
                        print(f"Error: Invalid metadata input - {e}")
                        continue
                        
                    # Create folder name with session ID and timestamp
                    timestamp = datetime.utcnow()
                    folder_name = f"{session_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
                    
                    # Save all images with metadata
                    print(f"\nSaving images and metadata to folder: {folder_name}")
                    for i, (frame, timestamp) in enumerate(images):
                        # Generate filename based on timestamp
                        filename = f"drone_image_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                        
                        # Save the captured frame
                        temp_path = f"temp_capture_{i}.jpg"
                        cv2.imwrite(temp_path, frame)
                        
                        # Create metadata object
                        metadata = DroneImageMetadata(
                            filename=filename,
                            latitude=latitude,
                            longitude=longitude,
                            altitude=altitude,
                            yaw=yaw,
                            timestamp=timestamp
                        )
                        
                        # Save image and metadata
                        image_path, metadata_path = manager.save_image(temp_path, metadata, folder_name)
                        print(f"Saved image {i+1}/{len(images)}: {image_path}")
                        
                        # Clean up temporary file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                            
                    print(f"\nSuccessfully saved {len(images)} images and metadata")
                    
                except ValueError as e:
                    print(f"Error: Invalid input - {e}")
            else:
                print("Invalid choice. Please enter 1 or 2.")
    finally:
        # Release camera
        if cap is not None:
            cap.release()
            print("\nCamera released.")

if __name__ == "__main__":
    main()
