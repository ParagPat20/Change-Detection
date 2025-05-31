import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from drone_image import DroneImageManager, DroneImageMetadata
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

class ChangeDetector:
    def __init__(self, manager: DroneImageManager):
        """Initialize the change detector.
        
        Args:
            manager: DroneImageManager instance for accessing images and metadata
        """
        self.manager = manager
        
    def _load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Load and preprocess an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array or None if loading fails
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Error: Could not load image {image_path}")
                return None
                
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            return blurred
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
            
    def _align_images(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align two images using feature matching.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Tuple of aligned images
        """
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        
        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        # FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Store good matches using Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
                
        if len(good_matches) < 4:
            print("Warning: Not enough good matches found for alignment")
            return img1, img2
            
        # Get matching points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Warp image
        h, w = img1.shape
        aligned_img1 = cv2.warpPerspective(img1, H, (w, h))
        
        return aligned_img1, img2
        
    def detect_changes(self, 
                      date1: str, 
                      date2: str, 
                      threshold: float = 30.0,
                      min_area: int = 100) -> List[Tuple[Path, Path, np.ndarray]]:
        """Detect changes between images from two different dates.
        
        Args:
            date1: First date (YYYY-MM-DD)
            date2: Second date (YYYY-MM-DD)
            threshold: Threshold for change detection (default: 30.0)
            min_area: Minimum area of change to consider (default: 100 pixels)
            
        Returns:
            List of tuples containing (image1_path, image2_path, change_mask)
        """
        # Get images from both dates
        images1 = self.manager.get_images_by_date(date1)
        images2 = self.manager.get_images_by_date(date2)
        
        if not images1 or not images2:
            print(f"Error: No images found for one or both dates")
            return []
            
        results = []
        
        # Compare each image from date1 with each image from date2
        for img1_path, metadata1 in images1:
            for img2_path, metadata2 in images2:
                # Load and preprocess images
                img1 = self._load_image(img1_path)
                img2 = self._load_image(img2_path)
                
                if img1 is None or img2 is None:
                    continue
                    
                # Align images
                aligned_img1, aligned_img2 = self._align_images(img1, img2)
                
                # Calculate absolute difference
                diff = cv2.absdiff(aligned_img1, aligned_img2)
                
                # Apply threshold
                _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
                
                # Find contours of changes
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter small changes
                significant_changes = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
                
                if significant_changes:
                    # Create mask of significant changes
                    change_mask = np.zeros_like(thresh)
                    cv2.drawContours(change_mask, significant_changes, -1, 255, -1)
                    results.append((img1_path, img2_path, change_mask))
                    
        return results
        
    def visualize_changes(self, 
                         results: List[Tuple[Path, Path, np.ndarray]], 
                         output_dir: str = "change_detection_results"):
        """Visualize detected changes.
        
        Args:
            results: List of change detection results
            output_dir: Directory to save visualization results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for i, (img1_path, img2_path, change_mask) in enumerate(results):
            # Load original images
            img1 = cv2.imread(str(img1_path))
            img2 = cv2.imread(str(img2_path))
            
            # Create visualization
            plt.figure(figsize=(15, 5))
            
            # Original images
            plt.subplot(131)
            plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            plt.title("Image 1")
            plt.axis('off')
            
            plt.subplot(132)
            plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            plt.title("Image 2")
            plt.axis('off')
            
            # Overlay changes on second image
            overlay = img2.copy()
            overlay[change_mask > 0] = [0, 0, 255]  # Red overlay for changes
            cv2.addWeighted(overlay, 0.5, img2, 0.5, 0, overlay)
            
            plt.subplot(133)
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.title("Detected Changes")
            plt.axis('off')
            
            # Save visualization
            plt.savefig(output_path / f"changes_{i+1}.png")
            plt.close()

def main():
    # Initialize manager and detector
    manager = DroneImageManager()
    detector = ChangeDetector(manager)
    
    print("Change Detection System")
    print("----------------------")
    
    # Get dates from user
    date1 = input("Enter first date (YYYY-MM-DD): ")
    date2 = input("Enter second date (YYYY-MM-DD): ")
    
    # Detect changes
    print("\nDetecting changes...")
    results = detector.detect_changes(date1, date2)
    
    if not results:
        print("No significant changes detected.")
        return
        
    print(f"\nFound {len(results)} pairs of images with significant changes.")
    
    # Visualize results
    print("\nGenerating visualizations...")
    detector.visualize_changes(results)
    print(f"Visualizations saved to 'change_detection_results' directory.")

if __name__ == "__main__":
    main() 