import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from drone_image import DroneImageManager, DroneImageMetadata
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os

@dataclass
class FrameMatch:
    """Class for storing matched frame information."""
    folder1: str
    folder2: str
    image1_path: Path
    image2_path: Path
    metadata1: DroneImageMetadata
    metadata2: DroneImageMetadata
    similarity_score: float

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
            print(f"DEBUG: Attempting to load image: {image_path}")
            if not image_path.exists():
                print(f"DEBUG: Image file does not exist: {image_path}")
                return None
                
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"DEBUG: cv2.imread returned None for: {image_path}")
                return None
                
            print(f"DEBUG: Successfully loaded image: {image_path}")
            print(f"DEBUG: Image shape: {img.shape}")
                
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            return blurred
        except Exception as e:
            print(f"DEBUG: Error processing image {image_path}: {str(e)}")
            return None
            
    def _compute_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute similarity score between two images.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Similarity score between 0 and 1
        """
        # Resize images to same size if needed
        if img1.shape != img2.shape:
            print(f"DEBUG: Resizing images to match shapes: {img1.shape} -> {img2.shape}")
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
        # Compute structural similarity index
        score = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
        similarity = float(np.max(score))
        print(f"DEBUG: Computed similarity score: {similarity:.3f}")
        return similarity
        
    def find_matching_frames(self, 
                           folder1: str, 
                           folder2: str, 
                           similarity_threshold: float = 0.5) -> List[FrameMatch]:
        """Find matching frames between two folders.
        
        Args:
            folder1: First folder name
            folder2: Second folder name
            similarity_threshold: Minimum similarity score to consider frames as matching
            
        Returns:
            List of FrameMatch objects containing matched frames
        """
        # Get images from both folders
        print(f"\nDEBUG: Looking for images in folder 1: {folder1}")
        images1 = self.manager.get_images_by_folder(folder1)
        print(f"DEBUG: Found {len(images1)} images in folder 1")
        
        print(f"\nDEBUG: Looking for images in folder 2: {folder2}")
        images2 = self.manager.get_images_by_folder(folder2)
        print(f"DEBUG: Found {len(images2)} images in folder 2")
        
        if not images1 or not images2:
            print(f"DEBUG: No images found in one or both folders")
            print(f"DEBUG: Folder 1 path: {self.manager.base_output_dir / folder1}")
            print(f"DEBUG: Folder 2 path: {self.manager.base_output_dir / folder2}")
            return []
            
        matches = []
        print(f"\nFinding matching frames between {folder1} and {folder2}...")
        
        # Compare each image from folder1 with each image from folder2
        for img1_path, metadata1 in images1:
            print(f"\nDEBUG: Processing image from folder 1: {img1_path.name}")
            img1 = self._load_image(img1_path)
            if img1 is None:
                continue
                
            best_match = None
            best_score = 0
            
            for img2_path, metadata2 in images2:
                print(f"DEBUG: Comparing with image from folder 2: {img2_path.name}")
                img2 = self._load_image(img2_path)
                if img2 is None:
                    continue
                    
                # Compute similarity
                similarity = self._compute_similarity(img1, img2)
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = (img2_path, metadata2)
                    
            # If we found a good match, add it to our results
            if best_match and best_score >= similarity_threshold:
                img2_path, metadata2 = best_match
                matches.append(FrameMatch(
                    folder1=folder1,
                    folder2=folder2,
                    image1_path=img1_path,
                    image2_path=img2_path,
                    metadata1=metadata1,
                    metadata2=metadata2,
                    similarity_score=best_score
                ))
                print(f"Found match: {img1_path.name} <-> {img2_path.name} (score: {best_score:.2f})")
                
        return matches
        
    def _align_images(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align two images using feature matching.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Tuple of aligned images
        """
        print("DEBUG: Starting image alignment")
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        
        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        print(f"DEBUG: Found {len(kp1)} keypoints in image 1")
        print(f"DEBUG: Found {len(kp2)} keypoints in image 2")
        
        if len(kp1) < 4 or len(kp2) < 4:
            print("DEBUG: Not enough keypoints found for alignment")
            return img1, img2
            
        # FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        try:
            matches = flann.knnMatch(des1, des2, k=2)
        except Exception as e:
            print(f"DEBUG: Error during feature matching: {str(e)}")
            return img1, img2
            
        # Store good matches using Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
                
        print(f"DEBUG: Found {len(good_matches)} good matches")
        
        if len(good_matches) < 10:  # Increased minimum matches threshold
            print("DEBUG: Not enough good matches found for alignment")
            return img1, img2
            
        try:
            # Get matching points
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is None:
                print("DEBUG: Failed to compute homography matrix")
                return img1, img2
                
            # Verify homography matrix
            if not isinstance(H, np.ndarray) or H.shape != (3, 3):
                print("DEBUG: Invalid homography matrix")
                return img1, img2
                
            # Warp image
            h, w = img1.shape
            aligned_img1 = cv2.warpPerspective(img1, H, (w, h))
            
            print("DEBUG: Image alignment completed successfully")
            return aligned_img1, img2
            
        except Exception as e:
            print(f"DEBUG: Error during image alignment: {str(e)}")
            return img1, img2
        
    def detect_changes(self, 
                      matches: List[FrameMatch],
                      threshold: float = 30.0,
                      min_area: int = 100) -> List[Tuple[FrameMatch, np.ndarray]]:
        """Detect changes between matched frames.
        
        Args:
            matches: List of FrameMatch objects
            threshold: Threshold for change detection (default: 30.0)
            min_area: Minimum area of change to consider (default: 100 pixels)
            
        Returns:
            List of tuples containing (FrameMatch, change_mask)
        """
        results = []
        
        for match in matches:
            print(f"\nDEBUG: Processing matched pair: {match.image1_path.name} <-> {match.image2_path.name}")
            # Load and preprocess images
            img1 = self._load_image(match.image1_path)
            img2 = self._load_image(match.image2_path)
            
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
            
            print(f"DEBUG: Found {len(significant_changes)} significant changes")
            
            if significant_changes:
                # Create mask of significant changes
                change_mask = np.zeros_like(thresh)
                cv2.drawContours(change_mask, significant_changes, -1, 255, -1)
                results.append((match, change_mask))
                
        return results
        
    def visualize_changes(self, 
                         results: List[Tuple[FrameMatch, np.ndarray]], 
                         output_dir: str = "change_detection_results"):
        """Visualize detected changes.
        
        Args:
            results: List of change detection results
            output_dir: Directory to save visualization results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\nDEBUG: Saving visualizations to: {output_path}")
        
        for i, (match, change_mask) in enumerate(results):
            print(f"\nDEBUG: Creating visualization {i+1}/{len(results)}")
            # Load original images
            img1 = cv2.imread(str(match.image1_path))
            img2 = cv2.imread(str(match.image2_path))
            
            # Create visualization
            plt.figure(figsize=(15, 5))
            
            # Original images
            plt.subplot(131)
            plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            plt.title(f"Image 1 ({match.folder1})")
            plt.axis('off')
            
            plt.subplot(132)
            plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            plt.title(f"Image 2 ({match.folder2})")
            plt.axis('off')
            
            # Overlay changes on second image
            overlay = img2.copy()
            overlay[change_mask > 0] = [0, 0, 255]  # Red overlay for changes
            cv2.addWeighted(overlay, 0.5, img2, 0.5, 0, overlay)
            
            plt.subplot(133)
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.title(f"Detected Changes (Score: {match.similarity_score:.2f})")
            plt.axis('off')
            
            # Save visualization
            output_file = output_path / f"changes_{i+1}.png"
            plt.savefig(output_file)
            plt.close()
            print(f"DEBUG: Saved visualization to: {output_file}")

def main():
    # Initialize manager and detector
    manager = DroneImageManager()
    detector = ChangeDetector(manager)
    
    print("Change Detection System")
    print("----------------------")
    
    # List available folders
    base_dir = Path("dataset")
    if base_dir.exists():
        print("\nAvailable folders:")
        for folder in base_dir.iterdir():
            if folder.is_dir():
                print(f"- {folder.name}")
    
    # Get folder names
    folder1 = input("\nEnter first folder name: ")
    folder2 = input("Enter second folder name: ")
    
    # Find matching frames
    print("\nFinding matching frames...")
    matches = detector.find_matching_frames(folder1, folder2, similarity_threshold=0.5)
    
    if not matches:
        print("No matching frames found.")
        return
        
    print(f"\nFound {len(matches)} matching frame pairs.")
    
    # Detect changes
    print("\nDetecting changes...")
    results = detector.detect_changes(matches)
    
    if not results:
        print("No significant changes detected.")
        return
        
    print(f"\nFound {len(results)} pairs with significant changes.")
    
    # Visualize results
    print("\nGenerating visualizations...")
    detector.visualize_changes(results)
    print(f"Visualizations saved to 'change_detection_results' directory.")

if __name__ == "__main__":
    main() 