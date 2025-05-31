from dataclasses import dataclass
from datetime import datetime
import os
import csv
import shutil
from typing import Optional, Dict, Any
from pathlib import Path

@dataclass
class DroneImageMetadata:
    """Class for storing drone image metadata."""
    latitude: float
    longitude: float
    altitude: float
    yaw: float
    timestamp: datetime
    filename: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format for CSV storage."""
        return {
            "filename": self.filename,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
            "yaw": self.yaw,
            "timestamp": self.timestamp.isoformat()
        }

class DroneImageManager:
    """Manages drone images and their metadata."""
    
    def __init__(self, base_output_dir: str = "dataset"):
        """Initialize the DroneImageManager.
        
        Args:
            base_output_dir: Base directory for storing images and metadata
        """
        self.base_output_dir = Path(base_output_dir)
        
    def _validate_metadata(self, metadata: DroneImageMetadata) -> None:
        """Validate metadata values.
        
        Args:
            metadata: DroneImageMetadata object to validate
            
        Raises:
            ValueError: If any metadata values are invalid
        """
        if not -90 <= metadata.latitude <= 90:
            raise ValueError("Latitude must be between -90 and 90 degrees")
        if not -180 <= metadata.longitude <= 180:
            raise ValueError("Longitude must be between -180 and 180 degrees")
        if metadata.altitude < 0:
            raise ValueError("Altitude must be positive")
        if not 0 <= metadata.yaw <= 360:
            raise ValueError("Yaw must be between 0 and 360 degrees")

    def save_image(self, 
                  image_path: str, 
                  metadata: DroneImageMetadata,
                  folder_name: Optional[str] = None) -> tuple[Path, Path]:
        """Save an image and its metadata.
        
        Args:
            image_path: Path to the source image
            metadata: DroneImageMetadata object containing image metadata
            folder_name: Optional folder name (defaults to timestamp)
            
        Returns:
            Tuple of (image_path, metadata_path)
            
        Raises:
            FileNotFoundError: If source image doesn't exist
            ValueError: If metadata is invalid
        """
        # Validate input image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Validate metadata
        self._validate_metadata(metadata)
        
        # Create folder with timestamp
        if folder_name is None:
            folder_name = metadata.timestamp.strftime("%Y%m%d_%H%M%S")
        output_dir = self.base_output_dir / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy image
        image_filename = os.path.basename(image_path)
        image_output_path = output_dir / image_filename
        shutil.copy(image_path, image_output_path)
        
        # Save metadata
        metadata_path = output_dir / "metadata.csv"
        file_exists = metadata_path.exists()
        
        with open(metadata_path, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=DroneImageMetadata.__annotations__.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metadata.to_dict())
            
        return image_output_path, metadata_path

    def get_images_by_folder(self, folder_name: str) -> list[tuple[Path, DroneImageMetadata]]:
        """Retrieve all images and their metadata from a specific folder.
        
        Args:
            folder_name: Name of the folder containing images
            
        Returns:
            List of tuples containing (image_path, metadata)
        """
        folder_path = self.base_output_dir / folder_name
        if not folder_path.exists():
            return []
            
        metadata_path = folder_path / "metadata.csv"
        if not metadata_path.exists():
            return []
            
        images = []
        with open(metadata_path, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                image_path = folder_path / row['filename']
                if image_path.exists():
                    metadata = DroneImageMetadata(
                        filename=row['filename'],
                        latitude=float(row['latitude']),
                        longitude=float(row['longitude']),
                        altitude=float(row['altitude']),
                        yaw=float(row['yaw']),
                        timestamp=datetime.fromisoformat(row['timestamp'])
                    )
                    images.append((image_path, metadata))
                    
        return images 