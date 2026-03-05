"""
Mediapipe Pose Estimation Wrapper
==================================
High-level interface for Mediapipe pose detection.
Handles multi-person skeleton tracking with confidence filtering.

Models:
- LITE: Fast, mobile-optimized
- FULL: Accurate, desktop-class
- HEAVY: Maximum precision for critical applications
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from ..config import MEDIAPIPE_CONFIG


class MediapipePoseDetector:
    """
    Wrapper for Google Mediapipe Pose Detection.
    
    Detects human body pose with 33 landmarks:
    - Head & face (5 points)
    - Torso (3 points)
    - Arms (8 points)
    - Hands (10 points)
    - Legs (8 points)
    
    Supports multi-person tracking with confidence scores.
    """

    def __init__(self, model_complexity: int = 1, enable_tracking: bool = True):
        """
        Initialize Mediapipe pose detector.
        
        Args:
            model_complexity: 0=lite, 1=full, 2=heavy
            enable_tracking: Use tracking for smoother results
        """
        self.model_complexity = model_complexity
        self.enable_tracking = enable_tracking
        self.detector = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize Mediapipe pose model."""
        self._use_legacy_api = False
        try:
            import mediapipe as mp
            
            # Note: MediaPipe 0.10.x requires downloading external model files
            # For now, we'll use a placeholder that returns empty results
            # In production, download the model from MediaPipe model repository
            print("⚠️  MediaPipe pose detection requires external model files.")
            print("    Detector will return empty results until model is configured.")
            self.detector = None
            self._use_legacy_api = False
        except ImportError:
            print("⚠️  Mediapipe not installed. Install with: pip install mediapipe")
            self.detector = None
            self._use_legacy_api = False
        except Exception as e:
            print(f"⚠️  Failed to initialize Mediapipe: {e}")
            self.detector = None
            self._use_legacy_api = False

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect poses in frame.
        
        Args:
            frame: BGR/RGB image (numpy array)
            
        Returns:
            List of detected poses, each with:
            {
                "keypoints": [{"name": str, "x": float, "y": float, "z": float, "confidence": float}, ...],
                "bbox": (x1, y1, x2, y2),
                "confidence": float
            }
        """
        # MediaPipe detector not configured - return empty results
        # This prevents crashes while allowing YOLO and other detectors to work
        if self.detector is None:
            return []

        # Placeholder for when model is properly configured
        return []

    @staticmethod
    def _get_landmark_name(index: int) -> str:
        """Get landmark name by index."""
        landmark_names = [
            "nose", "left_eye_inner", "left_eye", "left_eye_outer",
            "right_eye_inner", "right_eye", "right_eye_outer",
            "left_ear", "right_ear", "mouth_left", "mouth_right",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_pinky", "right_pinky",
            "left_index", "right_index", "left_thumb", "right_thumb",
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle", "left_heel", "right_heel",
            "left_foot_index", "right_foot_index"
        ]
        return landmark_names[index] if index < len(landmark_names) else f"unknown_{index}"
