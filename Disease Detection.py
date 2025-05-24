#!/usr/bin/env python3
"""
Disease Detection Module for Hydroponic System
Uses OpenCV for computer vision-based plant disease detection

Based on research achieving 85-87% accuracy with Amaranthus caudatus
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Tuple, List
import pickle
import os

logger = logging.getLogger(__name__)

class DiseaseDetector:
    """Computer vision-based plant disease detector"""
    
    def __init__(self, model_path: str = None):
        """Initialize the disease detector"""
        self.model_path = model_path
        self.model = None
        
        # Disease thresholds and parameters
        self.disease_threshold = 2.0  # Percentage of diseased area
        self.confidence_threshold = 0.7
        
        # Color ranges for disease detection (HSV)
        self.disease_color_ranges = {
            'yellow_spots': {
                'lower': np.array([20, 100, 100]),
                'upper': np.array([30, 255, 255])
            },
            'brown_spots': {
                'lower': np.array([10, 50, 20]),
                'upper': np.array([20, 255, 200])
            },
            'black_spots': {
                'lower': np.array([0, 0, 0]),
                'upper': np.array([180, 255, 50])
            }
        }
        
        if model_path and os.path.exists(model_path):
            self._load_model()
        else:
            logger.warning("No trained model found, using basic color-based detection")
    
    def _load_model(self):
        """Load the trained machine learning model"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Disease detection model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze an image for plant diseases
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            if self.model:
                # Use trained model if available
                result = self._analyze_with_model(processed_image)
            else:
                # Use color-based detection as fallback
                result = self._analyze_with_colors(processed_image)
            
            # Add metadata
            result['image_shape'] = image.shape
            result['analysis_method'] = 'model' if self.model else 'color_based'
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {
                'disease_detected': False,
                'confidence': 0.0,
                'disease_type': 'unknown',
                'error': str(e)
            }
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for analysis"""
        # Resize image for consistency
        height, width = image.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Apply Gaussian blur to reduce noise
        image = cv2.GaussianBlur(image, (5, 5), 0)
        
        return image
    
    def _analyze_with_model(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image using trained machine learning model"""
        try:
            # Extract features from image
            features = self._extract_features(image)
            
            # Make prediction
            prediction = self.model.predict([features])[0]
            confidence = self.model.predict_proba([features])[0].max()
            
            disease_types = ['healthy', 'bacterial_spot', 'leaf_blight', 'nutrient_deficiency']
            disease_type = disease_types[prediction]
            
            return {
                'disease_detected': prediction != 0,
                'confidence': confidence,
                'disease_type': disease_type,
                'recommendations': self._get_recommendations(disease_type)
            }
            
        except Exception as e:
            logger.error(f"Error in model-based analysis: {e}")
            return self._analyze_with_colors(image)
    
    def _analyze_with_colors(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image using color-based disease detection
        Based on the research paper's approach using channel separation
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Separate RGB channels as mentioned in paper
        b, g, r = cv2.split(image)
        
        # Create leaf mask (green areas)
        leaf_mask = self._create_leaf_mask(hsv)
        
        if np.sum(leaf_mask) == 0:
            return {
                'disease_detected': False,
                'confidence': 0.0,
                'disease_type': 'no_leaf_detected',
                'recommendations': ['Ensure proper camera positioning']
            }
        
        # Detect diseased areas
        disease_areas = {}
        total_diseased_pixels = 0
        
        for disease_name, color_range in self.disease_color_ranges.items():
            mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
            mask = cv2.bitwise_and(mask, leaf_mask)  # Only within leaf area
            
            diseased_pixels = np.sum(mask > 0)
            disease_areas[disease_name] = diseased_pixels
            total_diseased_pixels += diseased_pixels
        
        # Calculate disease percentage
        total_leaf_pixels = np.sum(leaf_mask > 0)
        disease_percentage = (total_diseased_pixels / total_leaf_pixels) * 100
        
        # Determine if disease is detected
        disease_detected = disease_percentage > self.disease_threshold
        
        # Find dominant disease type
        if disease_detected:
            dominant_disease = max(disease_areas, key=disease_areas.get)
            confidence = min(disease_percentage / 10.0, 1.0)  # Normalize to 0-1
        else:
            dominant_disease = 'healthy'
            confidence = 1.0 - (disease_percentage / self.disease_threshold)
        
        return {
            'disease_detected': disease_detected,
            'confidence': confidence,
            'disease_type': dominant_disease,
            'disease_percentage': disease_percentage,
            'disease_areas': disease_areas,
            'recommendations': self._get_recommendations(dominant_disease)
        }
    
    def _create_leaf_mask(self, hsv_image: np.ndarray) -> np.ndarray:
        """Create a mask for leaf areas (green regions)"""
        # Define green color range for leaves
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Create initial mask
        mask = cv2.inRange(hsv_image, lower_green, upper_green)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Remove small noise
        mask = cv2.medianBlur(mask, 5)
        
        return mask
    
    def _extract_features(self, image: np.ndarray) -> List[float]:
        """Extract features from image for machine learning model"""
        features = []
        
        # Color histogram features
        for channel in cv2.split(image):
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            features.extend(hist.flatten())
        
        # Texture features using Local Binary Pattern
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate LBP
        lbp = self._calculate_lbp(gray)
        lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        features.extend(lbp_hist.flatten())
        
        # Shape features
        contours, _ = cv2.findContours(
            self._create_leaf_mask(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            features.extend([
                area,
                perimeter,
                area / (perimeter ** 2) if perimeter > 0 else 0  # Compactness
            ])
        else:
            features.extend([0, 0, 0])
        
        return features
    
    def _calculate_lbp(self, gray_image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """Calculate Local Binary Pattern for texture analysis"""
        height, width = gray_image.shape
        lbp = np.zeros((height, width), dtype=np.uint8)
        
        for i in range(radius, height - radius):
            for j in range(radius, width - radius):
                center = gray_image[i, j]
                code = 0
                
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(j + radius * np.cos(angle))
                    y = int(i - radius * np.sin(angle))
                    
                    if gray_image[y, x] >= center:
                        code |= (1 << k)
                
                lbp[i, j] = code
        
        return lbp
    
    def _get_recommendations(self, disease_type: str) -> List[str]:
        """Get treatment recommendations based on disease type"""
        recommendations = {
            'healthy': [
                'Continue current care routine',
                'Monitor regularly for any changes'
            ],
            'yellow_spots': [
                'Check for nutrient deficiency (nitrogen)',
                'Adjust pH levels (6.0-6.5)',
                'Ensure proper lighting',
                'Remove affected leaves'
            ],
            'brown_spots': [
                'Possible bacterial or fungal infection',
                'Improve air circulation',
                'Reduce humidity if too high',
                'Consider organic fungicide treatment',
                'Remove severely affected leaves'
            ],
            'black_spots': [
                'Serious fungal infection likely',
                'Isolate affected plants',
                'Improve ventilation immediately',
                'Apply appropriate fungicide',
                'Monitor other plants closely'
            ],
            'bacterial_spot': [
                'Bacterial infection detected',
                'Remove affected leaves immediately',
                'Improve air circulation',
                'Reduce leaf wetness',
                'Consider copper-based bactericide'
            ],
            'leaf_blight': [
                'Fungal leaf blight detected',
                'Remove affected foliage',
                'Improve air circulation',
                'Apply fungicide treatment',
                'Monitor environmental conditions'
            ],
            'nutrient_deficiency': [
                'Nutrient deficiency detected',
                'Check and adjust nutrient solution',
                'Verify pH levels (5.5-6.5)',
                'Ensure proper EC levels',
                'Consider foliar feeding'
            ]
        }
        
        return recommendations.get(disease_type, ['Consult plant pathology expert'])
    
    def save_analysis_image(self, image: np.ndarray, result: Dict[str, Any], 
                           output_path: str):
        """Save analysis results as annotated image"""
        try:
            # Create a copy for annotation
            annotated = image.copy()
            
            # Add text overlay with results
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            color = (0, 255, 0) if not result['disease_detected'] else (0, 0, 255)
            thickness = 2
            
            # Disease status
            status = "HEALTHY" if not result['disease_detected'] else "DISEASE DETECTED"
            cv2.putText(annotated, status, (10, 30), font, font_scale, color, thickness)
            
            # Disease type and confidence
            if result['disease_detected']:
                text = f"Type: {result['disease_type']}"
                cv2.putText(annotated, text, (10, 60), font, font_scale, color, thickness)
                
                conf_text = f"Confidence: {result['confidence']:.2f}"
                cv2.putText(annotated, conf_text, (10, 90), font, font_scale, color, thickness)
            
            # Save annotated image
            cv2.imwrite(output_path, annotated)
            logger.info(f"Analysis image saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving analysis image: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize detector
    detector = DiseaseDetector()
    
    # Load test image
    test_image = cv2.imread('test_images/plant_sample.jpg')
    
    if test_image is not None:
        # Analyze image
        result = detector.analyze_image(test_image)
        
        print("Disease Detection Results:")
        print(f"Disease Detected: {result['disease_detected']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Disease Type: {result['disease_type']}")
        print(f"Recommendations: {result['recommendations']}")
        
        # Save annotated result
        detector.save_analysis_image(test_image, result, 'analysis_result.jpg')
    else:
        print("Test image not found")
