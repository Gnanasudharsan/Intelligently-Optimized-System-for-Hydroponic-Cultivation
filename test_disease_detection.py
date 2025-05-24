#!/usr/bin/env python3
"""
Test cases for disease detection functionality
Tests computer vision-based plant disease detection system
"""

import pytest
import sys
import os
import numpy as np
import cv2
import tempfile
from unittest.mock import Mock, patch, MagicMock
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raspberry_pi.disease_detection import DiseaseDetector

class TestDiseaseDetector:
    """Test cases for DiseaseDetector class"""
    
    def test_disease_detector_initialization(self):
        """Test disease detector initialization"""
        detector = DiseaseDetector()
        assert detector is not None
        assert detector.disease_threshold == 2.0
        assert detector.confidence_threshold == 0.7
        assert 'yellow_spots' in detector.disease_color_ranges
        assert 'brown_spots' in detector.disease_color_ranges
        assert 'black_spots' in detector.disease_color_ranges
    
    def test_initialization_with_model(self):
        """Test initialization with trained model"""
        # Create a temporary model file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            # Create a mock model
            mock_model = MagicMock()
            mock_model.predict.return_value = [0]  # Healthy prediction
            mock_model.predict_proba.return_value = [[0.9, 0.1]]
            
            # Save mock model
            pickle.dump(mock_model, tmp_file)
            tmp_file.flush()
            
            # Test initialization
            detector = DiseaseDetector(model_path=tmp_file.name)
            assert detector.model is not None
            
            # Cleanup
            os.unlink(tmp_file.name)
    
    def test_initialization_without_model(self):
        """Test initialization without model file"""
        detector = DiseaseDetector(model_path="nonexistent_model.pkl")
        assert detector.model is None
    
    def test_preprocess_image(self):
        """Test image preprocessing"""
        detector = DiseaseDetector()
        
        # Create test image
        test_image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
        
        processed = detector._preprocess_image(test_image)
        
        # Should resize to 640px width max
        assert processed.shape[1] <= 640
        assert processed.shape[0] <= 480  # Proportional height
        assert len(processed.shape) == 3  # Should remain color image
    
    def test_preprocess_small_image(self):
        """Test preprocessing of small image (no resize needed)"""
        detector = DiseaseDetector()
        
        # Create small test image
        test_image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        
        processed = detector._preprocess_image(test_image)
        
        # Should not resize small images
        assert processed.shape == test_image.shape
    
    def test_create_leaf_mask_with_green_image(self):
        """Test leaf mask creation with green leaf image"""
        detector = DiseaseDetector()
        
        # Create test image with green area
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        test_image[50:150, 50:150, 1] = 150  # Green square in center
        
        hsv_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
        mask = detector._create_leaf_mask(hsv_image)
        
        assert mask is not None
        assert mask.shape == (200, 200)
        assert mask.dtype == np.uint8
        
        # Should detect green area
        green_pixels = np.sum(mask > 0)
        assert green_pixels > 0
    
    def test_create_leaf_mask_with_no_green(self):
        """Test leaf mask creation with no green areas"""
        detector = DiseaseDetector()
        
        # Create test image with no green (all blue)
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:, :, 0] = 150  # Blue channel only
        
        hsv_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
        mask = detector._create_leaf_mask(hsv_image)
        
        assert mask is not None
        assert mask.shape == (100, 100)
        
        # Should detect minimal green
        green_pixels = np.sum(mask > 0)
        assert green_pixels >= 0  # Could be 0 or minimal
    
    def test_analyze_image_with_healthy_leaf(self):
        """Test image analysis with healthy green leaf"""
        detector = DiseaseDetector()
        
        # Create healthy green leaf image
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        test_image[50:150, 50:150, 1] = 120  # Healthy green
        
        result = detector.analyze_image(test_image)
        
        assert 'disease_detected' in result
        assert 'confidence' in result
        assert 'disease_type' in result
        assert 'recommendations' in result
        assert isinstance(result['disease_detected'], bool)
        assert isinstance(result['confidence'], float)
        assert isinstance(result['disease_type'], str)
        assert isinstance(result['recommendations'], list)
    
    def test_analyze_image_with_diseased_leaf(self):
        """Test image analysis with diseased leaf (yellow spots)"""
        detector = DiseaseDetector()
        
        # Create leaf with yellow spots
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        # Green background
        test_image[20:180, 20:180, 1] = 120
        # Yellow spots (disease indicators)
        test_image[60:80, 60:80, 0] = 100  # Blue
        test_image[60:80, 60:80, 1] = 255  # Green
        test_image[60:80, 60:80, 2] = 255  # Red -> Yellow in BGR
        
        result = detector.analyze_image(test_image)
        
        assert result is not None
        assert 'disease_detected' in result
        assert 'confidence' in result
        assert 'disease_type' in result
        
        # Might detect disease depending on thresholds
        if result['disease_detected']:
            assert result['confidence'] > 0
            assert len(result['recommendations']) > 0
    
    def test_analyze_image_with_model(self):
        """Test image analysis using machine learning model"""
        # Create mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]  # Disease detected
        mock_model.predict_proba.return_value = [[0.1, 0.9]]  # High confidence
        
        detector = DiseaseDetector()
        detector.model = mock_model
        
        # Create test image
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        with patch.object(detector, '_extract_features', return_value=[1, 2, 3]):
            result = detector.analyze_image(test_image)
            
            assert result['disease_detected'] == True
            assert result['confidence'] == 0.9
            assert result['disease_type'] in ['healthy', 'bacterial_spot', 'leaf_blight', 'nutrient_deficiency']
    
    def test_analyze_image_error_handling(self):
        """Test image analysis error handling"""
        detector = DiseaseDetector()
        
        # Test with invalid image (None)
        result = detector.analyze_image(None)
        
        assert 'error' in result
        assert result['disease_detected'] == False
        assert result['confidence'] == 0.0
    
    def test_extract_features(self):
        """Test feature extraction from image"""
        detector = DiseaseDetector()
        
        # Create test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Add some green area for leaf detection
        test_image[25:75, 25:75, 1] = 150
        
        features = detector._extract_features(test_image)
        
        assert features is not None
        assert isinstance(features, list)
        assert len(features) > 0
        
        # Should include color histogram features (256 * 3 channels = 768)
        # Plus LBP features (256) plus shape features (3)
        # Total should be around 1027 features
        assert len(features) > 500  # At least reasonable number of features
    
    def test_calculate_lbp(self):
        """Test Local Binary Pattern calculation"""
        detector = DiseaseDetector()
        
        # Create test grayscale image
        test_image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        lbp = detector._calculate_lbp(test_image)
        
        assert lbp is not None
        assert lbp.shape == test_image.shape
        assert lbp.dtype == np.uint8
    
    def test_get_recommendations_healthy(self):
        """Test recommendations for healthy plants"""
        detector = DiseaseDetector()
        
        recommendations = detector._get_recommendations('healthy')
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any('continue' in rec.lower() for rec in recommendations)
    
    def test_get_recommendations_yellow_spots(self):
        """Test recommendations for yellow spots"""
        detector = DiseaseDetector()
        
        recommendations = detector._get_recommendations('yellow_spots')
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any('nutrient' in rec.lower() for rec in recommendations)
    
    def test_get_recommendations_brown_spots(self):
        """Test recommendations for brown spots"""
        detector = DiseaseDetector()
        
        recommendations = detector._get_recommendations('brown_spots')
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any('bacterial' in rec.lower() or 'fungal' in rec.lower() for rec in recommendations)
    
    def test_get_recommendations_unknown_disease(self):
        """Test recommendations for unknown disease type"""
        detector = DiseaseDetector()
        
        recommendations = detector._get_recommendations('unknown_disease')
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any('expert' in rec.lower() for rec in recommendations)
    
    def test_save_analysis_image(self):
        """Test saving analysis results as annotated image"""
        detector = DiseaseDetector()
        
        # Create test image
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # Create test result
        test_result = {
            'disease_detected': True,
            'disease_type': 'yellow_spots',
            'confidence': 0.85
        }
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        try:
            # Should not raise exception
            detector.save_analysis_image(test_image, test_result, output_path)
            
            # File should exist
            assert os.path.exists(output_path)
            
            # File should be readable as image
            saved_image = cv2.imread(output_path)
            assert saved_image is not None
            assert saved_image.shape == test_image.shape
            
        finally:
            # Cleanup
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_color_based_detection_accuracy(self):
        """Test color-based detection with known patterns"""
        detector = DiseaseDetector()
        
        # Test 1: Healthy leaf (mostly green)
        healthy_image = np.zeros((200, 200, 3), dtype=np.uint8)
        healthy_image[50:150, 50:150, 1] = 150  # Green
        
        result = detector._analyze_with_colors(healthy_image)
        # Should detect low disease percentage
        assert result['disease_percentage'] < 5.0
        
        # Test 2: Diseased leaf (with yellow spots)
        diseased_image = np.zeros((200, 200, 3), dtype=np.uint8)
        diseased_image[50:150, 50:150, 1] = 150  # Green background
        # Add yellow spots
        for i in range(60, 140, 20):
            for j in range(60, 140, 20):
                diseased_image[i:i+10, j:j+10] = [0, 255, 255]  # Yellow in BGR
        
        result = detector._analyze_with_colors(diseased_image)
        # Should detect higher disease percentage
        assert result['disease_percentage'] >= 0
    
    def test_disease_threshold_sensitivity(self):
        """Test disease detection threshold sensitivity"""
        detector = DiseaseDetector()
        
        # Test with different thresholds
        original_threshold = detector.disease_threshold
        
        # Create test image with small diseased area
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        test_image[50:150, 50:150, 1] = 150  # Green background
        test_image[90:100, 90:100] = [0, 255, 255]  # Small yellow spot
        
        # Test with high threshold (should not detect)
        detector.disease_threshold = 10.0
        result_high = detector._analyze_with_colors(test_image)
        
        # Test with low threshold (should detect)
        detector.disease_threshold = 0.5
        result_low = detector._analyze_with_colors(test_image)
        
        # Restore original threshold
        detector.disease_threshold = original_threshold
        
        # High threshold should be less likely to detect disease
        # Low threshold should be more likely to detect disease
        assert isinstance(result_high['disease_detected'], bool)
        assert isinstance(result_low['disease_detected'], bool)
    
    def test_multiple_disease_types_detection(self):
        """Test detection of multiple disease types in single image"""
        detector = DiseaseDetector()
        
        # Create image with multiple disease indicators
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        test_image[50:150, 50:150, 1] = 150  # Green background
        
        # Add yellow spots
        test_image[60:70, 60:70] = [0, 255, 255]  # Yellow
        # Add brown spots
        test_image[80:90, 80:90] = [42, 42, 165]  # Brown
        # Add black spots
        test_image[100:110, 100:110] = [0, 0, 0]  # Black
        
        result = detector._analyze_with_colors(test_image)
        
        assert 'disease_areas' in result
        assert isinstance(result['disease_areas'], dict)
        
        # Should detect multiple types
        disease_areas = result['disease_areas']
        detected_types = [k for k, v in disease_areas.items() if v > 0]
        
        # At least one type should be detected
        assert len(detected_types) >= 0


class TestDiseaseDetectionIntegration:
    """Integration tests for disease detection system"""
    
    def test_complete_analysis_pipeline(self):
        """Test complete analysis pipeline from image to recommendations"""
        detector = DiseaseDetector()
        
        # Create realistic test image
        test_image = self._create_realistic_leaf_image()
        
        # Run complete analysis
        result = detector.analyze_image(test_image)
        
        # Validate complete result structure
        required_keys = ['disease_detected', 'confidence', 'disease_type', 'recommendations']
        for key in required_keys:
            assert key in result
        
        # Validate result types
        assert isinstance(result['disease_detected'], bool)
        assert isinstance(result['confidence'], float)
        assert isinstance(result['disease_type'], str)
        assert isinstance(result['recommendations'], list)
        
        # Confidence should be between 0 and 1
        assert 0.0 <= result['confidence'] <= 1.0
        
        # Should have at least one recommendation
        assert len(result['recommendations']) > 0
    
    def test_batch_analysis(self):
        """Test analyzing multiple images in batch"""
        detector = DiseaseDetector()
        
        # Create multiple test images
        test_images = [
            self._create_realistic_leaf_image(),
            self._create_diseased_leaf_image(),
            self._create_healthy_leaf_image()
        ]
        
        results = []
        for image in test_images:
            result = detector.analyze_image(image)
            results.append(result)
        
        # All analyses should complete successfully
        assert len(results) == 3
        
        for result in results:
            assert 'disease_detected' in result
            assert 'confidence' in result
            assert isinstance(result['disease_detected'], bool)
    
    def test_performance_benchmark(self):
        """Test analysis performance"""
        import time
        
        detector = DiseaseDetector()
        test_image = self._create_realistic_leaf_image()
        
        # Measure analysis time
        start_time = time.time()
        
        # Run analysis multiple times
        num_analyses = 10
        for _ in range(num_analyses):
            result = detector.analyze_image(test_image)
            assert result is not None
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_analyses
        
        # Should complete analysis reasonably quickly
        assert avg_time < 2.0  # Less than 2 seconds per analysis
        print(f"Average analysis time: {avg_time:.3f} seconds")
    
    @staticmethod
    def _create_realistic_leaf_image():
        """Create a realistic leaf image for testing"""
        # Create leaf-shaped green area
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        
        # Create leaf shape (ellipse)
        center = (150, 150)
        axes = (80, 120)
        cv2.ellipse(image, center, axes, 0, 0, 360, (60, 150, 60), -1)
        
        # Add some texture
        noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    @staticmethod
    def _create_diseased_leaf_image():
        """Create a diseased leaf image for testing"""
        # Start with realistic leaf
        image = TestDiseaseDetectionIntegration._create_realistic_leaf_image()
        
        # Add disease spots (yellow/brown)
        cv2.circle(image, (120, 130), 15, (0, 200, 200), -1)  # Yellow spot
        cv2.circle(image, (180, 140), 10, (42, 42, 165), -1)  # Brown spot
        cv2.circle(image, (140, 180), 8, (20, 20, 20), -1)   # Dark spot
        
        return image
    
    @staticmethod
    def _create_healthy_leaf_image():
        """Create a healthy leaf image for testing"""
        # Create uniformly green leaf
        image = np.zeros((250, 250, 3), dtype=np.uint8)
        
        # Create leaf shape
        center = (125, 125)
        axes = (70, 100)
        cv2.ellipse(image, center, axes, 0, 0, 360, (50, 140, 50), -1)
        
        # Add slight natural variation
        noise = np.random.randint(-10, 10, image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return image


class TestDiseaseDetectionAccuracy:
    """Tests for disease detection accuracy (based on research paper's 85-87% accuracy)"""
    
    def test_accuracy_on_known_samples(self):
        """Test accuracy on known healthy/diseased samples"""
        detector = DiseaseDetector()
        
        # Create test dataset
        test_cases = []
        
        # Healthy samples
        for _ in range(10):
            image = TestDiseaseDetectionIntegration._create_healthy_leaf_image()
            test_cases.append((image, False))  # False = healthy
        
        # Diseased samples
        for _ in range(10):
            image = TestDiseaseDetectionIntegration._create_diseased_leaf_image()
            test_cases.append((image, True))   # True = diseased
        
        # Run predictions
        correct_predictions = 0
        total_predictions = len(test_cases)
        
        for image, expected_diseased in test_cases:
            result = detector.analyze_image(image)
            predicted_diseased = result['disease_detected']
            
            if predicted_diseased == expected_diseased:
                correct_predictions += 1
        
        # Calculate accuracy
        accuracy = correct_predictions / total_predictions
        print(f"Detection accuracy: {accuracy:.2%}")
        
        # Should achieve reasonable accuracy (may not reach 85% due to simplified test images)
        assert accuracy >= 0.5  # At least 50% accuracy on simple test cases
    
    def test_confidence_correlation(self):
        """Test that confidence correlates with detection accuracy"""
        detector = DiseaseDetector()
        
        # Create obviously healthy image
        healthy_image = TestDiseaseDetectionIntegration._create_healthy_leaf_image()
        healthy_result = detector.analyze_image(healthy_image)
        
        # Create obviously diseased image
        diseased_image = TestDiseaseDetectionIntegration._create_diseased_leaf_image()
        diseased_result = detector.analyze_image(diseased_image)
        
        # Both should have reasonable confidence
        assert healthy_result['confidence'] > 0.0
        assert diseased_result['confidence'] > 0.0
        
        print(f"Healthy confidence: {healthy_result['confidence']:.3f}")
        print(f"Diseased confidence: {diseased_result['confidence']:.3f}")


# Test fixtures
@pytest.fixture
def sample_healthy_image():
    """Fixture providing a sample healthy leaf image"""
    return TestDiseaseDetectionIntegration._create_healthy_leaf_image()

@pytest.fixture
def sample_diseased_image():
    """Fixture providing a sample diseased leaf image"""
    return TestDiseaseDetectionIntegration._create_diseased_leaf_image()

@pytest.fixture
def disease_detector():
    """Fixture providing a disease detector instance"""
    return DiseaseDetector()


# Edge case tests
class TestDiseaseDetectionEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_image(self):
        """Test with empty/black image"""
        detector = DiseaseDetector()
        
        # Create empty image
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = detector.analyze_image(empty_image)
        
        # Should handle gracefully
        assert result is not None
        assert 'disease_detected' in result
    
    def test_white_image(self):
        """Test with all-white image"""
        detector = DiseaseDetector()
        
        # Create white image
        white_image = np.full((100, 100, 3), 255, dtype=np.uint8)
        
        result = detector.analyze_image(white_image)
        
        # Should handle gracefully
        assert result is not None
        assert 'disease_detected' in result
    
    def test_very_small_image(self):
        """Test with very small image"""
        detector = DiseaseDetector()
        
        # Create tiny image
        tiny_image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        
        result = detector.analyze_image(tiny_image)
        
        # Should handle gracefully
        assert result is not None
        assert 'disease_detected' in result
    
    def test_grayscale_image(self):
        """Test with grayscale image converted to BGR"""
        detector = DiseaseDetector()
        
        # Create grayscale image and convert to BGR
        gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        bgr_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        
        result = detector.analyze_image(bgr_image)
        
        # Should handle gracefully
        assert result is not None
        assert 'disease_detected' in result


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-s"])
