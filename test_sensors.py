#!/usr/bin/env python3
"""
Test cases for sensor functionality
Tests both real and simulated sensor readers
"""

import pytest
import sys
import os
import time
import json
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raspberry_pi.sensor_reader import SensorReader, SimulatedSensorReader

class TestSensorReader:
    """Test cases for SensorReader class"""
    
    def test_sensor_reader_initialization(self):
        """Test sensor reader initialization with mocked serial"""
        with patch('raspberry_pi.sensor_reader.serial.Serial') as mock_serial:
            mock_serial.return_value.is_open = True
            mock_serial.return_value.in_waiting = 0
            
            reader = SensorReader()
            assert reader is not None
            assert reader.port == '/dev/ttyUSB0'
            assert reader.baudrate == 9600
            assert reader.timeout == 5
    
    def test_connection_with_valid_port(self):
        """Test connection with valid serial port"""
        with patch('raspberry_pi.sensor_reader.serial.Serial') as mock_serial:
            mock_serial.return_value.is_open = True
            mock_serial.return_value.in_waiting = 0
            mock_serial.return_value.readline.return_value = b'{"status":"ok"}\n'
            
            reader = SensorReader()
            # Mock the test connection
            with patch.object(reader, '_test_connection', return_value=True):
                result = reader.connect()
                assert result == True
                assert reader.is_connected == True
    
    def test_connection_failure(self):
        """Test connection failure handling"""
        with patch('raspberry_pi.sensor_reader.serial.Serial') as mock_serial:
            mock_serial.side_effect = Exception("Port not found")
            
            reader = SensorReader()
            assert reader.is_connected == False
    
    def test_sensor_data_validation_valid(self):
        """Test sensor data validation with valid data"""
        reader = SensorReader()
        
        valid_data = {
            'ph': 6.5,
            'tds': 800,
            'air_temp': 22.5,
            'humidity': 65.0
        }
        
        assert reader._validate_sensor_data(valid_data) == True
    
    def test_sensor_data_validation_invalid_ph(self):
        """Test sensor data validation with invalid pH"""
        reader = SensorReader()
        
        invalid_data = {
            'ph': 15.0,  # Invalid pH
            'tds': 800,
            'air_temp': 22.5,
            'humidity': 65.0
        }
        
        assert reader._validate_sensor_data(invalid_data) == False
    
    def test_sensor_data_validation_invalid_tds(self):
        """Test sensor data validation with invalid TDS"""
        reader = SensorReader()
        
        invalid_data = {
            'ph': 6.5,
            'tds': 6000,  # Invalid TDS
            'air_temp': 22.5,
            'humidity': 65.0
        }
        
        assert reader._validate_sensor_data(invalid_data) == False
    
    def test_sensor_data_validation_invalid_temperature(self):
        """Test sensor data validation with invalid temperature"""
        reader = SensorReader()
        
        invalid_data = {
            'ph': 6.5,
            'tds': 800,
            'air_temp': 60.0,  # Invalid temperature
            'humidity': 65.0
        }
        
        assert reader._validate_sensor_data(invalid_data) == False
    
    def test_sensor_data_validation_invalid_humidity(self):
        """Test sensor data validation with invalid humidity"""
        reader = SensorReader()
        
        invalid_data = {
            'ph': 6.5,
            'tds': 800,
            'air_temp': 22.5,
            'humidity': 110.0  # Invalid humidity
        }
        
        assert reader._validate_sensor_data(invalid_data) == False
    
    def test_sensor_data_validation_missing_field(self):
        """Test sensor data validation with missing required field"""
        reader = SensorReader()
        
        incomplete_data = {
            'ph': 6.5,
            'tds': 800,
            'air_temp': 22.5
            # Missing humidity
        }
        
        assert reader._validate_sensor_data(incomplete_data) == False
    
    def test_command_sending_success(self):
        """Test successful command sending"""
        with patch('raspberry_pi.sensor_reader.serial.Serial') as mock_serial:
            mock_serial_instance = MagicMock()
            mock_serial.return_value = mock_serial_instance
            mock_serial_instance.write.return_value = None
            mock_serial_instance.flush.return_value = None
            
            reader = SensorReader()
            reader.is_connected = True
            reader.serial_connection = mock_serial_instance
            
            result = reader.send_command("STATUS")
            assert result == True
            mock_serial_instance.write.assert_called_once_with(b"STATUS\n")
    
    def test_command_sending_failure(self):
        """Test command sending failure"""
        with patch('raspberry_pi.sensor_reader.serial.Serial'):
            reader = SensorReader()
            reader.is_connected = False
            
            result = reader.send_command("STATUS")
            assert result == False
    
    def test_read_sensors_with_valid_json(self):
        """Test reading sensors with valid JSON response"""
        with patch('raspberry_pi.sensor_reader.serial.Serial') as mock_serial:
            mock_serial_instance = MagicMock()
            mock_serial.return_value = mock_serial_instance
            
            # Mock valid sensor data
            test_data = {
                'ph': 6.5,
                'tds': 800,
                'air_temp': 22.5,
                'humidity': 65.0,
                'timestamp': 1234567890
            }
            
            mock_serial_instance.in_waiting = 1
            mock_serial_instance.readline.return_value = (json.dumps(test_data) + '\n').encode()
            mock_serial_instance.reset_input_buffer.return_value = None
            
            reader = SensorReader()
            reader.is_connected = True
            reader.serial_connection = mock_serial_instance
            
            with patch.object(reader, '_validate_sensor_data', return_value=True):
                data = reader.read_sensors()
                
                assert data is not None
                assert data['ph'] == 6.5
                assert data['tds'] == 800
                assert data['air_temp'] == 22.5
                assert data['humidity'] == 65.0
    
    def test_read_sensors_with_invalid_json(self):
        """Test reading sensors with invalid JSON response"""
        with patch('raspberry_pi.sensor_reader.serial.Serial') as mock_serial:
            mock_serial_instance = MagicMock()
            mock_serial.return_value = mock_serial_instance
            
            mock_serial_instance.in_waiting = 1
            mock_serial_instance.readline.return_value = b"invalid json\n"
            mock_serial_instance.reset_input_buffer.return_value = None
            
            reader = SensorReader()
            reader.is_connected = True
            reader.serial_connection = mock_serial_instance
            
            # Should timeout and return None or last_data
            with patch('time.time', side_effect=[0, 20]):  # Simulate timeout
                data = reader.read_sensors()
                assert data is None or isinstance(data, dict)
    
    def test_get_system_status(self):
        """Test getting system status"""
        with patch('raspberry_pi.sensor_reader.serial.Serial') as mock_serial:
            mock_serial_instance = MagicMock()
            mock_serial.return_value = mock_serial_instance
            
            status_data = {
                'status': 'running',
                'pump': True,
                'light': False,
                'uptime': 3600
            }
            
            mock_serial_instance.in_waiting = 1
            mock_serial_instance.readline.return_value = (json.dumps(status_data) + '\n').encode()
            
            reader = SensorReader()
            reader.is_connected = True
            reader.serial_connection = mock_serial_instance
            
            with patch.object(reader, 'send_command', return_value=True):
                with patch('time.sleep'):
                    status = reader.get_system_status()
                    
                    assert status is not None
                    assert status['status'] == 'running'
                    assert status['pump'] == True
                    assert status['light'] == False
    
    def test_data_freshness_check(self):
        """Test data freshness checking"""
        reader = SensorReader()
        
        # Test with fresh data
        current_time = time.time() * 1000
        reader.last_data = {
            'ph': 6.5,
            'timestamp': current_time - 10000  # 10 seconds ago
        }
        
        assert reader.is_data_fresh(30) == True
        
        # Test with stale data
        reader.last_data = {
            'ph': 6.5,
            'timestamp': current_time - 60000  # 60 seconds ago
        }
        
        assert reader.is_data_fresh(30) == False
    
    def test_connection_close(self):
        """Test connection closing"""
        with patch('raspberry_pi.sensor_reader.serial.Serial') as mock_serial:
            mock_serial_instance = MagicMock()
            mock_serial.return_value = mock_serial_instance
            mock_serial_instance.is_open = True
            
            reader = SensorReader()
            reader.serial_connection = mock_serial_instance
            reader.is_connected = True
            
            reader.close()
            
            mock_serial_instance.close.assert_called_once()
            assert reader.is_connected == False


class TestSimulatedSensorReader:
    """Test cases for SimulatedSensorReader class"""
    
    def test_simulated_sensor_reader_initialization(self):
        """Test simulated sensor reader initialization"""
        reader = SimulatedSensorReader()
        assert reader.is_connected == True
        assert hasattr(reader, 'last_data')
        assert hasattr(reader, 'data_lock')
    
    def test_simulated_connection(self):
        """Test simulated connection"""
        reader = SimulatedSensorReader()
        result = reader.connect()
        assert result == True
        assert reader.is_connected == True
    
    def test_simulated_sensor_data_generation(self):
        """Test simulated sensor data generation"""
        reader = SimulatedSensorReader()
        data = reader.read_sensors()
        
        assert data is not None
        assert 'ph' in data
        assert 'tds' in data
        assert 'air_temp' in data
        assert 'humidity' in data
        assert 'solution_temp' in data
        assert 'thermocouple_temp' in data
        assert 'timestamp' in data
        
        # Validate realistic ranges
        assert 0 <= data['ph'] <= 14
        assert 0 <= data['tds'] <= 5000
        assert -10 <= data['air_temp'] <= 50
        assert 0 <= data['humidity'] <= 100
        assert isinstance(data['timestamp'], int)
    
    def test_simulated_sensor_data_variation(self):
        """Test that simulated data varies between readings"""
        reader = SimulatedSensorReader()
        
        data1 = reader.read_sensors()
        time.sleep(0.1)
        data2 = reader.read_sensors()
        
        # Data should be different (with high probability)
        assert data1['ph'] != data2['ph'] or data1['tds'] != data2['tds']
    
    def test_simulated_command_sending(self):
        """Test simulated command sending"""
        reader = SimulatedSensorReader()
        
        assert reader.send_command("STATUS") == True
        assert reader.send_command("PUMP_ON") == True
        assert reader.send_command("LIGHT_OFF") == True
    
    def test_simulated_system_status(self):
        """Test simulated system status"""
        reader = SimulatedSensorReader()
        status = reader.get_system_status()
        
        assert status is not None
        assert 'status' in status
        assert 'pump' in status
        assert 'light' in status
        assert 'uptime' in status
        
        assert status['status'] == 'running'
        assert isinstance(status['pump'], bool)
        assert isinstance(status['light'], bool)
        assert isinstance(status['uptime'], int)
    
    def test_simulated_data_persistence(self):
        """Test that simulated data is stored in last_data"""
        reader = SimulatedSensorReader()
        
        data1 = reader.read_sensors()
        last_data = reader.get_last_data()
        
        assert last_data == data1
        assert 'ph' in last_data
        assert 'timestamp' in last_data
    
    def test_simulated_trend_generation(self):
        """Test that simulated data generates realistic trends"""
        reader = SimulatedSensorReader()
        
        # Generate multiple readings to test trend
        readings = []
        for _ in range(10):
            data = reader.read_sensors()
            readings.append(data)
            time.sleep(0.01)
        
        # Check that all readings are valid
        for reading in readings:
            assert 5.0 <= reading['ph'] <= 8.0  # Reasonable pH range
            assert 700 <= reading['tds'] <= 1000  # Reasonable TDS range
            assert 18 <= reading['air_temp'] <= 28  # Reasonable temp range


class TestSensorIntegration:
    """Integration tests for sensor functionality"""
    
    def test_sensor_reader_factory(self):
        """Test creating appropriate sensor reader based on simulation mode"""
        # Test real sensor reader creation (mocked)
        with patch('raspberry_pi.sensor_reader.serial.Serial'):
            reader = SensorReader()
            assert isinstance(reader, SensorReader)
        
        # Test simulated sensor reader creation
        sim_reader = SimulatedSensorReader()
        assert isinstance(sim_reader, SimulatedSensorReader)
    
    def test_sensor_data_pipeline(self):
        """Test complete sensor data pipeline"""
        reader = SimulatedSensorReader()
        
        # Test multiple readings
        for i in range(5):
            data = reader.read_sensors()
            
            assert data is not None
            assert isinstance(data, dict)
            
            # Validate all required fields are present
            required_fields = ['ph', 'tds', 'air_temp', 'humidity']
            for field in required_fields:
                assert field in data
                assert isinstance(data[field], (int, float))
            
            time.sleep(0.1)
    
    def test_error_handling(self):
        """Test error handling in sensor operations"""
        reader = SimulatedSensorReader()
        
        # Test with various edge cases
        assert reader.is_data_fresh(0) == False  # No data yet
        
        # Generate some data first
        reader.read_sensors()
        assert reader.is_data_fresh(60) == True  # Should be fresh
    
    def test_concurrent_access(self):
        """Test concurrent access to sensor data"""
        import threading
        
        reader = SimulatedSensorReader()
        results = []
        errors = []
        
        def read_sensor_data():
            try:
                data = reader.read_sensors()
                results.append(data)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=read_sensor_data)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0  # No errors should occur
        assert len(results) == 5  # All threads should succeed
        
        # Each result should be valid
        for result in results:
            assert result is not None
            assert 'ph' in result


# Test configuration
@pytest.fixture
def mock_serial():
    """Fixture for mocking serial connection"""
    with patch('raspberry_pi.sensor_reader.serial.Serial') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        mock_instance.is_open = True
        mock_instance.in_waiting = 0
        yield mock_instance


@pytest.fixture
def sample_sensor_data():
    """Fixture providing sample sensor data"""
    return {
        'ph': 6.5,
        'tds': 800,
        'air_temp': 22.5,
        'humidity': 65.0,
        'solution_temp': 21.8,
        'thermocouple_temp': 22.1,
        'timestamp': int(time.time() * 1000)
    }


# Performance tests
class TestSensorPerformance:
    """Performance tests for sensor operations"""
    
    def test_sensor_reading_performance(self):
        """Test sensor reading performance"""
        reader = SimulatedSensorReader()
        
        start_time = time.time()
        
        # Read sensors 100 times
        for _ in range(100):
            data = reader.read_sensors()
            assert data is not None
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should complete 100 readings in less than 1 second
        assert elapsed < 1.0
        
        # Average time per reading should be reasonable
        avg_time = elapsed / 100
        assert avg_time < 0.01  # Less than 10ms per reading
    
    def test_data_validation_performance(self):
        """Test data validation performance"""
        reader = SensorReader()
        
        test_data = {
            'ph': 6.5,
            'tds': 800,
            'air_temp': 22.5,
            'humidity': 65.0
        }
        
        start_time = time.time()
        
        # Validate data 1000 times
        for _ in range(1000):
            result = reader._validate_sensor_data(test_data)
            assert result == True
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should complete 1000 validations quickly
        assert elapsed < 0.1  # Less than 100ms


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
