#!/usr/bin/env python3
"""
Test cases for actuator functionality
Tests pump control, lighting, and safety systems
"""

import pytest
import sys
import os
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import threading

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raspberry_pi.actuator_controller import ActuatorController, SimulatedActuatorController
from raspberry_pi.sensor_reader import SimulatedSensorReader

class TestActuatorController:
    """Test cases for ActuatorController class"""
    
    def test_actuator_controller_initialization(self):
        """Test actuator controller initialization"""
        sensor_reader = SimulatedSensorReader()
        controller = ActuatorController(sensor_reader)
        
        assert controller is not None
        assert controller.sensor_reader == sensor_reader
        assert isinstance(controller.actuator_states, dict)
        assert isinstance(controller.pump_timers, dict)
        assert isinstance(controller.pump_limits, dict)
        
        # Check initial actuator states
        expected_actuators = [
            'water_pump', 'grow_lights', 'ph_up_pump', 
            'ph_down_pump', 'nutrient_pump', 'exhaust_fan', 'circulation_fan'
        ]
        
        for actuator in expected_actuators:
            assert actuator in controller.actuator_states
            assert controller.actuator_states[actuator] == False
    
    def test_pump_limits_configuration(self):
        """Test pump safety limits configuration"""
        sensor_reader = SimulatedSensorReader()
        controller = ActuatorController(sensor_reader)
        
        # Check pump limits are properly configured
        assert 'ph_up_pump' in controller.pump_limits
        assert 'ph_down_pump' in controller.pump_limits
        assert 'nutrient_pump' in controller.pump_limits
        assert 'water_pump' in controller.pump_limits
        
        # Check limit structure
        for pump, limits in controller.pump_limits.items():
            assert 'max_duration' in limits
            assert 'cooldown' in limits
            assert isinstance(limits['max_duration'], (int, float))
            assert isinstance(limits['cooldown'], (int, float))
            assert limits['max_duration'] > 0
            assert limits['cooldown'] > 0
    
    def test_turn_on_pump(self):
        """Test turning on water pump"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        result = controller.turn_on_pump()
        
        assert result == True
        assert controller.actuator_states['water_pump'] == True
    
    def test_turn_off_pump(self):
        """Test turning off water pump"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # First turn on
        controller.turn_on_pump()
        assert controller.actuator_states['water_pump'] == True
        
        # Then turn off
        result = controller.turn_off_pump()
        
        assert result == True
        assert controller.actuator_states['water_pump'] == False
    
    def test_turn_on_lights(self):
        """Test turning on grow lights"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        result = controller.turn_on_lights()
        
        assert result == True
        assert controller.actuator_states['grow_lights'] == True
    
    def test_turn_off_lights(self):
        """Test turning off grow lights"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # First turn on
        controller.turn_on_lights()
        assert controller.actuator_states['grow_lights'] == True
        
        # Then turn off
        result = controller.turn_off_lights()
        
        assert result == True
        assert controller.actuator_states['grow_lights'] == False
    
    def test_adjust_ph_up(self):
        """Test pH up adjustment"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        result = controller.adjust_ph_up(1.0)
        
        assert result == True
        
        # Check that usage was recorded
        assert 'ph_up_pump' in controller.pump_timers
        assert 'last_run' in controller.pump_timers['ph_up_pump']
        assert 'duration' in controller.pump_timers['ph_up_pump']
    
    def test_adjust_ph_down(self):
        """Test pH down adjustment"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        result = controller.adjust_ph_down(1.0)
        
        assert result == True
        
        # Check that usage was recorded
        assert 'ph_down_pump' in controller.pump_timers
        assert 'last_run' in controller.pump_timers['ph_down_pump']
        assert 'duration' in controller.pump_timers['ph_down_pump']
    
    def test_add_nutrients(self):
        """Test adding nutrients"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        result = controller.add_nutrients(2.0)
        
        assert result == True
        
        # Check that usage was recorded
        assert 'nutrient_pump' in controller.pump_timers
        assert 'last_run' in controller.pump_timers['nutrient_pump']
        assert 'duration' in controller.pump_timers['nutrient_pump']
    
    def test_pump_duration_limits(self):
        """Test pump duration safety limits"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # Test exceeding pH pump duration limit
        max_duration = controller.pump_limits['ph_up_pump']['max_duration']
        excessive_duration = max_duration + 1
        
        result = controller.adjust_ph_up(excessive_duration)
        
        # Should be rejected due to safety limit
        assert result == False
    
    def test_pump_cooldown_period(self):
        """Test pump cooldown period enforcement"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # First use within limits
        result1 = controller.adjust_ph_up(1.0)
        assert result1 == True
        
        # Immediate second use should be blocked by cooldown
        result2 = controller.adjust_ph_up(1.0)
        assert result2 == False
        
        # Simulate cooldown period passing
        cooldown_time = controller.pump_limits['ph_up_pump']['cooldown']
        past_time = datetime.now() - timedelta(seconds=cooldown_time + 1)
        controller.pump_timers['ph_up_pump']['last_run'] = past_time
        
        # Should now be allowed
        result3 = controller.adjust_ph_up(1.0)
        assert result3 == True
    
    def test_get_actuator_states(self):
        """Test getting actuator states"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # Turn on some actuators
        controller.turn_on_pump()
        controller.turn_on_lights()
        
        states = controller.get_actuator_states()
        
        assert isinstance(states, dict)
        assert states['water_pump'] == True
        assert states['grow_lights'] == True
        assert states['ph_up_pump'] == False  # Not turned on
    
    def test_get_pump_status(self):
        """Test getting detailed pump status"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # Use a pump
        controller.adjust_ph_up(1.0)
        
        status = controller.get_pump_status()
        
        assert isinstance(status, dict)
        assert 'ph_up_pump' in status
        
        pump_status = status['ph_up_pump']
        assert 'can_run' in pump_status
        assert 'daily_runtime' in pump_status
        assert 'last_run' in pump_status
        assert 'cooldown_remaining' in pump_status
        
        # Should not be able to run immediately due to cooldown
        assert pump_status['can_run'] == False
        assert pump_status['cooldown_remaining'] > 0
    
    def test_turn_off_all_actuators(self):
        """Test emergency shutdown of all actuators"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # Turn on some actuators
        controller.turn_on_pump()
        controller.turn_on_lights()
        
        # Emergency shutdown
        result = controller.turn_off_all()
        
        assert result == True
        assert controller.actuator_states['water_pump'] == False
        assert controller.actuator_states['grow_lights'] == False
    
    def test_light_schedule_setting(self):
        """Test setting light schedule"""
        sensor_reader = SimulatedSensorReader()
        controller = ActuatorController(sensor_reader)
        
        result = controller.set_light_schedule("06:00", "20:00")
        
        assert result == True
        assert controller.light_schedule['enabled'] == True
        assert controller.light_schedule['on_time'] == "06:00"
        assert controller.light_schedule['off_time'] == "20:00"
    
    def test_light_schedule_invalid_format(self):
        """Test setting light schedule with invalid time format"""
        sensor_reader = SimulatedSensorReader()
        controller = ActuatorController(sensor_reader)
        
        result = controller.set_light_schedule("6am", "8pm")
        
        assert result == False
        assert controller.light_schedule['enabled'] == False
    
    def test_auto_ph_control_low_ph(self):
        """Test automatic pH control when pH is too low"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # Test with low pH
        current_ph = 5.0
        target_range = (6.0, 7.0)
        
        result = controller.auto_control_ph(current_ph, target_range)
        
        assert result == True
        # Should have recorded pH up pump usage
        assert 'ph_up_pump' in controller.pump_timers
    
    def test_auto_ph_control_high_ph(self):
        """Test automatic pH control when pH is too high"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # Test with high pH
        current_ph = 8.0
        target_range = (6.0, 7.0)
        
        result = controller.auto_control_ph(current_ph, target_range)
        
        assert result == True
        # Should have recorded pH down pump usage
        assert 'ph_down_pump' in controller.pump_timers
    
    def test_auto_ph_control_optimal_ph(self):
        """Test automatic pH control when pH is optimal"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # Test with optimal pH
        current_ph = 6.5
        target_range = (6.0, 7.0)
        
        result = controller.auto_control_ph(current_ph, target_range)
        
        assert result == True
        # Should not have used any pumps
        assert 'ph_up_pump' not in controller.pump_timers
        assert 'ph_down_pump' not in controller.pump_timers
    
    def test_auto_nutrient_control_low_tds(self):
        """Test automatic nutrient control when TDS is too low"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # Test with low TDS
        current_tds = 400
        target_range = (800, 1200)
        
        result = controller.auto_control_nutrients(current_tds, target_range)
        
        assert result == True
        # Should have recorded nutrient pump usage
        assert 'nutrient_pump' in controller.pump_timers
    
    def test_auto_nutrient_control_optimal_tds(self):
        """Test automatic nutrient control when TDS is optimal"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # Test with optimal TDS
        current_tds = 1000
        target_range = (800, 1200)
        
        result = controller.auto_control_nutrients(current_tds, target_range)
        
        assert result == True
        # Should not have used nutrient pump
        assert 'nutrient_pump' not in controller.pump_timers
    
    def test_daily_runtime_tracking(self):
        """Test daily runtime tracking for pumps"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # Use pump multiple times
        controller.adjust_ph_up(1.0)
        
        # Wait for cooldown (simulate)
        cooldown_time = controller.pump_limits['ph_up_pump']['cooldown']
        past_time = datetime.now() - timedelta(seconds=cooldown_time + 1)
        controller.pump_timers['ph_up_pump']['last_run'] = past_time
        
        controller.adjust_ph_up(2.0)
        
        # Check daily runtime
        daily_runtime = controller._get_daily_runtime('ph_up_pump')
        assert daily_runtime >= 2.0  # Should be at least 3 seconds total
    
    def test_maintenance_cycle(self):
        """Test maintenance cycle execution"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # Mock time.sleep to speed up test
        with patch('time.sleep'):
            result = controller.perform_maintenance_cycle()
        
        assert result == True
        # Should have used pumps during maintenance
        assert 'nutrient_pump' in controller.pump_timers


class TestActuatorControllerSafety:
    """Test safety features of actuator controller"""
    
    def test_concurrent_pump_operations(self):
        """Test thread safety of pump operations"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        results = []
        errors = []
        
        def operate_pump():
            try:
                # Try to operate different pumps concurrently
                result = controller.adjust_ph_up(0.5)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=operate_pump)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should not have any errors
        assert len(errors) == 0
        assert len(results) == 5
        
        # Only one should succeed due to cooldown
        successful_operations = sum(results)
        assert successful_operations == 1
    
    def test_pump_safety_with_extreme_durations(self):
        """Test pump safety with extreme duration values"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # Test with negative duration
        result1 = controller.adjust_ph_up(-1.0)
        assert result1 == False
        
        # Test with zero duration
        result2 = controller.adjust_ph_up(0.0)
        assert result2 == True  # Zero duration should be allowed
        
        # Test with extremely large duration
        result3 = controller.adjust_ph_up(1000.0)
        assert result3 == False  # Should be blocked by safety limits
    
    def test_actuator_state_consistency(self):
        """Test actuator state consistency under rapid operations"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # Rapidly toggle actuators
        for i in range(10):
            if i % 2 == 0:
                controller.turn_on_pump()
                controller.turn_on_lights()
            else:
                controller.turn_off_pump()
                controller.turn_off_lights()
        
        # Final state should be consistent
        states = controller.get_actuator_states()
        assert states['water_pump'] == False  # Should end in OFF state
        assert states['grow_lights'] == False  # Should end in OFF state
    
    def test_pump_limits_modification_safety(self):
        """Test safety when pump limits are modified"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # Modify pump limits
        original_limit = controller.pump_limits['ph_up_pump']['max_duration']
        controller.pump_limits['ph_up_pump']['max_duration'] = 0.5
        
        # Test with new limit
        result1 = controller.adjust_ph_up(0.3)  # Within new limit
        assert result1 == True
        
        result2 = controller.adjust_ph_up(1.0)  # Exceeds new limit
        assert result2 == False
        
        # Restore original limit
        controller.pump_limits['ph_up_pump']['max_duration'] = original_limit
    
    def test_system_state_after_errors(self):
        """Test system state consistency after errors"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # Simulate some errors by trying invalid operations
        controller.adjust_ph_up(100.0)  # Should fail due to limits
        controller.adjust_ph_down(100.0)  # Should fail due to limits
        
        # System should still be in valid state
        states = controller.get_actuator_states()
        assert isinstance(states, dict)
        
        # Valid operations should still work
        result = controller.turn_on_pump()
        assert result == True


class TestActuatorControllerIntegration:
    """Integration tests for actuator controller"""
    
    def test_complete_growing_cycle_simulation(self):
        """Test complete growing cycle with automatic controls"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # Simulate daily growing cycle
        daily_schedule = [
            # Morning: Turn on lights, check pH
            {'time': '06:00', 'action': 'lights_on'},
            {'time': '06:30', 'action': 'check_ph', 'ph': 5.5},  # Low pH
            {'time': '12:00', 'action': 'check_nutrients', 'tds': 600},  # Low TDS
            {'time': '18:00', 'action': 'check_ph', 'ph': 7.5},  # High pH
            {'time': '22:00', 'action': 'lights_off'},
        ]
        
        for event in daily_schedule:
            if event['action'] == 'lights_on':
                result = controller.turn_on_lights()
                assert result == True
                
            elif event['action'] == 'lights_off':
                result = controller.turn_off_lights()
                assert result == True
                
            elif event['action'] == 'check_ph':
                # Allow enough time to pass for cooldown
                if 'ph_up_pump' in controller.pump_timers:
                    cooldown = controller.pump_limits['ph_up_pump']['cooldown']
                    past_time = datetime.now() - timedelta(seconds=cooldown + 1)
                    controller.pump_timers['ph_up_pump']['last_run'] = past_time
                
                if 'ph_down_pump' in controller.pump_timers:
                    cooldown = controller.pump_limits['ph_down_pump']['cooldown']
                    past_time = datetime.now() - timedelta(seconds=cooldown + 1)
                    controller.pump_timers['ph_down_pump']['last_run'] = past_time
                
                result = controller.auto_control_ph(event['ph'], (6.0, 7.0))
                assert result == True
                
            elif event['action'] == 'check_nutrients':
                # Allow enough time to pass for cooldown
                if 'nutrient_pump' in controller.pump_timers:
                    cooldown = controller.pump_limits['nutrient_pump']['cooldown']
                    past_time = datetime.now() - timedelta(seconds=cooldown + 1)
                    controller.pump_timers['nutrient_pump']['last_run'] = past_time
                
                result = controller.auto_control_nutrients(event['tds'], (800, 1200))
                assert result == True
        
        # Verify final system state
        states = controller.get_actuator_states()
        assert states['grow_lights'] == False  # Should be off after 22:00
        
        # Verify pump usage was recorded
        pump_status = controller.get_pump_status()
        assert len(pump_status) > 0
    
    def test_sensor_actuator_feedback_loop(self):
        """Test feedback loop between sensors and actuators"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # Simulate feedback loop
        for cycle in range(3):
            # Get sensor data
            sensor_data = sensor_reader.read_sensors()
            assert sensor_data is not None
            
            # Allow time for cooldown if needed
            for pump in ['ph_up_pump', 'ph_down_pump', 'nutrient_pump']:
                if pump in controller.pump_timers:
                    cooldown = controller.pump_limits[pump]['cooldown']
                    past_time = datetime.now() - timedelta(seconds=cooldown + 1)
                    controller.pump_timers[pump]['last_run'] = past_time
            
            # React to sensor data
            ph = sensor_data.get('ph', 7.0)
            tds = sensor_data.get('tds', 800)
            
            # Control pH
            ph_result = controller.auto_control_ph(ph, (6.0, 7.0))
            assert isinstance(ph_result, bool)
            
            # Control nutrients
            nutrient_result = controller.auto_control_nutrients(tds, (800, 1200))
            assert isinstance(nutrient_result, bool)
            
            # Small delay between cycles
            time.sleep(0.1)
    
    def test_emergency_scenarios(self):
        """Test emergency shutdown scenarios"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # Start some operations
        controller.turn_on_pump()
        controller.turn_on_lights()
        
        # Simulate emergency
        emergency_result = controller.turn_off_all()
        assert emergency_result == True
        
        # Verify all actuators are off
        states = controller.get_actuator_states()
        for actuator, state in states.items():
            assert state == False, f"Actuator {actuator} should be OFF after emergency shutdown"
    
    def test_long_term_operation_simulation(self):
        """Test long-term operation simulation"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # Simulate operations over time
        operation_count = 0
        error_count = 0
        
        for i in range(20):  # Simulate 20 operations
            try:
                # Alternate between different operations
                if i % 4 == 0:
                    result = controller.turn_on_pump()
                elif i % 4 == 1:
                    result = controller.turn_off_pump()
                elif i % 4 == 2:
                    result = controller.turn_on_lights()
                else:
                    result = controller.turn_off_lights()
                
                if result:
                    operation_count += 1
                    
            except Exception as e:
                error_count += 1
        
        # Most operations should succeed
        assert operation_count >= 10
        assert error_count == 0  # No exceptions should occur
        
        # System should be in valid state
        states = controller.get_actuator_states()
        assert isinstance(states, dict)


class TestSimulatedActuatorController:
    """Test cases specific to SimulatedActuatorController"""
    
    def test_simulated_controller_initialization(self):
        """Test simulated controller initialization"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        assert isinstance(controller, SimulatedActuatorController)
        assert controller.sensor_reader == sensor_reader
    
    def test_simulated_actuator_operations(self):
        """Test simulated actuator operations"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # All operations should succeed in simulation
        assert controller.turn_on_pump() == True
        assert controller.turn_off_pump() == True
        assert controller.turn_on_lights() == True
        assert controller.turn_off_lights() == True
        assert controller.adjust_ph_up(1.0) == True
        assert controller.adjust_ph_down(1.0) == True
        assert controller.add_nutrients(2.0) == True
    
    def test_simulated_vs_real_controller_interface(self):
        """Test that simulated controller has same interface as real controller"""
        sensor_reader = SimulatedSensorReader()
        
        real_controller = ActuatorController(sensor_reader)
        simulated_controller = SimulatedActuatorController(sensor_reader)
        
        # Both should have same methods
        real_methods = [method for method in dir(real_controller) if not method.startswith('_')]
        simulated_methods = [method for method in dir(simulated_controller) if not method.startswith('_')]
        
        # Simulated should have all methods of real controller
        for method in real_methods:
            assert hasattr(simulated_controller, method), f"Simulated controller missing method: {method}"


class TestActuatorControllerPerformance:
    """Performance tests for actuator controller"""
    
    def test_actuator_response_time(self):
        """Test actuator response time"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # Measure response time for actuator operations
        start_time = time.time()
        
        for _ in range(100):
            controller.turn_on_pump()
            controller.turn_off_pump()
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 200  # 200 operations total
        
        # Should be very fast for simulated operations
        assert avg_time < 0.001  # Less than 1ms per operation
    
    def test_state_update_performance(self):
        """Test performance of state updates"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        start_time = time.time()
        
        # Rapid state updates
        for i in range(1000):
            states = controller.get_actuator_states()
            assert isinstance(states, dict)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete quickly
        assert total_time < 1.0  # Less than 1 second for 1000 queries
    
    def test_concurrent_operations_performance(self):
        """Test performance under concurrent operations"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        results = []
        
        def perform_operations():
            local_results = []
            for _ in range(10):
                result = controller.turn_on_pump()
                local_results.append(result)
                result = controller.turn_off_pump()
                local_results.append(result)
            results.extend(local_results)
        
        # Create multiple threads
        threads = []
        start_time = time.time()
        
        for _ in range(5):
            thread = threading.Thread(target=perform_operations)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete reasonably quickly even with concurrency
        assert total_time < 5.0  # Less than 5 seconds
        assert len(results) == 100  # All operations should complete


# Test fixtures
@pytest.fixture
def sensor_reader():
    """Fixture providing a sensor reader"""
    return SimulatedSensorReader()

@pytest.fixture
def actuator_controller(sensor_reader):
    """Fixture providing an actuator controller"""
    return SimulatedActuatorController(sensor_reader)

@pytest.fixture
def real_actuator_controller(sensor_reader):
    """Fixture providing a real actuator controller (with mocked serial)"""
    return ActuatorController(sensor_reader)


class TestActuatorControllerEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_none_sensor_reader(self):
        """Test with None sensor reader"""
        with pytest.raises(Exception):
            ActuatorController(None)
    
    def test_invalid_duration_values(self):
        """Test with invalid duration values"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # Test with string duration
        with pytest.raises((TypeError, ValueError)):
            controller.adjust_ph_up("invalid")
        
        # Test with None duration
        with pytest.raises((TypeError, ValueError)):
            controller.adjust_ph_up(None)
    
    def test_actuator_state_corruption_recovery(self):
        """Test recovery from actuator state corruption"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # Corrupt actuator states
        controller.actuator_states = {}
        
        # Operations should handle gracefully or raise appropriate exceptions
        try:
            result = controller.turn_on_pump()
            # If it succeeds, state should be restored
            assert 'water_pump' in controller.actuator_states
        except KeyError:
            # Expected behavior for corrupted state
            pass
    
    def test_pump_timer_corruption_recovery(self):
        """Test recovery from pump timer corruption"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # Corrupt pump timers with invalid data
        controller.pump_timers['ph_up_pump'] = {'invalid': 'data'}
        
        # Should handle gracefully
        result = controller.adjust_ph_up(1.0)
        # May succeed or fail, but should not crash
        assert isinstance(result, bool)
    
    def test_extreme_time_scenarios(self):
        """Test with extreme time scenarios"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # Test with future timestamp
        future_time = datetime.now() + timedelta(days=1)
        controller.pump_timers['ph_up_pump'] = {
            'last_run': future_time,
            'duration': 1.0
        }
        
        # Should handle gracefully
        result = controller.adjust_ph_up(1.0)
        assert isinstance(result, bool)


class TestActuatorControllerMaintenance:
    """Test maintenance and monitoring features"""
    
    def test_pump_usage_statistics(self):
        """Test pump usage statistics collection"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # Use pumps multiple times
        pumps_to_test = ['ph_up_pump', 'ph_down_pump', 'nutrient_pump']
        
        for pump in pumps_to_test:
            # Use each pump (with cooldown management)
            if pump == 'ph_up_pump':
                controller.adjust_ph_up(1.0)
            elif pump == 'ph_down_pump':
                controller.adjust_ph_down(1.0)
            elif pump == 'nutrient_pump':
                controller.add_nutrients(2.0)
            
            # Simulate cooldown period
            if pump in controller.pump_timers:
                cooldown = controller.pump_limits[pump]['cooldown']
                past_time = datetime.now() - timedelta(seconds=cooldown + 1)
                controller.pump_timers[pump]['last_run'] = past_time
        
        # Check statistics
        status = controller.get_pump_status()
        
        for pump in pumps_to_test:
            assert pump in status
            pump_info = status[pump]
            
            assert 'daily_runtime' in pump_info
            assert pump_info['daily_runtime'] > 0  # Should have some runtime
            assert pump_info['last_run'] is not None
    
    def test_system_health_monitoring(self):
        """Test system health monitoring capabilities"""
        sensor_reader = SimulatedSensorReader()
        controller = SimulatedActuatorController(sensor_reader)
        
        # Perform various operations
        controller.turn_on_pump()
        controller.turn_on_lights()
        controller.adjust_ph_up(1.0)
        
        # Check system health indicators
        states = controller.get_actuator_states()
        pump_status = controller.get_pump_status()
        
        # Should provide comprehensive system information
        assert len(states) > 0
        assert len(pump_status) > 0
        
        # All data should be valid
        for actuator, state in states.items():
            assert isinstance(state, bool)
        
        for pump, info in pump_status.items():
            assert isinstance(info, dict)
            assert 'can_run' in info
            assert isinstance(info['can_run'], bool)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-s"])
