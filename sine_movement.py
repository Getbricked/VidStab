#!/usr/bin/env python3
"""
Sine Wave Movement for RoboMaster EP
====================================

This script makes the RoboMaster EP robot move in a sine wave pattern.
The robot will move forward while oscillating left and right in a sinusoidal motion.

Usage:
    python sine_movement.py [--ip IP_ADDRESS] [--amplitude AMPLITUDE] [--frequency FREQUENCY] [--speed SPEED] [--duration DURATION]

Requirements:
    - RoboMaster EP robot
    - robomasterpy library
    - Robot should be in router mode or connected to the same network
"""

import time
import math
import argparse
import sys
from typing import Optional

# Import the robomasterpy library
import robomasterpy as rm


class SineMovementController:
    """Controller for making the robot move in a sine wave pattern."""

    def __init__(self, robot_ip: str = "", timeout: float = 30.0):
        """
        Initialize the sine movement controller.

        Args:
            robot_ip: IP address of the RoboMaster EP (empty string for auto-detection)
            timeout: Connection timeout in seconds
        """
        self.commander: Optional[rm.Commander] = None
        self.robot_ip = robot_ip
        self.timeout = timeout
        self.is_connected = False

    def connect(self) -> bool:
        """
        Connect to the RoboMaster EP robot.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            print(f"Connecting to RoboMaster EP...")
            if self.robot_ip:
                print(f"Using IP: {self.robot_ip}")
            else:
                print("Auto-detecting robot IP...")

            self.commander = rm.Commander(ip=self.robot_ip, timeout=self.timeout)
            actual_ip = self.commander.get_ip()
            print(f"Successfully connected to robot at {actual_ip}")

            # Set robot to free mode for movement
            self.commander.robot_mode(rm.MODE_FREE)
            print("Robot set to FREE mode")

            self.is_connected = True
            return True

        except Exception as e:
            print(f"Failed to connect to robot: {e}")
            return False

    def disconnect(self):
        """Disconnect from the robot and stop all movement."""
        if self.commander and self.is_connected:
            try:
                # Stop the robot before disconnecting
                self.commander.chassis_speed(0, 0, 0)
                print("Robot stopped")

                self.commander.close()
                print("Disconnected from robot")

            except Exception as e:
                print(f"Error during disconnect: {e}")
            finally:
                self.is_connected = False
                self.commander = None

    def move_sine_wave(
        self,
        amplitude: float = 0.5,
        frequency: float = 0.5,
        forward_speed: float = 0.3,
        duration: float = 30.0,
        update_rate: float = 20.0,
    ):
        """
        Make the robot move in a sine wave pattern.

        Args:
            amplitude: Maximum sideways displacement in meters (0.1 - 1.0)
            frequency: Frequency of the sine wave in Hz (0.1 - 2.0)
            forward_speed: Forward movement speed in m/s (0.1 - 1.0)
            duration: Total duration of movement in seconds
            update_rate: Control update rate in Hz (higher = smoother movement)
        """
        if not self.is_connected or not self.commander:
            print("Error: Robot not connected")
            return

        # Validate parameters
        amplitude = max(0.1, min(1.0, amplitude))
        frequency = max(0.1, min(2.0, frequency))
        forward_speed = max(0.1, min(1.0, forward_speed))
        update_rate = max(5.0, min(50.0, update_rate))

        print(f"Starting sine wave movement:")
        print(f"  Amplitude: {amplitude:.2f}m")
        print(f"  Frequency: {frequency:.2f}Hz")
        print(f"  Forward speed: {forward_speed:.2f}m/s")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Update rate: {update_rate:.1f}Hz")
        print("Press Ctrl+C to stop...")

        start_time = time.time()
        dt = 1.0 / update_rate

        try:
            while time.time() - start_time < duration:
                current_time = time.time() - start_time

                # Calculate sine wave position for y-axis (sideways movement)
                # y_speed is the derivative of sine position to get smooth velocity
                sine_angle = 2 * math.pi * frequency * current_time
                y_speed = amplitude * 2 * math.pi * frequency * math.cos(sine_angle)

                # Limit y_speed to robot's maximum capability (-3.5 to 3.5 m/s)
                y_speed = max(-3.0, min(3.0, y_speed))

                # Send movement command
                self.commander.chassis_speed(
                    x=forward_speed,  # Constant forward movement
                    y=y_speed,  # Sinusoidal sideways movement
                    z=0,  # No rotation
                )

                # Print status every second
                if int(current_time) != int(current_time - dt):
                    remaining = duration - current_time
                    print(
                        f"Time: {current_time:.1f}s, Y-speed: {y_speed:.2f}m/s, Remaining: {remaining:.1f}s"
                    )

                time.sleep(dt)

        except KeyboardInterrupt:
            print("\nMovement interrupted by user")
        except Exception as e:
            print(f"Error during movement: {e}")
        finally:
            # Stop the robot
            self.commander.chassis_speed(0, 0, 0)
            print("Movement completed, robot stopped")

    def move_sine_wave_position_based(
        self,
        amplitude: float = 1.0,
        wavelength: float = 2.0,
        num_waves: int = 3,
        speed: float = 0.5,
    ):
        """
        Alternative sine wave movement using position-based commands.
        This method moves the robot through discrete points along a sine wave.

        Args:
            amplitude: Maximum sideways displacement in meters
            wavelength: Length of one complete sine wave in meters
            num_waves: Number of complete waves to trace
            speed: Movement speed in m/s
        """
        if not self.is_connected or not self.commander:
            print("Error: Robot not connected")
            return

        print(f"Starting position-based sine wave movement:")
        print(f"  Amplitude: {amplitude:.2f}m")
        print(f"  Wavelength: {wavelength:.2f}m")
        print(f"  Number of waves: {num_waves}")
        print(f"  Speed: {speed:.2f}m/s")
        print("Press Ctrl+C to stop...")

        total_distance = wavelength * num_waves
        num_points = max(20, int(total_distance * 10))  # At least 10 points per meter

        try:
            for i in range(num_points + 1):
                progress = i / num_points
                x_distance = total_distance / num_points

                # Calculate y position based on sine wave
                sine_angle = 2 * math.pi * num_waves * progress
                y_distance = amplitude * math.sin(sine_angle)

                # Move to the next point
                self.commander.chassis_move(
                    x=x_distance,
                    y=y_distance
                    - (
                        amplitude
                        * math.sin(
                            2 * math.pi * num_waves * (progress - 1 / num_points)
                        )
                        if i > 0
                        else 0
                    ),
                    z=0,
                    speed_xy=speed,
                )

                print(
                    f"Point {i+1}/{num_points+1}: x={x_distance:.3f}m, y={y_distance:.3f}m"
                )

                # Wait for movement to complete (estimated time + buffer)
                move_time = x_distance / speed + 0.5
                time.sleep(move_time)

        except KeyboardInterrupt:
            print("\nMovement interrupted by user")
        except Exception as e:
            print(f"Error during movement: {e}")
        finally:
            print("Position-based sine wave movement completed")


def main():
    """Main function to run the sine wave movement."""
    parser = argparse.ArgumentParser(
        description="Make RoboMaster EP move in a sine wave pattern"
    )
    parser.add_argument(
        "--ip",
        type=str,
        default="",
        help="IP address of RoboMaster EP (auto-detect if not specified)",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.5,
        help="Sine wave amplitude in meters (default: 0.5)",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=0.5,
        help="Sine wave frequency in Hz (default: 0.5)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.3,
        help="Forward movement speed in m/s (default: 0.3)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Movement duration in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--method",
        choices=["speed", "position"],
        default="speed",
        help="Movement method: 'speed' (velocity-based) or 'position' (position-based)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Connection timeout in seconds (default: 30.0)",
    )

    args = parser.parse_args()

    # Create controller and connect
    controller = SineMovementController(robot_ip=args.ip, timeout=args.timeout)

    if not controller.connect():
        print("Failed to connect to robot. Exiting.")
        return

    try:
        if args.method == "speed":
            # Velocity-based sine wave movement
            controller.move_sine_wave(
                amplitude=args.amplitude,
                frequency=args.frequency,
                forward_speed=args.speed,
                duration=args.duration,
            )
        else:
            # Position-based sine wave movement
            controller.move_sine_wave_position_based(
                amplitude=args.amplitude,
                wavelength=2.0,  # Fixed wavelength for position-based method
                num_waves=3,  # Fixed number of waves
                speed=args.speed,
            )

    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()
