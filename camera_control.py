####################################################################
# This code is not originally mine I took it from: https://github.com/nanmu42/robo-playground - drive.py

import logging
import multiprocessing as mp
import pickle
import queue
import threading
from typing import Tuple, List
import numpy as np
import click
import cv2 as cv
import robomasterpy as rm
from pynput import keyboard
from pynput.keyboard import Key, KeyCode
from robomasterpy import CTX
from robomasterpy import framework as rmf
from pynput.keyboard import Controller
import time

rm.LOG_LEVEL = logging.INFO
pickle.DEFAULT_PROTOCOL = pickle.HIGHEST_PROTOCOL

QUEUE_SIZE: int = 10
PUSH_FREQUENCY: int = 1
TIMEOUT_UNIT: float = 0.1
QUEUE_TIMEOUT: float = TIMEOUT_UNIT / PUSH_FREQUENCY

global delta_x, delta_y
delta_x = 0
delta_y = 0


def follow_target(delta_x, delta_y):
    if abs(delta_x) <= 30 and abs(delta_y) <= 30:
        return

    cmd = rm.Commander()

    # time.sleep(2)
    if delta_x > 0:
        cmd.gimbal_move(0, -15)

    if delta_x < 0:
        cmd.gimbal_move(0, 15)

    # time.sleep(2)
    if delta_y > 0:
        cmd.gimbal_move(-15, 0)

    if delta_y < 0:
        cmd.gimbal_move(15, 0)


# just display the streaming video
def display(frame, **kwargs):
    # Adjust contrast
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 0  # Brightness control (0-100)
    adjusted_frame = cv.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # Convert the frame to HSV color space
    hsv_frame = cv.cvtColor(adjusted_frame, cv.COLOR_BGR2HSV)

    # Define the lower and upper bounds for red color in HSV
    lower_red1 = np.array([0, 120, 70])  # Lower range of red
    upper_red1 = np.array([10, 255, 255])  # Upper range of red
    lower_red2 = np.array([170, 120, 70])  # Lower range for another red hue
    upper_red2 = np.array([180, 255, 255])  # Upper range for another red hue

    # Create masks for red color (both ranges)
    mask1 = cv.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv_frame, lower_red2, upper_red2)
    red_mask = cv.bitwise_or(mask1, mask2)

    # Apply the mask to the original frame
    red_only = cv.bitwise_and(frame, frame, mask=red_mask)

    # Convert the masked frame to grayscale
    gray_frame = cv.cvtColor(red_only, cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_frame = cv.GaussianBlur(gray_frame, (9, 9), 2)

    # Detect circles using Hough Circle Transform
    circles = cv.HoughCircles(
        blurred_frame,
        cv.HOUGH_GRADIENT,
        dp=1.2,  # Inverse ratio of the accumulator resolution to the image resolution
        minDist=50,  # Minimum distance between the centers of detected circles
        param1=50,  # Upper threshold for Canny edge detector
        param2=30,  # Threshold for center detection
        minRadius=10,  # Minimum circle radius
        maxRadius=100,  # Maximum circle radius
    )

    # If circles are detected
    if circles is not None:
        circles = np.uint16(np.around(circles))  # Round circle parameters to integers

        # Frame center for calculating deltas
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2

        for circle in circles[0, :]:

            x, y, radius = circle

            # Calculate delta_x and delta_y
            global delta_x, delta_y
            delta_x = int(x) - int(frame_center_x)
            delta_y = int(y) - int(frame_center_y)

            # Expand the circular region slightly for better contrast calculation
            expanded_radius = int(radius * 1.2)  # Expand by 20%
            mask = np.zeros_like(gray_frame)  # Create a black mask
            cv.circle(mask, (x, y), expanded_radius, 255, -1)  # Expanded white circle
            target_region = cv.bitwise_and(
                gray_frame, gray_frame, mask=mask
            )  # Apply mask

            # Apply Gaussian blur to the target region for stability
            blurred_target_region = cv.GaussianBlur(target_region, (5, 5), 0)

            # Calculate the contrast in the target region
            target_pixels = blurred_target_region[
                mask == 255
            ]  # Extract non-zero pixels
            if len(target_pixels) > 0:
                mean, stddev = cv.meanStdDev(target_pixels)
                contrast = stddev[0][0]
            else:
                contrast = 0

            # Draw the detected circle
            cv.circle(frame, (x, y), radius, (0, 255, 0), 2)  # Green circle outline
            # Draw the circle center
            cv.circle(frame, (x, y), 2, (255, 0, 0), 3)  # Blue center dot

            # Display contrast near the circle
            cv.putText(
                frame,
                f"Contrast: {contrast:.2f}",
                (x - radius, y - radius - 20),  # Position text above the circle
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),  # Yellow text
                1,
            )

            # Display delta_x and delta_y near the circle
            cv.putText(
                frame,
                f"Delta: ({delta_x:.0f}, {delta_y:.0f})",
                (x - radius, y - radius - 40),  # Position text above contrast
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text
                1,
            )
            # time.sleep(2)
            follow_target(delta_x, delta_y)

    # Show the frame with detected circles, contrast, and deltas
    cv.imshow("frame", frame)
    cv.waitKey(1)


def handle_event(
    cmd: rm.Commander, queues: Tuple[mp.Queue, ...], logger: logging.Logger
) -> None:
    push_queue, event_queue = queues
    try:
        push = push_queue.get(timeout=QUEUE_TIMEOUT)
        logger.info("push: %s", push)
    except queue.Empty:
        pass

    try:
        event = event_queue.get(timeout=QUEUE_TIMEOUT)
        # safety first
        if type(event) == rm.ArmorHitEvent:
            cmd.chassis_speed(0, 0, 0)
        logger.info("event: %s", event)
    except queue.Empty:
        pass


class Controller:
    UNIT_DELTA_SPEED: float = 0.2
    UNIT_DELTA_DEGREE: float = 20

    def __init__(self, cmd: rm.Commander, logger: logging.Logger):
        self._mu = threading.Lock()
        with self._mu:
            self.gear: int = 1
            self.delta_v: float = self.UNIT_DELTA_SPEED
            self.delta_d: float = self.UNIT_DELTA_DEGREE
            self.cmd = cmd
            self.logger = logger
            self.v: List[float, float] = [0, 0]
            self.previous_v: List[float, float] = [0, 0]
            self.v_gimbal: List[float, float] = [0, 0]
            self.previous_v_gimbal: List[float, float] = [0, 0]
            self.ctrl_pressed: bool = False

    def on_press(self, key):
        with self._mu:
            self.logger.debug("pressed: %s", key)

            if key == Key.ctrl:
                self.ctrl_pressed = True
                return
            if self.ctrl_pressed and key == KeyCode(char="c"):
                # stop listener
                self.v = [0, 0]
                self.v_gimbal = [0, 0]
                self.send_command()
                return False
            if key == Key.space:
                self.cmd.blaster_fire()
                return

            if key == KeyCode(char="w"):
                self.v[0] = self.delta_v
            elif key == KeyCode(char="s"):
                self.v[0] = -self.delta_v
            elif key == KeyCode(char="a"):
                self.v[1] = -self.delta_v
            elif key == KeyCode(char="d"):
                self.v[1] = self.delta_v
            elif key == Key.up:
                self.v_gimbal[0] = self.delta_d
            elif key == Key.down:
                self.v_gimbal[0] = -self.delta_d
            elif key == Key.left:
                self.v_gimbal[1] = -self.delta_d
            elif key == Key.right:
                self.v_gimbal[1] = self.delta_d

            self.send_command()

    def _update_gear(self, gear: int):
        self.gear = gear
        self.delta_v = self.gear * self.UNIT_DELTA_SPEED
        self.delta_d = self.gear * self.UNIT_DELTA_DEGREE

    def on_release(self, key):
        with self._mu:
            self.logger.debug("released: %s", key)

            if key == Key.ctrl:
                self.ctrl_pressed = False
                return

            # gears
            if key in (
                KeyCode(char="1"),
                KeyCode(char="2"),
                KeyCode(char="3"),
                KeyCode(char="4"),
                KeyCode(char="5"),
            ):
                self._update_gear(int(key.char))
                return

            if key in (KeyCode(char="w"), KeyCode(char="s")):
                self.v[0] = 0
            elif key in (KeyCode(char="a"), KeyCode(char="d")):
                self.v[1] = 0
            elif key in (Key.up, Key.down):
                self.v_gimbal[0] = 0
            elif key in (Key.left, Key.right):
                self.v_gimbal[1] = 0

            self.send_command()

    def send_command(self):
        if self.v != self.previous_v:
            self.previous_v = [*self.v]
            self.logger.debug("chassis speed: x: %s, y: %s", self.v[0], self.v[1])
            self.cmd.chassis_speed(self.v[0], self.v[1], 0)
        if self.v_gimbal != self.previous_v_gimbal:
            self.logger.debug(
                "gimbal speed: pitch: %s, yaw: %s", self.v_gimbal[0], self.v_gimbal[1]
            )
            self.previous_v_gimbal = [*self.v_gimbal]
            self.cmd.gimbal_speed(self.v_gimbal[0], self.v_gimbal[1])


def control(cmd: rm.Commander, logger: logging.Logger, **kwargs) -> None:
    controller = Controller(cmd, logger)
    with keyboard.Listener(
        on_press=controller.on_press, on_release=controller.on_release
    ) as listener:
        listener.join()


@click.command()
@click.option("--ip", default="", type=str, help="(Optional) IP of Robomaster EP")
@click.option(
    "--timeout", default=10.0, type=float, help="(Optional) Timeout for commands"
)
def cli(ip: str, timeout: float):
    # manager is in charge of communicating among processes
    manager: mp.managers.SyncManager = CTX.Manager()
    global delta_x, delta_y
    with manager:
        # hub is the place to register your logic
        hub = rmf.Hub()
        cmd = rm.Commander(ip=ip, timeout=timeout)
        ip = cmd.get_ip()

        # initialize your Robomaster
        cmd.robot_mode(rm.MODE_FREE)
        # cmd.gimbal_recenter()

        # enable video streaming
        cmd.stream(True)
        # rm.Vision is a handler for video streaming
        # display is the callback function defined above
        hub.worker(rmf.Vision, "vision", (None, ip, display))

        # enable push and event
        cmd.chassis_push_on(PUSH_FREQUENCY, PUSH_FREQUENCY, PUSH_FREQUENCY)
        cmd.gimbal_push_on(PUSH_FREQUENCY)
        cmd.armor_sensitivity(10)
        cmd.armor_event(rm.ARMOR_HIT, True)
        cmd.sound_event(rm.SOUND_APPLAUSE, True)

        # the queues are where data flows
        push_queue = manager.Queue(QUEUE_SIZE)
        event_queue = manager.Queue(QUEUE_SIZE)

        # PushListener and EventListener handles push and event,
        # put parsed, well-defined data into queues.
        # hub.worker(rmf.Mind, "controller", ((), ip, control), {"loop": False})
        hub.worker(rmf.PushListener, "push", (push_queue,))
        hub.worker(rmf.EventListener, "event", (event_queue, ip))

        # Mind is the handler to let you bring your own controlling logic.
        # It can consume data from specified queues.
        # hub.worker(
        #     rmf.Mind, "event-handler", ((push_queue, event_queue), ip, handle_event)
        # )

        # a hub can have multiple Mind
        # follow_target(delta_x, delta_y)

        # Let's do this!
        hub.run()


if __name__ == "__main__":
    cli()

    def stop_program():
        cv.destroyAllWindows()
        exit(0)

    # class Controller:
    #     # existing code...

    #     def on_press(self, key):
    #         with self._mu:
    #             self.logger.debug("pressed: %s", key)

    #             if key == Key.esc:
    #                 stop_program()
    #                 return
    #             # existing code...

    #     # existing code...
