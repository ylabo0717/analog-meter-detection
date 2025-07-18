import cv2
import numpy as np
from pathlib import Path
from typing import Tuple
import math

from .detector import MeterReading, GaugeInfo


def save_debug_images(image_path: str, result: MeterReading, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        return

    base_name = Path(image_path).stem

    overlay_image = create_overlay_image(image, result)
    cv2.imwrite(str(output_path / f"{base_name}_overlay.jpg"), overlay_image)

    debug_image = create_debug_image(image, result)
    cv2.imwrite(str(output_path / f"{base_name}_debug.jpg"), debug_image)


def create_overlay_image(image: np.ndarray, result: MeterReading) -> np.ndarray:
    overlay = image.copy()

    if result.temp_gauge:
        _draw_gauge_overlay(
            overlay,
            result.temp_gauge,
            result.temp_angle,
            f"Temp: {result.temperature:.1f}°C",
            (0, 255, 0),
        )

    if result.humidity_gauge:
        _draw_gauge_overlay(
            overlay,
            result.humidity_gauge,
            result.humidity_angle,
            f"Humidity: {result.humidity:.0f}%",
            (255, 0, 0),
        )

    return overlay


def create_debug_image(image: np.ndarray, result: MeterReading) -> np.ndarray:
    debug = image.copy()

    if result.temp_gauge:
        _draw_gauge_debug(debug, result.temp_gauge, result.temp_angle, (0, 255, 0))

    if result.humidity_gauge:
        _draw_gauge_debug(
            debug, result.humidity_gauge, result.humidity_angle, (255, 0, 0)
        )

    return debug


def _draw_gauge_overlay(
    image: np.ndarray,
    gauge: GaugeInfo,
    angle: float,
    text: str,
    color: Tuple[int, int, int],
) -> None:
    center = gauge.center
    radius = gauge.radius

    cv2.circle(image, center, radius, color, 2)

    angle_rad = math.radians(angle)
    end_x = int(center[0] + radius * 0.8 * math.cos(angle_rad))
    end_y = int(center[1] + radius * 0.8 * math.sin(angle_rad))

    cv2.line(image, center, (end_x, end_y), color, 3)
    cv2.circle(image, center, 5, color, -1)

    text_pos = (center[0] - 50, center[1] - radius - 20)
    cv2.putText(image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def _draw_gauge_debug(
    image: np.ndarray, gauge: GaugeInfo, angle: float, color: Tuple[int, int, int]
) -> None:
    center = gauge.center
    radius = gauge.radius

    cv2.circle(image, center, radius, color, 1)
    cv2.circle(image, center, int(radius * 0.8), color, 1)
    cv2.circle(image, center, int(radius * 0.6), color, 1)

    for deg in range(-90, 91, 30):
        angle_rad = math.radians(deg)
        start_x = int(center[0] + radius * 0.9 * math.cos(angle_rad))
        start_y = int(center[1] + radius * 0.9 * math.sin(angle_rad))
        end_x = int(center[0] + radius * math.cos(angle_rad))
        end_y = int(center[1] + radius * math.sin(angle_rad))
        cv2.line(image, (start_x, start_y), (end_x, end_y), color, 1)

    angle_rad = math.radians(angle)
    end_x = int(center[0] + radius * 0.8 * math.cos(angle_rad))
    end_y = int(center[1] + radius * 0.8 * math.sin(angle_rad))
    cv2.line(image, center, (end_x, end_y), (0, 0, 255), 2)
    cv2.circle(image, center, 3, (0, 0, 255), -1)


def print_debug_info(result: MeterReading) -> None:
    print("=== Debug Information ===")
    print(
        f"Temperature: {result.temperature:.1f}°C (angle: {result.temp_angle:.1f}°, confidence: {result.temp_confidence:.2f})"
    )
    print(
        f"Humidity: {result.humidity:.0f}% (angle: {result.humidity_angle:.1f}°, confidence: {result.humidity_confidence:.2f})"
    )

    if result.temp_gauge:
        print(
            f"Temperature gauge center: {result.temp_gauge.center}, radius: {result.temp_gauge.radius}"
        )

    if result.humidity_gauge:
        print(
            f"Humidity gauge center: {result.humidity_gauge.center}, radius: {result.humidity_gauge.radius}"
        )

    print("========================")
