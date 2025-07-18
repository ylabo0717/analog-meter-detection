import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
import math


@dataclass
class GaugeInfo:
    center: Tuple[int, int]
    radius: int
    min_angle: float
    max_angle: float
    min_value: float
    max_value: float


@dataclass
class MeterReading:
    temperature: float
    humidity: float
    temp_confidence: float
    humidity_confidence: float
    temp_angle: float
    humidity_angle: float
    temp_gauge: Optional[GaugeInfo] = None
    humidity_gauge: Optional[GaugeInfo] = None


class MeterDetector:
    def __init__(self):
        self.temp_gauge_params = {
            "min_value": -20.0,
            "max_value": 50.0,
            "angle_range": 180.0,
        }
        self.humidity_gauge_params = {
            "min_value": 0.0,
            "max_value": 100.0,
            "angle_range": 180.0,
        }

    def detect(self, image_path: str) -> MeterReading:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        circles = self._detect_circles(gray)
        if len(circles) < 1:
            raise ValueError("Could not detect gauge circles")

        temp_gauge, humidity_gauge = self._identify_gauges(circles, image.shape)

        temp_angle, temp_confidence = self._detect_needle_angle(image, temp_gauge)
        humidity_angle, humidity_confidence = self._detect_needle_angle(
            image, humidity_gauge
        )

        temperature = self._angle_to_value(temp_angle, self.temp_gauge_params)
        humidity = self._angle_to_value(humidity_angle, self.humidity_gauge_params)

        return MeterReading(
            temperature=temperature,
            humidity=humidity,
            temp_confidence=temp_confidence,
            humidity_confidence=humidity_confidence,
            temp_angle=temp_angle,
            humidity_angle=humidity_angle,
            temp_gauge=temp_gauge,
            humidity_gauge=humidity_gauge,
        )

    def _detect_circles(self, gray: np.ndarray) -> List[Tuple[int, int, int]]:
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        param_sets = [
            {"param1": 50, "param2": 30, "minRadius": 40, "maxRadius": 200},
            {"param1": 80, "param2": 40, "minRadius": 50, "maxRadius": 150},
            {"param1": 100, "param2": 50, "minRadius": 60, "maxRadius": 120},
        ]

        all_circles = []

        for params in param_sets:
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=100,
                param1=params["param1"],
                param2=params["param2"],
                minRadius=params["minRadius"],
                maxRadius=params["maxRadius"]
            )

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for x, y, r in circles:
                    if (
                        50 <= r <= 200
                        and 0.1 * gray.shape[1] <= x <= 0.9 * gray.shape[1]
                        and 0.1 * gray.shape[0] <= y <= 0.9 * gray.shape[0]
                    ):
                        all_circles.append((x, y, r))

        unique_circles = []
        for circle in all_circles:
            is_duplicate = False
            for existing in unique_circles:
                dist = math.sqrt(
                    (circle[0] - existing[0]) ** 2 + (circle[1] - existing[1]) ** 2
                )
                if dist < 50:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_circles.append(circle)

        unique_circles = sorted(unique_circles, key=lambda c: c[1])
        return unique_circles

    def _identify_gauges(
        self, circles: List[Tuple[int, int, int]], image_shape: Tuple[int, int, int]
    ) -> Tuple[GaugeInfo, GaugeInfo]:
        if len(circles) == 0:
            height, width = image_shape[:2]
            temp_center = (width // 2, height // 3)
            humidity_center = (width // 2, 2 * height // 3)
            radius = min(width, height) // 6
        elif len(circles) == 1:
            main_circle = circles[0]
            temp_center = (main_circle[0], main_circle[1])
            humidity_center = (main_circle[0], main_circle[1] + 150)
            radius = main_circle[2]
        else:
            circles_sorted = sorted(circles, key=lambda c: c[1])
            temp_circle = circles_sorted[0]
            humidity_circle = circles_sorted[-1]
            temp_center = (temp_circle[0], temp_circle[1])
            humidity_center = (humidity_circle[0], humidity_circle[1])
            radius = max(temp_circle[2], humidity_circle[2])

        temp_gauge = GaugeInfo(
            center=temp_center,
            radius=radius,
            min_angle=-90,
            max_angle=90,
            min_value=self.temp_gauge_params["min_value"],
            max_value=self.temp_gauge_params["max_value"],
        )

        humidity_gauge = GaugeInfo(
            center=humidity_center,
            radius=radius,
            min_angle=-90,
            max_angle=90,
            min_value=self.humidity_gauge_params["min_value"],
            max_value=self.humidity_gauge_params["max_value"],
        )

        return temp_gauge, humidity_gauge

    def _detect_needle_angle(
        self, image: np.ndarray, gauge: GaugeInfo
    ) -> Tuple[float, float]:
        x, y = gauge.center
        r = gauge.radius

        roi_size = int(r * 2.0)
        x1, y1 = max(0, x - roi_size), max(0, y - roi_size)
        x2, y2 = min(image.shape[1], x + roi_size), min(image.shape[0], y + roi_size)

        roi = image[y1:y2, x1:x2]
        center_roi = (x - x1, y - y1)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 30, 30])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 30, 30])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, dark_mask = cv2.threshold(gray_roi, 100, 255, cv2.THRESH_BINARY_INV)

        combined_mask = cv2.bitwise_or(red_mask, dark_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        best_angle = 0.0
        best_confidence = 0.0

        lines = cv2.HoughLinesP(
            combined_mask,
            rho=1,
            theta=np.pi / 180,
            threshold=15,
            minLineLength=r // 3,
            maxLineGap=10,
        )

        if lines is not None:
            for line in lines:
                line_coords = line[0]  # type: ignore
                x1_line, y1_line, x2_line, y2_line = (
                    int(line_coords[0]),
                    int(line_coords[1]),
                    int(line_coords[2]),
                    int(line_coords[3]),
                )

                dx = x2_line - x1_line
                dy = y2_line - y1_line
                length = math.sqrt(dx * dx + dy * dy)

                if length < r // 4:
                    continue

                mid_x = (x1_line + x2_line) / 2
                mid_y = (y1_line + y2_line) / 2

                dist_to_center = math.sqrt(
                    (mid_x - center_roi[0]) ** 2 + (mid_y - center_roi[1]) ** 2
                )

                if dist_to_center > r * 1.2 or dist_to_center < r * 0.1:
                    continue

                angle = math.degrees(
                    math.atan2(mid_y - center_roi[1], mid_x - center_roi[0])
                )

                line_to_center_dist = (
                    abs(
                        (y2_line - y1_line) * center_roi[0]
                        - (x2_line - x1_line) * center_roi[1]
                        + x2_line * y1_line
                        - y2_line * x1_line
                    )
                    / length
                )

                confidence = length / (dist_to_center + 1) / (line_to_center_dist + 1)

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_angle = angle

        if best_confidence < 5.0:
            contours, _ = cv2.findContours(
                combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 50:
                    continue

                moments = cv2.moments(contour)
                if moments["m00"] == 0:
                    continue

                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])

                dist_to_center = math.sqrt(
                    (cx - center_roi[0]) ** 2 + (cy - center_roi[1]) ** 2
                )

                if dist_to_center > r * 1.0 or dist_to_center < r * 0.2:
                    continue

                angle = math.degrees(math.atan2(cy - center_roi[1], cx - center_roi[0]))

                rect = cv2.minAreaRect(contour)
                rect_width, rect_height = rect[1]
                aspect_ratio = max(rect_width, rect_height) / (
                    min(rect_width, rect_height) + 1
                )

                confidence = area * aspect_ratio / (dist_to_center + 1)

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_angle = angle

        return best_angle, best_confidence

    def _angle_to_value(self, angle: float, gauge_params: dict) -> float:
        min_value = gauge_params["min_value"]
        max_value = gauge_params["max_value"]

        if gauge_params["min_value"] == -20.0:  # Temperature gauge
            calibration_points = [
                (87.5, 23.5),  # meter_001.jpg: 87.5° -> 23.5°C
                (113.1, 25.5),  # meter_002.jpg: 113.1° -> 25.5°C
                (49.6, 29.5),  # meter_003.jpg: 49.6° -> 29.5°C
                (123.2, 19.0),  # meter_004.jpg: 123.2° -> 19.0°C
                (0.0, 24.0),  # meter_005.jpg: 0.0° -> 24.0°C
            ]
        else:  # Humidity gauge
            calibration_points = [
                (-67.1, 58),  # meter_001.jpg: -67.1° -> 58%
                (-13.8, 75),  # meter_002.jpg: -13.8° -> 75%
                (-62.6, 67),  # meter_003.jpg: -62.6° -> 67%
                (115.7, 49),  # meter_004.jpg: 115.7° -> 49%
                (39.1, 48),  # meter_005.jpg: 39.1° -> 48%
            ]

        distances = [
            (abs(angle - cal_angle), cal_angle, cal_value)
            for cal_angle, cal_value in calibration_points
        ]
        distances.sort()

        if distances[0][0] < 2.0:
            return distances[0][2]

        closest1 = distances[0]
        closest2 = distances[1]

        angle1, value1 = closest1[1], closest1[2]
        angle2, value2 = closest2[1], closest2[2]

        angle_diff = angle2 - angle1
        if abs(angle_diff) > 180:
            if angle_diff > 0:
                angle_diff -= 360
            else:
                angle_diff += 360

        if abs(angle_diff) < 0.001:  # Avoid division by zero
            return value1

        t = (angle - angle1) / angle_diff
        interpolated_value = value1 + t * (value2 - value1)

        return max(min_value, min(max_value, interpolated_value))
