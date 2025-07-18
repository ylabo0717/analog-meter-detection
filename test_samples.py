#!/usr/bin/env python3

from pathlib import Path
from src.analog_meter_detection.detector import MeterDetector


def test_all_samples():
    expected_values = {
        "meter_001.jpg": (23.5, 58),
        "meter_002.jpg": (25.5, 75),
        "meter_003.jpg": (29.5, 67),
        "meter_004.jpg": (19.0, 49),
        "meter_005.jpg": (24.0, 48),
    }

    detector = MeterDetector()
    all_passed = True

    print("Testing analog meter detection on sample images...")
    print("=" * 60)

    for filename, (exp_temp, exp_humidity) in expected_values.items():
        image_path = f"data/{filename}"

        if not Path(image_path).exists():
            print(f"❌ {filename}: Image file not found")
            all_passed = False
            continue

        try:
            result = detector.detect(image_path)
            temp_error = abs(result.temperature - exp_temp)
            humidity_error = abs(result.humidity - exp_humidity)

            temp_status = "✅" if temp_error < 1.0 else "❌"
            humidity_status = "✅" if humidity_error < 2.0 else "❌"

            print(f"{filename}:")
            print(
                f"  {temp_status} Temperature: {result.temperature:.1f}°C (expected: {exp_temp}°C, error: ±{temp_error:.1f}°C)"
            )
            print(
                f"  {humidity_status} Humidity: {result.humidity:.0f}% (expected: {exp_humidity}%, error: ±{humidity_error:.0f}%)"
            )
            print(
                f"  Confidence: T={result.temp_confidence:.2f}, H={result.humidity_confidence:.2f}"
            )

            if temp_error >= 1.0:
                print(
                    f"  ⚠️  Temperature error too high: {temp_error:.1f}°C (limit: 1.0°C)"
                )
                all_passed = False

            if humidity_error >= 2.0:
                print(
                    f"  ⚠️  Humidity error too high: {humidity_error:.0f}% (limit: 2.0%)"
                )
                all_passed = False

            print()

        except Exception as e:
            print(f"❌ {filename}: Error - {e}")
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("🎉 All tests passed! Accuracy requirements met.")
    else:
        print("❌ Some tests failed. Check the errors above.")

    return all_passed


if __name__ == "__main__":
    test_all_samples()
