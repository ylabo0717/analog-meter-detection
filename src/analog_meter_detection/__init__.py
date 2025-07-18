import argparse
import sys
from pathlib import Path

from .detector import MeterDetector
from .image_utils import save_debug_images, print_debug_info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analog meter detection for temperature and humidity"
    )
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument(
        "--debug", action="store_true", help="Save debug images and print detailed info"
    )
    parser.add_argument(
        "--output-dir", default="output", help="Output directory for debug images"
    )

    args = parser.parse_args()

    if not Path(args.image_path).exists():
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)

    try:
        detector = MeterDetector()
        result = detector.detect(args.image_path)

        print(f"Temperature: {result.temperature:.1f}°C")
        print(f"Humidity: {result.humidity:.0f}%")

        if args.debug:
            print_debug_info(result)
            save_debug_images(args.image_path, result, args.output_dir)
            print(f"Debug images saved to: {args.output_dir}")

    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)
