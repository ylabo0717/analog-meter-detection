import argparse
import math
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def detect_meter_center(image: np.ndarray) -> Tuple[int, int]:
    """メーターの中心を検出する"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 円検出のためのパラメータ調整
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=80,
        maxRadius=200,
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # 最も大きな円を選択
        largest_circle = max(circles, key=lambda c: c[2])
        return int(largest_circle[0]), int(largest_circle[1])

    # 円が検出されない場合は画像中心を返す
    h, w = image.shape[:2]
    return w // 2, h // 2


def detect_temperature_needle(image: np.ndarray, cx: int, cy: int) -> float:
    """温度計の針を検出"""
    # 温度計の中心を調整
    temp_cy = cy - 80

    # 複数の手法で針を検出
    # 1. 赤色フィルタリング + 輪郭検出
    angle = detect_needle_by_contour(image, cx, temp_cy, is_temperature=True)
    if angle != 0.0:
        return angle

    # 2. 線検出アルゴリズム
    angle = detect_needle_by_lines(image, cx, temp_cy, is_temperature=True)
    if angle != 0.0:
        return angle

    # 3. 放射線探索アルゴリズム
    angle = detect_needle_by_radial_search(image, cx, temp_cy, is_temperature=True)
    return angle


def detect_humidity_needle(image: np.ndarray, cx: int, cy: int) -> float:
    """湿度計の針を検出"""
    # 湿度計の中心を調整
    hum_cy = cy + 80

    # 複数の手法で針を検出
    # 1. 赤色フィルタリング + 輪郭検出
    angle = detect_needle_by_contour(image, cx, hum_cy, is_temperature=False)
    if angle != 0.0:
        return angle

    # 2. 線検出アルゴリズム
    angle = detect_needle_by_lines(image, cx, hum_cy, is_temperature=False)
    if angle != 0.0:
        return angle

    # 3. 放射線探索アルゴリズム
    angle = detect_needle_by_radial_search(image, cx, hum_cy, is_temperature=False)
    return angle


def detect_needle_by_contour(
    image: np.ndarray, cx: int, cy: int, is_temperature: bool
) -> float:
    """輪郭検出による針検出"""
    # 赤色を検出
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # より広い赤色範囲
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    # 検出範囲を設定
    if is_temperature:
        y_min = max(0, cy - 100)
        y_max = min(image.shape[0], cy + 30)
        x_min = max(0, cx - 120)
        x_max = min(image.shape[1], cx + 120)
    else:
        y_min = max(0, cy - 30)
        y_max = min(image.shape[0], cy + 100)
        x_min = max(0, cx - 100)
        x_max = min(image.shape[1], cx + 100)

    # 輪郭検出で針を見つける
    roi = red_mask[y_min:y_max, x_min:x_max]
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 最も大きな輪郭を選択
        largest_contour = max(contours, key=cv2.contourArea)

        # 輪郭の面積をチェック
        if cv2.contourArea(largest_contour) > 10:
            # 輪郭の中心を計算
            if len(largest_contour) >= 5:
                # 楕円フィッティング
                try:
                    ellipse = cv2.fitEllipse(largest_contour)
                    angle = ellipse[2] - 90  # 12時方向を0度とする

                    # 角度を-180度から180度に正規化
                    angle = angle % 360
                    if angle > 180:
                        angle -= 360

                    return angle
                except cv2.error:
                    pass

    return 0.0


def detect_needle_by_lines(
    image: np.ndarray, cx: int, cy: int, is_temperature: bool
) -> float:
    """線検出による針検出"""
    # 赤色を検出
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 赤色範囲
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    # 検出範囲を設定
    if is_temperature:
        y_min = max(0, cy - 100)
        y_max = min(image.shape[0], cy + 30)
        x_min = max(0, cx - 120)
        x_max = min(image.shape[1], cx + 120)
    else:
        y_min = max(0, cy - 30)
        y_max = min(image.shape[0], cy + 100)
        x_min = max(0, cx - 100)
        x_max = min(image.shape[1], cx + 100)

    # ROIを作成
    roi = red_mask[y_min:y_max, x_min:x_max]

    # エッジ検出
    edges = cv2.Canny(roi, 50, 150)

    # 線検出
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=15, minLineLength=20, maxLineGap=10
    )

    if lines is not None:
        # 最も長い線を選択
        longest_line = max(
            lines,
            key=lambda line: math.sqrt(
                (line[0][2] - line[0][0]) ** 2 + (line[0][3] - line[0][1]) ** 2
            ),
        )

        x1, y1, x2, y2 = longest_line[0]

        # 針の角度を計算（12時を0度とする）
        angle = math.atan2(x2 - x1, y1 - y2) * 180 / math.pi
        return angle

    return 0.0


def detect_needle_by_radial_search(
    image: np.ndarray, cx: int, cy: int, is_temperature: bool
) -> float:
    """放射線探索による針検出"""
    # 赤色を検出
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 赤色範囲
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    # 検出パラメータ
    if is_temperature:
        radius = 80
    else:
        radius = 60

    # 針の角度を探す
    best_angle = 0.0
    max_red_pixels = 0

    # 角度を5度刻みで探索
    for angle_deg in range(-140, 141, 5):
        angle_rad = math.radians(angle_deg)
        red_count = 0

        # 針の方向に沿って赤い画素を数える
        for r in range(15, radius):
            x = int(cx + r * math.sin(angle_rad))
            y = int(cy - r * math.cos(angle_rad))

            # 画像範囲内かチェック
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                if red_mask[y, x] > 0:
                    red_count += 1

        # 最も赤い画素が多い角度を選択
        if red_count > max_red_pixels:
            max_red_pixels = red_count
            best_angle = angle_deg

    return best_angle if max_red_pixels > 3 else 0.0


def detect_needle_angle(
    image: np.ndarray, center: Tuple[int, int], is_temperature: bool = True
) -> float:
    """針の角度を検出する（度単位）"""
    cx, cy = center

    # 実際の画像を分析して最適な針検出を行う
    if is_temperature:
        # 温度計の針検出（画像の上半分）
        return detect_temperature_needle(image, cx, cy)
    else:
        # 湿度計の針検出（画像の下半分）
        return detect_humidity_needle(image, cx, cy)


def angle_to_temperature(angle: float) -> float:
    """角度から温度を算出（-20℃～50℃）"""
    # メーターの針の角度範囲を実際の画像に合わせて調整
    # 左端（約-120度）が-20℃、右端（約120度）が50℃
    if angle < -120:
        angle = -120
    elif angle > 120:
        angle = 120

    # 角度を0～1の範囲に正規化
    normalized_angle = (angle + 120) / 240

    # 温度範囲（-20℃～50℃）にマッピング
    temperature = -20 + normalized_angle * 70

    return temperature


def angle_to_humidity(angle: float) -> float:
    """角度から湿度を算出（0%～100%）"""
    # メーターの針の角度範囲を実際の画像に合わせて調整
    # 左端（約-120度）が0%、右端（約120度）が100%
    if angle < -120:
        angle = -120
    elif angle > 120:
        angle = 120

    # 角度を0～1の範囲に正規化
    normalized_angle = (angle + 120) / 240

    # 湿度範囲（0%～100%）にマッピング
    humidity = normalized_angle * 100

    return humidity


def create_debug_image(
    image: np.ndarray,
    center: Tuple[int, int],
    temp_angle: float,
    humidity_angle: float,
    temperature: float,
    humidity: float,
) -> np.ndarray:
    """デバッグ用の画像を作成"""
    debug_img = image.copy()
    cx, cy = center

    # 中心点を描画
    cv2.circle(debug_img, (cx, cy), 5, (0, 255, 0), -1)

    # 温度針の方向を描画
    temp_x = cx + int(80 * math.sin(math.radians(temp_angle)))
    temp_y = cy - int(80 * math.cos(math.radians(temp_angle)))
    cv2.line(debug_img, (cx, cy), (temp_x, temp_y), (255, 0, 0), 3)

    # 湿度針の方向を描画
    humidity_x = cx + int(60 * math.sin(math.radians(humidity_angle)))
    humidity_y = cy - int(60 * math.cos(math.radians(humidity_angle)))
    cv2.line(debug_img, (cx, cy), (humidity_x, humidity_y), (0, 0, 255), 3)

    # 結果をテキストで表示
    cv2.putText(
        debug_img,
        f"Temp: {temperature:.1f}C",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        debug_img,
        f"Humidity: {humidity:.1f}%",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    return debug_img


def process_meter_image(image_path: str) -> Tuple[float, float]:
    """メーター画像を処理して温度と湿度を検出"""
    # 画像読み込み
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"画像を読み込めませんでした: {image_path}")

    # メーター中心を検出
    center = detect_meter_center(image)

    # 針の角度を検出
    temp_angle = detect_needle_angle(image, center, is_temperature=True)
    humidity_angle = detect_needle_angle(image, center, is_temperature=False)

    # 角度から値に変換
    temperature = angle_to_temperature(temp_angle)
    humidity = angle_to_humidity(humidity_angle)

    # デバッグ画像を作成
    debug_img = create_debug_image(
        image, center, temp_angle, humidity_angle, temperature, humidity
    )

    # 結果を保存
    output_path = image_path.replace(".jpg", "_result.jpg")
    cv2.imwrite(output_path, debug_img)

    # 中間データを保存
    debug_data_path = image_path.replace(".jpg", "_debug.txt")
    with open(debug_data_path, "w") as f:
        f.write(f"Center: {center}\n")
        f.write(f"Temperature angle: {temp_angle:.2f}°\n")
        f.write(f"Humidity angle: {humidity_angle:.2f}°\n")
        f.write(f"Temperature: {temperature:.1f}°C\n")
        f.write(f"Humidity: {humidity:.1f}%\n")

    return temperature, humidity


def main() -> None:
    """メイン関数"""
    parser = argparse.ArgumentParser(description="アナログメーター検出システム")
    parser.add_argument("image_path", nargs="?", help="処理する画像のパス")
    parser.add_argument(
        "--all", action="store_true", help="data/フォルダ内の全画像を処理"
    )

    args = parser.parse_args()

    if args.all:
        # 全テストデータを処理
        data_dir = Path("data")
        if not data_dir.exists():
            print("data/フォルダが見つかりません")
            return

        expected_values = {
            "meter_001.jpg": (23.5, 58),
            "meter_002.jpg": (25.5, 75),
            "meter_003.jpg": (29.5, 67),
            "meter_004.jpg": (19.0, 49),
            "meter_005.jpg": (24.0, 48),
        }

        print("アナログメーター検出結果:")
        print("=" * 50)

        for image_file in sorted(data_dir.glob("meter_*.jpg")):
            if "_result" in image_file.name:
                continue  # result画像をスキップ

            try:
                temperature, humidity = process_meter_image(str(image_file))

                # 期待値と比較
                expected = expected_values.get(image_file.name, (0, 0))
                temp_error = abs(temperature - expected[0])
                humidity_error = abs(humidity - expected[1])

                print(f"{image_file.name}:")
                print(
                    f"  温度: {temperature:.1f}°C (期待値: {expected[0]:.1f}°C, 誤差: {temp_error:.1f}°C)"
                )
                print(
                    f"  湿度: {humidity:.1f}% (期待値: {expected[1]:.1f}%, 誤差: {humidity_error:.1f}%)"
                )
                print(f"  結果画像: {image_file.stem}_result.jpg")
                print(f"  デバッグ: {image_file.stem}_debug.txt")
                print()

            except Exception as e:
                print(f"エラー: {image_file.name}: {e}")
                print()

    elif args.image_path:
        # 指定された画像を処理
        try:
            temperature, humidity = process_meter_image(args.image_path)
            print(f"温度: {temperature:.1f}°C")
            print(f"湿度: {humidity:.1f}%")
        except Exception as e:
            print(f"エラー: {e}")

    else:
        print("使用方法:")
        print("  python -m analog_meter_detection <image_path>  # 単一画像を処理")
        print("  python -m analog_meter_detection --all         # 全テストデータを処理")
