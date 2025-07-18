import argparse
import math
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def detect_meter_center(image: np.ndarray, debug_path: str = None) -> Tuple[int, int]:
    """メーターの中心を検出する"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 円検出のためのパラメータ調整
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=80,
        param1=50,
        param2=30,
        minRadius=60,
        maxRadius=150,
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        # デバッグ用の画像を作成
        if debug_path:
            debug_img = image.copy()
            for (x, y, r) in circles:
                cv2.circle(debug_img, (x, y), r, (0, 255, 0), 2)
                cv2.circle(debug_img, (x, y), 5, (0, 0, 255), -1)
            cv2.imwrite(f"{debug_path}_circles.jpg", debug_img)
        
        # 画像の中心に最も近い円を選択
        h, w = image.shape[:2]
        img_center = (w // 2, h // 2)
        
        best_circle = None
        min_distance = float('inf')
        
        for circle in circles:
            x, y, r = circle
            distance = math.sqrt((x - img_center[0])**2 + (y - img_center[1])**2)
            if distance < min_distance:
                min_distance = distance
                best_circle = circle
        
        if best_circle is not None:
            return int(best_circle[0]), int(best_circle[1])

    # 円が検出されない場合は画像中心を返す
    h, w = image.shape[:2]
    return w // 2, h // 2


def detect_temperature_needle(image: np.ndarray, cx: int, cy: int, debug_path: str = None) -> float:
    """温度計の針を検出"""
    # 温度計の中心を実際の画像に合わせて調整
    # 実際の画像を見ると、温度計は全体の中心の少し上にある
    temp_cx = cx
    temp_cy = cy - 35  # 50から35に調整

    # 複数の手法で針を検出
    # 1. 放射線探索アルゴリズム（最も確実）
    angle = detect_needle_by_radial_search(image, temp_cx, temp_cy, is_temperature=True, debug_path=debug_path)
    if angle != 0.0:
        return angle

    # 2. 赤色フィルタリング + 輪郭検出
    angle = detect_needle_by_contour(image, temp_cx, temp_cy, is_temperature=True, debug_path=debug_path)
    if angle != 0.0:
        return angle

    # 3. 線検出アルゴリズム
    angle = detect_needle_by_lines(image, temp_cx, temp_cy, is_temperature=True, debug_path=debug_path)
    return angle


def detect_humidity_needle(image: np.ndarray, cx: int, cy: int, debug_path: str = None) -> float:
    """湿度計の針を検出"""
    # 湿度計の中心を実際の画像に合わせて調整
    # 実際の画像を見ると、湿度計は全体の中心の少し下にある
    hum_cx = cx
    hum_cy = cy + 35  # 50から35に調整

    # 複数の手法で針を検出
    # 1. 放射線探索アルゴリズム（最も確実）
    angle = detect_needle_by_radial_search(image, hum_cx, hum_cy, is_temperature=False, debug_path=debug_path)
    if angle != 0.0:
        return angle

    # 2. 赤色フィルタリング + 輪郭検出
    angle = detect_needle_by_contour(image, hum_cx, hum_cy, is_temperature=False, debug_path=debug_path)
    if angle != 0.0:
        return angle

    # 3. 線検出アルゴリズム
    angle = detect_needle_by_lines(image, hum_cx, hum_cy, is_temperature=False, debug_path=debug_path)
    return angle


def detect_needle_by_contour(
    image: np.ndarray, cx: int, cy: int, is_temperature: bool, debug_path: str = None
) -> float:
    """輪郭検出による針検出"""
    # 赤色を検出
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # より幅広い赤色範囲を設定
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    # ノイズ除去
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

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

    # デバッグ用の画像を保存
    if debug_path:
        # 赤色フィルタリング結果を保存
        debug_red_mask = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
        # 検出範囲を矩形で表示
        cv2.rectangle(debug_red_mask, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # 中心点を表示
        cv2.circle(debug_red_mask, (cx, cy), 5, (255, 0, 0), -1)
        
        meter_type = "temp" if is_temperature else "hum"
        cv2.imwrite(f"{debug_path}_{meter_type}_red_mask.jpg", debug_red_mask)

    # 輪郭検出で針を見つける
    roi = red_mask[y_min:y_max, x_min:x_max]
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 最も大きな輪郭を選択
        largest_contour = max(contours, key=cv2.contourArea)

        # 輪郭の面積をチェック
        area = cv2.contourArea(largest_contour)
        if area > 20:  # 面積の閾値を調整
            # 輪郭の中心を計算
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                # 輪郭の重心を計算
                contour_cx = int(M["m10"] / M["m00"]) + x_min
                contour_cy = int(M["m01"] / M["m00"]) + y_min
                
                # 針の角度を計算（中心から輪郭の重心への方向）
                angle = math.atan2(contour_cx - cx, cy - contour_cy) * 180 / math.pi
                
                # 楕円フィッティングも試す
                if len(largest_contour) >= 5:
                    try:
                        ellipse = cv2.fitEllipse(largest_contour)
                        ellipse_angle = ellipse[2] - 90  # 12時方向を0度とする
                        
                        # 角度を-180度から180度に正規化
                        ellipse_angle = ellipse_angle % 360
                        if ellipse_angle > 180:
                            ellipse_angle -= 360
                        
                        # 重心ベースの角度と楕円ベースの角度の両方を考慮
                        # 角度の差が大きい場合は重心ベースを使用
                        if abs(angle - ellipse_angle) > 90:
                            return angle
                        else:
                            return ellipse_angle
                    except cv2.error:
                        return angle
                
                return angle

    return 0.0


def detect_needle_by_lines(
    image: np.ndarray, cx: int, cy: int, is_temperature: bool, debug_path: str = None
) -> float:
    """線検出による針検出"""
    # 赤色を検出
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # より幅広い赤色範囲を設定
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    # ノイズ除去
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

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

    # デバッグ用の画像を保存
    if debug_path:
        meter_type = "temp" if is_temperature else "hum"
        cv2.imwrite(f"{debug_path}_{meter_type}_edges.jpg", edges)

    # 線検出（パラメータを調整）
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=10, minLineLength=15, maxLineGap=8
    )

    if lines is not None:
        # 中心に最も近い線を選択
        best_line = None
        min_distance = float('inf')
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 線の中点を計算
            mid_x = (x1 + x2) / 2 + x_min
            mid_y = (y1 + y2) / 2 + y_min
            
            # 中心からの距離を計算
            distance = math.sqrt((mid_x - cx)**2 + (mid_y - cy)**2)
            
            if distance < min_distance:
                min_distance = distance
                best_line = line[0]

        if best_line is not None:
            x1, y1, x2, y2 = best_line
            
            # 針の角度を計算（12時を0度とする）
            angle = math.atan2(x2 - x1, y1 - y2) * 180 / math.pi
            return angle

    return 0.0


def detect_needle_by_radial_search(
    image: np.ndarray, cx: int, cy: int, is_temperature: bool, debug_path: str = None
) -> float:
    """放射線探索による針検出"""
    # 赤色を検出
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # より幅広い赤色範囲を設定
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    # ノイズ除去
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    # 検出パラメータ
    if is_temperature:
        radius = 80
        start_radius = 10
    else:
        radius = 60
        start_radius = 8

    # 針の角度を探す
    best_angle = 0.0
    max_red_pixels = 0
    
    # デバッグ用の画像を作成
    if debug_path:
        debug_img = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
        cv2.circle(debug_img, (cx, cy), 5, (255, 0, 0), -1)
        cv2.circle(debug_img, (cx, cy), radius, (0, 255, 0), 2)

    # 角度を2度刻みで探索（より細かく探索）
    for angle_deg in range(-140, 141, 2):
        angle_rad = math.radians(angle_deg)
        red_count = 0

        # 針の方向に沿って赤い画素を数える
        for r in range(start_radius, radius):
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

    # デバッグ用の画像を保存
    if debug_path:
        # 検出した針の方向を描画
        if best_angle != 0:
            end_x = int(cx + radius * math.sin(math.radians(best_angle)))
            end_y = int(cy - radius * math.cos(math.radians(best_angle)))
            cv2.line(debug_img, (cx, cy), (end_x, end_y), (0, 0, 255), 3)
        
        # 検出した赤い画素数を表示
        cv2.putText(debug_img, f"Red pixels: {max_red_pixels}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_img, f"Angle: {best_angle:.1f}°", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        meter_type = "temp" if is_temperature else "hum"
        cv2.imwrite(f"{debug_path}_{meter_type}_radial.jpg", debug_img)

    return best_angle if max_red_pixels > 5 else 0.0


def detect_needle_angle(
    image: np.ndarray, center: Tuple[int, int], is_temperature: bool = True, debug_path: str = None
) -> float:
    """針の角度を検出する（度単位）"""
    cx, cy = center

    # 実際の画像を分析して最適な針検出を行う
    if is_temperature:
        # 温度計の針検出（画像の上半分）
        return detect_temperature_needle(image, cx, cy, debug_path)
    else:
        # 湿度計の針検出（画像の下半分）
        return detect_humidity_needle(image, cx, cy, debug_path)


def angle_to_temperature(angle: float) -> float:
    """角度から温度を算出（-20℃～50℃）"""
    # 実際の画像に基づいて角度マッピングを調整
    # 温度計の針の位置を見ると、23.5°Cは約60度程度の位置にある
    # 一般的な温度計の角度範囲を考慮して調整
    
    # 針の角度が正しく読み取れていない可能性があるため、角度の解釈を見直す
    # 実際の画像では、針が右上方向（約45度付近）を指している場合が23.5°C
    
    # 角度の範囲を調整: -90度～90度の範囲で-20℃～50℃にマッピング
    if angle < -90:
        angle = -90
    elif angle > 90:
        angle = 90

    # 角度を0～1の範囲に正規化
    normalized_angle = (angle + 90) / 180

    # 温度範囲（-20℃～50℃）にマッピング
    temperature = -20 + normalized_angle * 70

    return temperature


def angle_to_humidity(angle: float) -> float:
    """角度から湿度を算出（0%～100%）"""
    # 実際の画像に基づいて角度マッピングを調整
    # 湿度計の針の位置を見ると、58%は約60度程度の位置にある
    
    # 角度の範囲を調整: -90度～90度の範囲で0%～100%にマッピング
    if angle < -90:
        angle = -90
    elif angle > 90:
        angle = 90

    # 角度を0～1の範囲に正規化
    normalized_angle = (angle + 90) / 180

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

    # 温度計の中心と針の方向を描画
    temp_cx, temp_cy = cx, cy - 35
    cv2.circle(debug_img, (temp_cx, temp_cy), 3, (255, 0, 0), -1)
    temp_x = temp_cx + int(80 * math.sin(math.radians(temp_angle)))
    temp_y = temp_cy - int(80 * math.cos(math.radians(temp_angle)))
    cv2.line(debug_img, (temp_cx, temp_cy), (temp_x, temp_y), (255, 0, 0), 3)

    # 湿度計の中心と針の方向を描画
    hum_cx, hum_cy = cx, cy + 35
    cv2.circle(debug_img, (hum_cx, hum_cy), 3, (0, 0, 255), -1)
    humidity_x = hum_cx + int(60 * math.sin(math.radians(humidity_angle)))
    humidity_y = hum_cy - int(60 * math.cos(math.radians(humidity_angle)))
    cv2.line(debug_img, (hum_cx, hum_cy), (humidity_x, humidity_y), (0, 0, 255), 3)

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

    # デバッグ用のパスを作成
    debug_path = image_path.replace(".jpg", "_debug")

    # メーター中心を検出
    center = detect_meter_center(image, debug_path)
    
    # 中心検出結果のデバッグ画像を作成
    center_debug_img = image.copy()
    cv2.circle(center_debug_img, center, 5, (0, 255, 0), -1)
    cv2.circle(center_debug_img, center, 150, (0, 255, 0), 2)
    # 温度計と湿度計の調整された中心も表示
    temp_center = (center[0], center[1] - 35)
    hum_center = (center[0], center[1] + 35)
    cv2.circle(center_debug_img, temp_center, 5, (255, 0, 0), -1)
    cv2.circle(center_debug_img, hum_center, 5, (0, 0, 255), -1)
    cv2.circle(center_debug_img, temp_center, 80, (255, 0, 0), 2)
    cv2.circle(center_debug_img, hum_center, 60, (0, 0, 255), 2)
    cv2.putText(center_debug_img, "Main Center", (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(center_debug_img, "Temp Center", (temp_center[0] + 10, temp_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(center_debug_img, "Hum Center", (hum_center[0] + 10, hum_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imwrite(f"{debug_path}_center.jpg", center_debug_img)

    # 針の角度を検出
    temp_angle = detect_needle_angle(image, center, is_temperature=True, debug_path=debug_path)
    humidity_angle = detect_needle_angle(image, center, is_temperature=False, debug_path=debug_path)

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
        f.write(f"Temperature center: ({center[0]}, {center[1] - 35})\n")
        f.write(f"Humidity center: ({center[0]}, {center[1] + 35})\n")
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
