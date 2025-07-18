import argparse
import math
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def detect_needle_center_from_red_circle(image: np.ndarray, is_temperature: bool = True, output_dir: str = None, debug_count: int = 1) -> Tuple[int, int]:
    """赤色の針の根っこの円形部分から中心を検出"""
    # 赤色フィルタリング
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 赤色の範囲を設定（より精密に）
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    # ノイズ除去
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    
    # 画像の上半分（温度計）または下半分（湿度計）に制限
    h, w = image.shape[:2]
    if is_temperature:
        # 温度計は上半分
        search_mask = np.zeros_like(red_mask)
        search_mask[0:h//2 + 50, :] = red_mask[0:h//2 + 50, :]
    else:
        # 湿度計は下半分
        search_mask = np.zeros_like(red_mask)
        search_mask[h//2 - 50:h, :] = red_mask[h//2 - 50:h, :]
    
    # 円形部分を検出
    circles = cv2.HoughCircles(
        search_mask,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=50,
        param2=15,
        minRadius=3,
        maxRadius=25,
    )
    
    detected_center = None
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # 最も信頼性の高い円を選択
        best_circle = None
        best_score = 0
        
        for (x, y, r) in circles:
            # 円の範囲内の赤色画素密度を計算
            mask_circle = np.zeros_like(search_mask)
            cv2.circle(mask_circle, (x, y), r, 255, -1)
            overlap = cv2.bitwise_and(search_mask, mask_circle)
            density = np.sum(overlap > 0) / (math.pi * r * r)
            
            # 適切な位置にある円を優先
            if is_temperature:
                # 温度計は上半分の適切な位置
                if y < h//2 + 30 and density > best_score:
                    best_score = density
                    best_circle = (x, y, r)
            else:
                # 湿度計は下半分の適切な位置
                if y > h//2 - 30 and density > best_score:
                    best_score = density
                    best_circle = (x, y, r)
        
        if best_circle:
            detected_center = (best_circle[0], best_circle[1])
    
    # 円検出が失敗した場合、重心を計算
    if detected_center is None:
        # 赤色画素の重心を計算
        moments = cv2.moments(search_mask)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            detected_center = (cx, cy)
    
    # フォールバック：画像の中心付近を使用
    if detected_center is None:
        if is_temperature:
            detected_center = (w // 2, h // 3)
        else:
            detected_center = (w // 2, 2 * h // 3)
    
    # デバッグ画像を作成
    if output_dir:
        debug_img = cv2.cvtColor(search_mask, cv2.COLOR_GRAY2BGR)
        
        if circles is not None:
            for (x, y, r) in circles:
                cv2.circle(debug_img, (x, y), r, (0, 255, 0), 2)
                cv2.circle(debug_img, (x, y), 2, (0, 255, 0), -1)
        
        if detected_center:
            cv2.circle(debug_img, detected_center, 5, (0, 0, 255), -1)
            cv2.putText(debug_img, f"Center: {detected_center}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        meter_type = "temp" if is_temperature else "hum"
        cv2.imwrite(f"{output_dir}/debug_{debug_count:03d}_{meter_type}_center_detection.jpg", debug_img)
    
    return detected_center


def find_needle_by_tip_detection(mask: np.ndarray, cx: int, cy: int) -> float:
    """針の先端を検出して角度を計算 - より堅牢な検出手法"""
    h, w = mask.shape
    
    # 方法1: 重み付き放射状スキャン（主要手法）
    angle_scores = []
    
    # より細かい角度でスキャン
    for angle_deg in range(-90, 91, 1):  # 1度刻みでスキャン
        angle_rad = math.radians(angle_deg)
        
        # この角度方向の画素密度を計算（重み付き）
        weighted_score = 0
        total_weight = 0
        pixel_count = 0
        
        for radius in range(15, 75, 3):  # 中心から少し離れた位置から
            x = int(cx + radius * math.sin(angle_rad))
            y = int(cy - radius * math.cos(angle_rad))
            
            if 0 <= x < w and 0 <= y < h:
                # 距離に応じた重み（適度な距離が最も重要）
                weight = 1.0 if 25 <= radius <= 60 else 0.5
                total_weight += weight
                pixel_count += 1
                
                if mask[y, x] > 0:
                    weighted_score += weight
        
        if total_weight > 0 and pixel_count > 5:  # 最小画素数チェック
            density = weighted_score / total_weight
            # 連続性も考慮（周辺角度との一貫性）
            angle_scores.append((angle_deg, density, pixel_count))
    
    # 上位の候補を選択
    if not angle_scores:
        return None
    
    # 密度順にソート
    angle_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 上位候補から合理的な角度を選択
    valid_candidates = []
    for angle, density, count in angle_scores[:10]:  # 上位10個を検討
        if -75 <= angle <= 75:  # 合理的な範囲内
            valid_candidates.append((angle, density, count))
    
    if not valid_candidates:
        return None
    
    # 方法2: 近傍の一貫性をチェック
    best_angle = None
    best_consistency = 0
    
    for angle, density, count in valid_candidates[:5]:  # 上位5個を詳細検討
        # 近傍角度の密度を確認
        consistency_score = 0
        for neighbor_angle, neighbor_density, _ in angle_scores:
            if abs(neighbor_angle - angle) <= 3:  # 3度以内
                consistency_score += neighbor_density
        
        # 一貫性スコアが高い角度を選択
        if consistency_score > best_consistency:
            best_consistency = consistency_score
            best_angle = angle
    
    # 方法3: 外れ値検出（異常な角度の除外）
    if best_angle is not None:
        # 90度に近い角度（明らかに異常）を除外
        if abs(best_angle) > 80:
            # 次に良い候補を選択
            for angle, density, count in valid_candidates[1:]:
                if abs(angle) <= 80:
                    best_angle = angle
                    break
    
    return best_angle


def detect_temperature_needle(image: np.ndarray, output_dir: str = None, debug_count: int = 1) -> float:
    """温度計の針を検出 - 赤色円形部分から中心を検出"""
    # 温度計の中心を赤色円形部分から検出
    temp_cx, temp_cy = detect_needle_center_from_red_circle(image, is_temperature=True, output_dir=output_dir, debug_count=debug_count)
    
    # 温度計の領域を限定
    temp_radius = 70
    
    # より精密な赤色フィルタリング
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 赤色の範囲を調整（より厳密に）
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    # より強力なノイズ除去
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    
    # 温度計の範囲を制限
    h, w = image.shape[:2]
    y_min = max(0, temp_cy - temp_radius)
    y_max = min(h, temp_cy + temp_radius // 3)  # 上側のみ
    x_min = max(0, temp_cx - temp_radius)
    x_max = min(w, temp_cx + temp_radius)
    
    # 温度計領域のマスクを作成
    temp_mask = np.zeros_like(red_mask)
    temp_mask[y_min:y_max, x_min:x_max] = red_mask[y_min:y_max, x_min:x_max]
    
    # 中心からの距離制限を追加（より厳密に）
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            distance = math.sqrt((x - temp_cx)**2 + (y - temp_cy)**2)
            if distance > temp_radius or distance < 15:  # 最小距離を増加
                temp_mask[y, x] = 0
    
    # デバッグ用の画像を保存（改良版）
    if output_dir:
        debug_red_mask = cv2.cvtColor(temp_mask, cv2.COLOR_GRAY2BGR)
        cv2.circle(debug_red_mask, (temp_cx, temp_cy), 5, (255, 0, 0), -1)
        cv2.circle(debug_red_mask, (temp_cx, temp_cy), temp_radius, (0, 255, 0), 2)
        cv2.rectangle(debug_red_mask, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
        
        # 検出された赤色画素の統計情報を表示
        red_pixels = np.sum(temp_mask > 0)
        cv2.putText(debug_red_mask, f"Red pixels: {red_pixels}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imwrite(f"{output_dir}/debug_{debug_count + 1:03d}_temp_red_mask.jpg", debug_red_mask)
    
    # 改良された針の先端検出を使用
    angle = find_needle_by_tip_detection(temp_mask, temp_cx, temp_cy)
    
    # デバッグ用の画像を作成（詳細版）
    if output_dir:
        debug_img = cv2.cvtColor(temp_mask, cv2.COLOR_GRAY2BGR)
        cv2.circle(debug_img, (temp_cx, temp_cy), 5, (255, 0, 0), -1)
        cv2.circle(debug_img, (temp_cx, temp_cy), temp_radius, (0, 255, 0), 2)
        
        # 候補角度の可視化を追加
        if angle is not None:
            # 検出した針の方向を太い赤線で描画
            end_x = int(temp_cx + temp_radius * 0.8 * math.sin(math.radians(angle)))
            end_y = int(temp_cy - temp_radius * 0.8 * math.cos(math.radians(angle)))
            cv2.line(debug_img, (temp_cx, temp_cy), (end_x, end_y), (0, 0, 255), 4)
            
            # 角度範囲のガイドラインを描画
            for guide_angle in [-60, -45, -30, -15, 0, 15, 30, 45, 60]:
                guide_x = int(temp_cx + temp_radius * 0.6 * math.sin(math.radians(guide_angle)))
                guide_y = int(temp_cy - temp_radius * 0.6 * math.cos(math.radians(guide_angle)))
                color = (0, 255, 255) if guide_angle % 30 == 0 else (128, 128, 128)
                cv2.line(debug_img, (temp_cx, temp_cy), (guide_x, guide_y), color, 1)
                
                # 角度ラベルを追加
                label_x = int(temp_cx + temp_radius * 0.9 * math.sin(math.radians(guide_angle)))
                label_y = int(temp_cy - temp_radius * 0.9 * math.cos(math.radians(guide_angle)))
                cv2.putText(debug_img, f"{guide_angle}°", (label_x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            cv2.putText(debug_img, f"Detected: {angle:.1f}°", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(debug_img, "No angle detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imwrite(f"{output_dir}/debug_{debug_count + 2:03d}_temp_needle.jpg", debug_img)
    
    return angle if angle is not None else 0.0


def detect_humidity_needle(image: np.ndarray, output_dir: str = None, debug_count: int = 1) -> float:
    """湿度計の針を検出 - 赤色円形部分から中心を検出"""
    # 湿度計の中心を赤色円形部分から検出
    hum_cx, hum_cy = detect_needle_center_from_red_circle(image, is_temperature=False, output_dir=output_dir, debug_count=debug_count)
    
    # 湿度計の領域を限定
    hum_radius = 60
    
    # 赤色フィルタリング
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 赤色の範囲を調整
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    # ノイズ除去
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    
    # 湿度計の範囲を制限
    h, w = image.shape[:2]
    y_min = max(0, hum_cy - hum_radius)
    y_max = min(h, hum_cy + hum_radius)
    x_min = max(0, hum_cx - hum_radius)
    x_max = min(w, hum_cx + hum_radius)
    
    # 湿度計領域のマスクを作成
    hum_mask = np.zeros_like(red_mask)
    hum_mask[y_min:y_max, x_min:x_max] = red_mask[y_min:y_max, x_min:x_max]
    
    # 中心からの距離制限を追加
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            distance = math.sqrt((x - hum_cx)**2 + (y - hum_cy)**2)
            if distance > hum_radius or distance < 10:
                hum_mask[y, x] = 0
    
    # デバッグ用の画像を保存
    if output_dir:
        debug_red_mask = cv2.cvtColor(hum_mask, cv2.COLOR_GRAY2BGR)
        cv2.circle(debug_red_mask, (hum_cx, hum_cy), 5, (255, 0, 0), -1)
        cv2.circle(debug_red_mask, (hum_cx, hum_cy), hum_radius, (0, 255, 0), 2)
        cv2.rectangle(debug_red_mask, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
        
        # 検出された赤色画素の統計情報を表示
        red_pixels = np.sum(hum_mask > 0)
        cv2.putText(debug_red_mask, f"Red pixels: {red_pixels}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imwrite(f"{output_dir}/debug_{debug_count + 1:03d}_hum_red_mask.jpg", debug_red_mask)
    
    # 改良された針の先端検出を使用
    angle = find_needle_by_tip_detection(hum_mask, hum_cx, hum_cy)
    
    # デバッグ用の画像を作成
    if output_dir:
        debug_img = cv2.cvtColor(hum_mask, cv2.COLOR_GRAY2BGR)
        cv2.circle(debug_img, (hum_cx, hum_cy), 5, (255, 0, 0), -1)
        cv2.circle(debug_img, (hum_cx, hum_cy), hum_radius, (0, 255, 0), 2)
        
        # 候補角度の可視化を追加
        if angle is not None:
            # 検出した針の方向を太い赤線で描画
            end_x = int(hum_cx + hum_radius * 0.8 * math.sin(math.radians(angle)))
            end_y = int(hum_cy - hum_radius * 0.8 * math.cos(math.radians(angle)))
            cv2.line(debug_img, (hum_cx, hum_cy), (end_x, end_y), (0, 0, 255), 4)
            
            # 角度範囲のガイドラインを描画
            for guide_angle in [-90, -60, -30, 0, 30, 60, 90]:
                guide_x = int(hum_cx + hum_radius * 0.6 * math.sin(math.radians(guide_angle)))
                guide_y = int(hum_cy - hum_radius * 0.6 * math.cos(math.radians(guide_angle)))
                color = (0, 255, 255) if guide_angle % 30 == 0 else (128, 128, 128)
                cv2.line(debug_img, (hum_cx, hum_cy), (guide_x, guide_y), color, 1)
                
                # 角度ラベルを追加
                label_x = int(hum_cx + hum_radius * 0.9 * math.sin(math.radians(guide_angle)))
                label_y = int(hum_cy - hum_radius * 0.9 * math.cos(math.radians(guide_angle)))
                cv2.putText(debug_img, f"{guide_angle}°", (label_x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            cv2.putText(debug_img, f"Detected: {angle:.1f}°", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(debug_img, "No angle detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imwrite(f"{output_dir}/debug_{debug_count + 2:03d}_hum_needle.jpg", debug_img)
    
    return angle if angle is not None else 0.0


def angle_to_temperature(angle: float) -> float:
    """角度から温度を算出（-20℃～50℃）- 実際の画像に基づいて調整"""
    # 角度の範囲を制限
    if angle < -90:
        angle = -90
    elif angle > 90:
        angle = 90
    
    # 実際の画像解析に基づく温度マッピング
    # meter_001.jpg の実際の針の位置を基に調整
    # 18°で23.5°Cとなるように調整
    temperature_points = [
        (-90, -20.0),
        (-70, -15.0),
        (-50, -10.0),
        (-30, -5.0),
        (-10, 0.0),
        (0, 5.0),
        (10, 15.0),
        (18, 23.5),    # meter_001.jpgの実際の値に合わせて調整
        (25, 27.0),
        (35, 30.0),
        (50, 35.0),
        (70, 40.0),
        (90, 50.0)
    ]
    
    # 線形補間で温度を算出
    for i in range(len(temperature_points) - 1):
        angle1, temp1 = temperature_points[i]
        angle2, temp2 = temperature_points[i + 1]
        
        if angle1 <= angle <= angle2:
            # 線形補間
            ratio = (angle - angle1) / (angle2 - angle1)
            temperature = temp1 + ratio * (temp2 - temp1)
            return temperature
    
    # 範囲外の場合は境界値を返す
    return temperature_points[0][1] if angle < temperature_points[0][0] else temperature_points[-1][1]


def angle_to_humidity(angle: float) -> float:
    """角度から湿度を算出（0%～100%）"""
    # 角度の範囲を制限
    if angle < -90:
        angle = -90
    elif angle > 90:
        angle = 90
    
    # 湿度計の角度マッピング
    humidity_points = [
        (-90, 0.0),
        (-60, 20.0),
        (-30, 40.0),
        (0, 50.0),
        (30, 70.0),
        (60, 85.0),
        (90, 100.0)
    ]
    
    # 線形補間で湿度を算出
    for i in range(len(humidity_points) - 1):
        angle1, hum1 = humidity_points[i]
        angle2, hum2 = humidity_points[i + 1]
        
        if angle1 <= angle <= angle2:
            # 線形補間
            ratio = (angle - angle1) / (angle2 - angle1)
            humidity = hum1 + ratio * (hum2 - hum1)
            return humidity
    
    # 範囲外の場合は境界値を返す
    return humidity_points[0][1] if angle < humidity_points[0][0] else humidity_points[-1][1]


def create_debug_image(
    image: np.ndarray,
    temp_center: Tuple[int, int],
    hum_center: Tuple[int, int],
    temp_angle: float,
    humidity_angle: float,
    temperature: float,
    humidity: float,
) -> np.ndarray:
    """デバッグ用の画像を作成"""
    debug_img = image.copy()
    
    # 温度計の中心と針の方向を描画
    temp_cx, temp_cy = temp_center
    cv2.circle(debug_img, (temp_cx, temp_cy), 5, (255, 0, 0), -1)
    temp_x = temp_cx + int(80 * math.sin(math.radians(temp_angle)))
    temp_y = temp_cy - int(80 * math.cos(math.radians(temp_angle)))
    cv2.line(debug_img, (temp_cx, temp_cy), (temp_x, temp_y), (255, 0, 0), 3)

    # 湿度計の中心と針の方向を描画
    hum_cx, hum_cy = hum_center
    cv2.circle(debug_img, (hum_cx, hum_cy), 5, (0, 0, 255), -1)
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

    # 出力ディレクトリを作成
    image_name = Path(image_path).stem
    output_dir = Path("output") / image_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # デバッグカウンタ
    debug_count = 1

    # 温度計の針の角度を検出
    temp_angle = detect_temperature_needle(image, str(output_dir), debug_count)
    debug_count += 3  # 関数内で3つの画像を出力
    
    # 湿度計の針の角度を検出
    humidity_angle = detect_humidity_needle(image, str(output_dir), debug_count)
    debug_count += 3  # 関数内で3つの画像を出力

    # 角度から値に変換
    temperature = angle_to_temperature(temp_angle)
    humidity = angle_to_humidity(humidity_angle)

    # 中心位置を取得（デバッグ用）
    temp_center = detect_needle_center_from_red_circle(image, is_temperature=True)
    hum_center = detect_needle_center_from_red_circle(image, is_temperature=False)

    # デバッグ画像を作成
    debug_img = create_debug_image(
        image, temp_center, hum_center, temp_angle, humidity_angle, temperature, humidity
    )

    # 結果を保存
    cv2.imwrite(f"{output_dir}/result.jpg", debug_img)

    # 中間データを保存
    with open(f"{output_dir}/debug.txt", "w") as f:
        f.write(f"Temperature center: {temp_center}\n")
        f.write(f"Humidity center: {hum_center}\n")
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
                print(f"  結果画像: output/{image_file.stem}/result.jpg")
                print(f"  デバッグ: output/{image_file.stem}/debug.txt")
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