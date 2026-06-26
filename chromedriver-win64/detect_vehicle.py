from ultralytics import YOLO
import cv2
import numpy as np
import os

# COCO 类别：2=car, 5=bus, 7=truck
VEHICLE_CLASSES = [2, 5, 7]

# 尾灯判定阈值：候选区域框内红色像素占比低于此值视为无效（过滤红色衣物/招牌/建筑等误检）
TAILLIGHT_MIN_RED_RATIO = 0.10

# 尾灯配对验证阈值（策略2兜底用，区分尾灯与风景中的红色物体）
TAILLIGHT_MIN_V = 150          # 亮灯判定阈值；灭灯尾灯不发光，配对不再强制要求
TAILLIGHT_MAX_ASPECT = 8.0     # 尾灯长宽比上限（过细的条状红色多为招牌/栏杆）
TAILLIGHT_MIN_ASPECT = 0.15    # 长宽比下限
PAIR_Y_TOL_RATIO = 2.5         # 配对两灯中心y坐标差 / 灯高 的容差（放松，适应倾斜/俯视角度）
PAIR_AREA_RATIO_MAX = 8.0      # 配对两灯面积比上限（大小应相近）

# 是否保存 HSV/mask/形态学等中间过程图（汇报展示用）
SAVE_DEBUG_IMAGES = True

# 是否额外输出边缘检测图（Canny/Sobel，汇报对比展示用）
SAVE_EDGE_IMAGES = True


def _save_debug_images(prefix, original, hsv, raw_mask, morph_mask):
    """保存红色分割流程的中间过程图，用于汇报展示。"""
    TH = 320  # 缩略高度，便于拼图

    def to_th(img):
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * TH / h), TH))

    # 01 原图（车辆ROI或全图）
    cv2.imwrite(f"{prefix}_01_original.jpg", original)
    # 02 HSV 表示图
    cv2.imwrite(f"{prefix}_02_hsv.jpg", hsv)
    # 03 原始红色mask（红on黑，3通道）
    raw_color = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)
    raw_color[:, :, 0:2] = 0   # 置零B、G，保留R=红
    cv2.imwrite(f"{prefix}_03_red_mask_raw.jpg", raw_color)
    # 04 形态学后mask
    morph_color = cv2.cvtColor(morph_mask, cv2.COLOR_GRAY2BGR)
    morph_color[:, :, 0:2] = 0
    cv2.imwrite(f"{prefix}_04_red_mask_morph.jpg", morph_color)
    # 05 mask叠加原图（核心展示图：红色高亮尾灯位置）
    overlay = original.copy()
    overlay[morph_mask > 0] = (0, 0, 255)
    blended = cv2.addWeighted(original, 0.6, overlay, 0.4, 0)
    cv2.imwrite(f"{prefix}_05_mask_overlay.jpg", blended)
    # 06 流程拼图（PPT单页展示用：原图|HSV|原始mask|形态学mask|叠加）
    montage = np.hstack([to_th(x) for x in [original, hsv, raw_color, morph_color, blended]])
    cv2.imwrite(f"{prefix}_06_montage.jpg", montage)

    # 边缘检测对比图（汇报展示：为何选颜色分割而非边缘检测）
    if SAVE_EDGE_IMAGES:
        canny, sobel = _compute_edges(original)
        # 07 Canny 边缘图
        cv2.imwrite(f"{prefix}_07_canny.jpg", canny)
        # 08 Sobel 边缘图
        cv2.imwrite(f"{prefix}_08_sobel.jpg", sobel)
        # 09 颜色mask轮廓(绿)叠在Canny上，对比两种方法
        canny_color = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(morph_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canny_color, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(f"{prefix}_09_color_vs_canny.jpg", canny_color)
        # 10 边缘+颜色 融合拼图：原图|Canny|Sobel|颜色mask|对比
        canny_c = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        sobel_c = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
        montage2 = np.hstack([to_th(original), to_th(canny_c), to_th(sobel_c),
                              to_th(morph_color), to_th(canny_color)])
        cv2.imwrite(f"{prefix}_10_edge_montage.jpg", montage2)


def _build_red_mask(crop, debug_prefix=None):
    """HSV红色分割 + 形态学去噪。

    返回 (hsv, raw_mask, morph_mask)。
    若提供 debug_prefix 且 SAVE_DEBUG_IMAGES=True，保存中间过程图。
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # 红色在 HSV 中分两段
    mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    raw_mask = mask1 | mask2

    # 形态学操作去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_CLOSE, kernel)

    if debug_prefix and SAVE_DEBUG_IMAGES:
        _save_debug_images(debug_prefix, crop, hsv, raw_mask, morph_mask)

    return hsv, raw_mask, morph_mask


def _compute_edges(crop):
    """对输入图做灰度+Canny+Sobel边缘检测，返回 (canny, sobel) 两张单通道边缘图(0/255)。"""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny：高低阈值自适应（中位数的0.66/1.33倍，常见实践）
    v = np.median(gray_blur)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    canny = cv2.Canny(gray_blur, lower, upper)

    # Sobel：x/y方向梯度合成
    sx = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(np.sqrt(sx ** 2 + sy ** 2))
    _, sobel_bin = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)

    return canny, sobel_bin


def detect_vehicle_full(image_path, conf=0.5):
    """策略1：YOLOv8 预训练模型检测完整车辆"""
    model = YOLO("yolov8n.pt")
    results = model(image_path, classes=VEHICLE_CLASSES, conf=conf)
    img = results[0].orig_img
    regions = []

    for box in results[0].boxes:
        cls_id = int(box.cls)
        cls_name = results[0].names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        crop = img[y1:y2, x1:x2]
        regions.append({"crop": crop, "box": (x1, y1, x2, y2), "label": cls_name, "method": "yolo_full"})

    return regions, img


def _extract_red_candidates(hsv, red_mask, area_source=None):
    """从红色mask中提取候选尾灯区域，只做密度过滤（宽松，避免漏检）。

    area_source: ('image', h, w) 或 ('vehicle', x1, y1, x2, y2) 用于面积阈值与坐标映射。
    长宽比/亮度等更严格的筛选留给 _find_taillight_pair（仅策略2兜底使用）。
    返回候选列表，每个含 bbox(原图绝对坐标)、亮度、面积、宽高比、red_ratio。
    """
    src_kind, *args = area_source
    if src_kind == "image":
        h, w = args
        ox, oy = 0, 0
    else:
        ox, oy, x2, y2 = args
        h, w = y2 - oy, x2 - ox

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = h * w * 0.001
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]

    candidates = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        own_mask = red_mask[y:y + ch, x:x + cw]
        own_red_ratio = np.sum(own_mask > 0) / (own_mask.size) if own_mask.size > 0 else 0
        if own_red_ratio < TAILLIGHT_MIN_RED_RATIO:
            continue

        aspect = cw / ch if ch > 0 else 0
        roi_v = hsv[y:y + ch, x:x + cw, 2]
        v_mean = float(np.mean(roi_v)) if roi_v.size > 0 else 0

        candidates.append({
            "ox": ox + x, "oy": oy + y, "ow": cw, "oh": ch,
            "cx": ox + x + cw / 2, "cy": oy + y + ch / 2,
            "area": cv2.contourArea(cnt),
            "aspect": aspect,
            "v_mean": v_mean,
            "red_ratio": own_red_ratio,
        })
    return candidates


def _find_taillight_pair(candidates, ref_center_x):
    """从候选中找一对左右对称的尾灯。

    要求：分居中线两侧、高度接近、大小相近。
    亮度作为打分项而非硬约束（灭灯尾灯不发光，亮度低也应是尾灯）。
    返回 (left, right) 或 None。
    """
    n = len(candidates)
    best = None
    best_score = -1
    for i in range(n):
        for j in range(i + 1, n):
            a, b = candidates[i], candidates[j]
            # 分居中线两侧
            if a["cx"] < ref_center_x <= b["cx"]:
                left, right = a, b
            elif b["cx"] < ref_center_x <= a["cx"]:
                left, right = b, a
            else:
                continue
            # 长宽比过滤（过细条状为招牌/栏杆）
            bad_aspect = any(
                c["aspect"] < TAILLIGHT_MIN_ASPECT or c["aspect"] > TAILLIGHT_MAX_ASPECT
                for c in (left, right)
            )
            if bad_aspect:
                continue
            # 高度接近
            tol = max(left["oh"], right["oh"]) * PAIR_Y_TOL_RATIO
            if abs(left["cy"] - right["cy"]) > tol:
                continue
            # 大小相近
            big, small = max(left["area"], right["area"]), min(left["area"], right["area"])
            if big / small > PAIR_AREA_RATIO_MAX:
                continue
            # 综合打分：高度越齐、面积越近、亮度越高越好（亮度仅加分，不强制）
            y_score = 1 - abs(left["cy"] - right["cy"]) / (tol + 1e-6)
            area_score = small / big
            v_score = max(left["v_mean"], right["v_mean"]) / 255
            score = y_score * 0.45 + area_score * 0.45 + v_score * 0.10
            if score > best_score:
                best_score, best = score, (left, right)
    return best


def detect_tailight_by_color(image_path, out_dir=None, stem=None):
    """策略2：基于HSV红色检测直接定位尾灯区域（不依赖完整车辆检测）

    关键改进：单看颜色无法区分尾灯与风景中的红色物体，
    因此要求检测到【成对、左右对称、高度接近、亮度足够】的红色区域才算尾灯。
    out_dir/stem：指定中间图输出目录与文件名前缀；不传则按输入图名输出到同目录。
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图片 {image_path}")
        return [], img, None

    h, w = img.shape[:2]
    if stem is None:
        stem = os.path.basename(os.path.splitext(image_path)[0])
    prefix = f"{stem}_s2"
    if out_dir:
        prefix = os.path.join(out_dir, prefix)
    hsv, _, red_mask = _build_red_mask(img, debug_prefix=prefix)

    candidates = _extract_red_candidates(hsv, red_mask, ("image", h, w))
    pair = _find_taillight_pair(candidates, ref_center_x=w / 2)

    regions = []
    if pair is None:
        return regions, img, red_mask

    # 亮灯判定：灭灯尾灯不发光但仍是红色外壳，用亮度与红色密度综合判定
    max_v = max(pair[0]["v_mean"], pair[1]["v_mean"])
    max_red = max(pair[0]["red_ratio"], pair[1]["red_ratio"])
    is_lit = max_v >= TAILLIGHT_MIN_V or max_red > 0.25
    for side, c in (("left", pair[0]), ("right", pair[1])):
        x, y, cw, ch = c["ox"], c["oy"], c["ow"], c["oh"]
        pad_x = int(cw * 0.5)
        pad_y = int(ch * 0.5)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + cw + pad_x)
        y2 = min(h, y + ch + pad_y)
        crop = img[y1:y2, x1:x2]
        regions.append({
            "crop": crop,
            "box": (x1, y1, x2, y2),
            "label": f"{side}_taillight",
            "method": "color",
            "red_ratio": c["red_ratio"],
            "is_lit": is_lit,
        })

    return regions, img, red_mask


def crop_and_normalize(img_region, target_size=224):
    """裁剪区域归一化：保持宽高比 + padding + resize"""
    h, w = img_region.shape[:2]
    size = max(w, h)
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    padded[(size - h) // 2:(size + h) // 2, (size - w) // 2:(size + w) // 2] = img_region
    return cv2.resize(padded, (target_size, target_size))


def detect_tail_light(image_path, conf=0.25):
    """主流程：先尝试YOLOv8检测完整车辆，再在车辆内找尾灯；
    若检测不到完整车辆，直接用颜色检测定位尾灯区域。

    所有输出（中间过程图/裁剪图/结果图）统一写入 {输入图名}_result/ 文件夹。
    """

    base_name = os.path.splitext(image_path)[0]
    stem = os.path.basename(base_name)
    out_dir = f"{base_name}_result"
    os.makedirs(out_dir, exist_ok=True)

    # 策略1：检测完整车辆
    print("=" * 50)
    print("策略1：YOLOv8 检测完整车辆...")
    vehicle_regions, img = detect_vehicle_full(image_path, conf=conf)

    all_regions = []

    if vehicle_regions:
        print(f"  检测到 {len(vehicle_regions)} 辆车，在车辆区域内查找尾灯")

        for idx, v in enumerate(vehicle_regions):
            x1, y1, x2, y2 = v["box"]
            print(f"  车辆 [{x1},{y1},{x2},{y2}] - {v['label']}")

            # 在车辆区域内做红色检测
            vehicle_crop = img[y1:y2, x1:x2]
            hsv, _, red_mask = _build_red_mask(
                vehicle_crop, debug_prefix=os.path.join(out_dir, f"{stem}_s1_v{idx}"))

            candidates = _extract_red_candidates(
                hsv, red_mask, ("vehicle", x1, y1, x2, y2))

            if not candidates:
                print("    未检测到有效尾灯（车辆区域内无足够红色区域）")
                continue

            print(f"    在车辆区域内检测到 {len(candidates)} 个尾灯候选")
            for c in candidates:
                cx, cy, cw2, ch2 = c["ox"], c["oy"], c["ow"], c["oh"]
                tail_crop = img[cy:cy + ch2, cx:cx + cw2]
                # 位置判断：尾灯中心在车辆左半边=左尾灯，右半边=右尾灯
                cx_center = cx + cw2 / 2
                vehicle_center = (x1 + x2) / 2
                side = "left" if cx_center < vehicle_center else "right"
                # 宽松亮灯判定：红色密度高 或 亮度足够 即视为亮灯
                is_lit = c["red_ratio"] > 0.15 or c["v_mean"] >= TAILLIGHT_MIN_V
                all_regions.append({
                    "crop": tail_crop,
                    "box": (cx, cy, cx + cw2, cy + ch2),
                    "label": f"{side}_taillight",
                    "method": "vehicle_color",
                    "is_lit": is_lit
                })
    else:
        print("  未检测到完整车辆")

    # 策略2：直接用颜色检测（不依赖完整车辆）
    print("\n策略2：HSV颜色直接检测尾灯区域...")
    color_regions, _, _ = detect_tailight_by_color(image_path, out_dir=out_dir, stem=stem)

    if color_regions:
        print(f"  检测到 {len(color_regions)} 个有效尾灯区域")
        for cr in color_regions:
            all_regions.append(cr)
    else:
        print("  未检测到有效尾灯区域（未找到成对对称的红色区域）")

    # 汇总保存结果
    print("\n" + "=" * 50)
    result_img = img.copy()

    saved_files = []
    for i, region in enumerate(all_regions):
        label = region["label"]
        x1, y1, x2, y2 = region["box"]
        method = region["method"]
        is_lit = region.get("is_lit", None)

        # 归一化到 224×224
        normalized = crop_and_normalize(region["crop"], 224)
        crop_path = os.path.join(out_dir, f"{stem}_{label}_{i}.jpg")
        cv2.imwrite(crop_path, normalized)
        saved_files.append(crop_path)

        # 在结果图上画框
        color = (0, 0, 255) if (is_lit is True) else (0, 255, 0) if (is_lit is False) else (255, 255, 0)
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        status = "LIT" if is_lit else "OFF" if is_lit is False else "?"
        cv2.putText(result_img, f"{label} [{status}]", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        print(f"  [{label}] 方法={method}, 亮灯={is_lit}, 保存={crop_path}")

    result_path = os.path.join(out_dir, f"{stem}_result.jpg")
    cv2.imwrite(result_path, result_img)
    print(f"\n结果图已保存: {result_path}")
    print(f"所有输出文件位于: {out_dir}/")

    if not all_regions:
        print("未检测到车灯")
    else:
        print(f"共检测到 {len(all_regions)} 个尾灯区域")

    return all_regions


if __name__ == "__main__":
    image_path = input("请输入图片路径（如 car.png）: ").strip().strip('"')
    detect_tail_light(image_path)
