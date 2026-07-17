from ultralytics import YOLO
import cv2
import numpy as np
import os

# COCO 类别：2=car, 5=bus, 7=truck
VEHICLE_CLASSES = [2, 5, 7]

# 车辆检测模型（yolov8s 比 yolov8n 精度高；首次使用自动下载）
MODEL_PATH = "weights/yolov8s.pt"
# 部件检测模型（本项目训练，9 类部件，含 taillight）
PARTS_MODEL_PATH = "runs/detect/parts_v2/weights/best.pt"
# 尾灯在部件检测器类别中的索引（names=[taillight, headlight, ...]）
TAILLIGHT_CLS = 0

# 车辆框后处理过滤阈值（过滤误检）
MIN_BOX_AREA_RATIO = 0.005         # 框面积 < 图像 0.5% 视为噪点丢弃
VEHICLE_ASPECT_RANGE = (0.25, 4.0)  # 车辆宽高比合理范围

# 亮/灭灯判定阈值：尾灯裁剪图 V 通道均值（亮度）≥ 此值视为亮灯
TAILLIGHT_MIN_V = 150

# 部件检测置信度阈值
PARTS_CONF = 0.25

_MODEL_CACHE = {}


def _get_model(path):
    """缓存加载 YOLO 模型，避免重复加载（标注工具批量调用时关键）。"""
    if path not in _MODEL_CACHE:
        _MODEL_CACHE[path] = YOLO(path)
    return _MODEL_CACHE[path]


def _filter_vehicle_boxes(boxes, img_h, img_w):
    """后处理过滤误检：丢弃面积过小或宽高比异常的框。"""
    img_area = img_h * img_w
    kept = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        if bw * bh < img_area * MIN_BOX_AREA_RATIO:
            continue  # 面积过小（远车/噪点）
        aspect = bw / bh
        if aspect < VEHICLE_ASPECT_RANGE[0] or aspect > VEHICLE_ASPECT_RANGE[1]:
            continue  # 宽高比异常（细长条状多为招牌/栏杆）
        kept.append(box)
    return kept


def detect_vehicle_full(image_path, conf=0.4):
    """YOLOv8 检测车辆（yolov8s + 后处理过滤误检）"""
    model = _get_model(MODEL_PATH)
    results = model(image_path, classes=VEHICLE_CLASSES, conf=conf)
    img = results[0].orig_img
    h, w = img.shape[:2]
    regions = []

    for box in _filter_vehicle_boxes(results[0].boxes, h, w):
        cls_id = int(box.cls)
        cls_name = results[0].names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        crop = img[y1:y2, x1:x2]
        regions.append({"crop": crop, "box": (x1, y1, x2, y2), "label": cls_name, "method": "yolo_full"})

    return regions, img


def detect_taillights(vehicle_crop, conf=PARTS_CONF):
    """用部件检测器在车辆裁剪图中定位尾灯框。

    返回尾灯框列表，box 为相对车辆裁剪图的坐标 (x1,y1,x2,y2)，含 conf。
    用部件检测器替代旧的 HSV 红色分割（更准、灭灯也鲁棒）。
    """
    model = _get_model(PARTS_MODEL_PATH)
    res = model.predict(vehicle_crop, conf=conf, verbose=False)[0]
    taillights = []
    if res.boxes is None:
        return taillights
    for box in res.boxes:
        if int(box.cls) != TAILLIGHT_CLS:
            continue
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        taillights.append({"box": (x1, y1, x2, y2), "conf": float(box.conf)})
    return taillights


def _judge_lit(crop):
    """亮/灭灯判定：基于尾灯裁剪图 V 通道均值（亮度）。

    亮灯尾灯发光、亮度高；灭灯尾灯外壳暗。不依赖 HSV mask，只取亮度。
    返回 (is_lit, v_mean)。
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    v_mean = float(np.mean(hsv[:, :, 2]))
    return v_mean >= TAILLIGHT_MIN_V, v_mean


def crop_and_normalize(img_region, target_size=224):
    """裁剪区域归一化：保持宽高比 + padding + resize"""
    h, w = img_region.shape[:2]
    size = max(w, h)
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    padded[(size - h) // 2:(size + h) // 2, (size - w) // 2:(size + w) // 2] = img_region
    return cv2.resize(padded, (target_size, target_size))


def detect_tail_light(image_path, conf=0.4):
    """主流程：车辆检测 -> 部件检测器定位尾灯 -> 归一化+亮灭判定 -> 可视化。

    旧版用 HSV 红色分割 + 左右对称配对定位尾灯，实测不准（灭灯/红色干扰），
    改用训练好的部件检测器直接定位尾灯框（taillight 类 mAP50=0.99）。

    所有输出（裁剪图/结果图）统一写入 {输入图名}_result/ 文件夹。
    """

    base_name = os.path.splitext(image_path)[0]
    stem = os.path.basename(base_name)
    out_dir = f"{base_name}_result"
    os.makedirs(out_dir, exist_ok=True)

    # 步骤1：检测完整车辆
    print("=" * 50)
    print("步骤1：YOLOv8 检测车辆...")
    vehicle_regions, img = detect_vehicle_full(image_path, conf=conf)

    all_regions = []

    if not vehicle_regions:
        print("  未检测到车辆")
    else:
        print(f"  检测到 {len(vehicle_regions)} 辆车，用部件检测器定位尾灯")

        for idx, v in enumerate(vehicle_regions):
            vx1, vy1, vx2, vy2 = v["box"]
            print(f"  车辆 [{vx1},{vy1},{vx2},{vy2}] - {v['label']}")

            vehicle_crop = img[vy1:vy2, vx1:vx2]
            # 步骤2：部件检测器在车辆内找尾灯
            taillights = detect_taillights(vehicle_crop)

            if not taillights:
                print("    未检测到尾灯")
                continue

            print(f"    检测到 {len(taillights)} 个尾灯")
            for t in taillights:
                tx1, ty1, tx2, ty2 = t["box"]
                # 映射回原图坐标
                ox1, oy1 = vx1 + tx1, vy1 + ty1
                ox2, oy2 = vx1 + tx2, vy1 + ty2
                tail_crop = img[oy1:oy2, ox1:ox2]
                # 左/右尾灯：中心在车辆左半边=左尾灯
                cx_center = (ox1 + ox2) / 2
                vehicle_center = (vx1 + vx2) / 2
                side = "left" if cx_center < vehicle_center else "right"
                is_lit, v_mean = _judge_lit(tail_crop)
                all_regions.append({
                    "crop": tail_crop,
                    "box": (ox1, oy1, ox2, oy2),
                    "label": f"{side}_taillight",
                    "method": "parts",
                    "is_lit": is_lit,
                    "conf": t["conf"],
                })

    # 汇总保存结果
    print("\n" + "=" * 50)
    result_img = img.copy()

    saved_files = []
    for i, region in enumerate(all_regions):
        label = region["label"]
        x1, y1, x2, y2 = region["box"]
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

        print(f"  [{label}] conf={region.get('conf', 0):.2f}, 亮灯={is_lit}, 保存={crop_path}")

    result_path = os.path.join(out_dir, f"{stem}_result.jpg")
    cv2.imwrite(result_path, result_img)
    print(f"\n结果图已保存: {result_path}")
    print(f"所有输出文件位于: {out_dir}/")

    if not all_regions:
        print("未检测到尾灯")
    else:
        print(f"共检测到 {len(all_regions)} 个尾灯区域")

    return all_regions


if __name__ == "__main__":
    image_path = input("请输入图片路径（如 car.png）: ").strip().strip('"')
    detect_tail_light(image_path)
