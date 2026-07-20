# -*- coding: utf-8 -*-
"""交互式标注工具。

读 data/raw/{车型}/ 图片，自动检测车辆（绿框参考），人工标部件框 + 选朝向，
保存到 data/annotations/。

用法（daily 环境）:
    python annotate_tool.py

操作:
    - 左键拖动: 画当前选中类别的部件框
    - 删除最后部件框 / 保存 / 上一张(a) / 下一张(d) / 保存(s)

注: 尾灯自动检测（HSV）不准已移除，尾灯与其他部件一样人工画框；朝向人工选。
"""
import glob
import json
import os
import sys
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from detect_vehicle import detect_vehicle_full

DATA_DIR = os.path.join(HERE, "data", "raw")
OUT_DIR = os.path.join(HERE, "data", "annotations")
QUEUE_FILE = os.path.join(HERE, "data", "annotate_queue.json")
os.makedirs(OUT_DIR, exist_ok=True)

PART_LABELS = ["taillight", "headlight", "mirror", "window", "wheel",
               "plate", "grille", "bumper", "exhaust"]
PART_COLORS = {
    "taillight": (0, 0, 255), "headlight": (255, 255, 0),
    "mirror": (255, 0, 255), "window": (0, 255, 255),
    "wheel": (128, 128, 0), "plate": (0, 128, 255),
    "grille": (255, 128, 0), "bumper": (128, 0, 128),
    "exhaust": (200, 200, 0),
}
VIEW_LABELS = ["rear", "front", "side"]


def scan_images():
    imgs = []
    for model_dir in sorted(glob.glob(os.path.join(DATA_DIR, "*"))):
        if not os.path.isdir(model_dir):
            continue
        model = os.path.basename(model_dir)
        for p in sorted(glob.glob(os.path.join(model_dir, "*"))):
            if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                imgs.append((model, p))
    return imgs


def detect(path):
    """跑车辆检测，返回主车框（面积最大，近车优先；朝向人工标）。"""
    regions, img = detect_vehicle_full(path)
    if not regions:
        return {"vehicle_box": None}
    main = max(regions, key=lambda r: (r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1]))
    x1, y1, x2, y2 = main["box"]
    return {"vehicle_box": [x1, y1, x2, y2]}


def imread_unicode(path):
    """cv2.imread 不支持中文路径，用 imdecode+fromfile 替代。"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


class AnnoTool:
    def __init__(self, root):
        self.root = root
        self.images = scan_images()
        self.idx = 0
        self.cur = None
        self.parts = []
        self.view_var = tk.StringVar(value="")
        self.model_var = tk.StringVar(value="")
        self.drawing = False
        self.start = None
        self.rect_id = None
        self.scale = 1.0
        self.tk_img = None
        self.queue = self._load_queue()
        self.queue_idx = -1
        self._build_ui()
        if self.images:
            self.idx = self.first_unannotated()
            self.load(self.idx)
        else:
            self.status.config(text=f"未在 {DATA_DIR} 找到图片")
        if self.queue:
            print(f"标注队列: {len(self.queue)} 张需增强（按 n 跳到下一张）")

    def _build_ui(self):
        self.canvas = tk.Canvas(self.root, width=820, height=620, bg="gray")
        self.canvas.grid(row=0, column=0, rowspan=20, padx=5, pady=5)
        self.canvas.bind("<ButtonPress-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)

        r = 0
        tk.Label(self.root, text="车型:").grid(row=r, column=1, sticky="w"); r += 1
        tk.Entry(self.root, textvariable=self.model_var, state="readonly",
                 width=22).grid(row=r, column=1); r += 1
        tk.Label(self.root, text="朝向:").grid(row=r, column=1, sticky="w"); r += 1
        for v in VIEW_LABELS:
            tk.Radiobutton(self.root, text=v, value=v,
                           variable=self.view_var).grid(row=r, column=1, sticky="w")
            r += 1
        tk.Label(self.root, text="部件类别:").grid(row=r, column=1, sticky="w"); r += 1
        self.part_list = tk.Listbox(self.root, height=len(PART_LABELS), width=20)
        for p in PART_LABELS:
            self.part_list.insert("end", p)
        self.part_list.grid(row=r, column=1); self.part_list.selection_set(0); r += 1
        tk.Button(self.root, text="删除最后部件框",
                  command=self.del_last).grid(row=r, column=1, pady=2); r += 1
        tk.Button(self.root, text="保存 (S)", command=self.save).grid(row=r, column=1, pady=2); r += 1
        tk.Button(self.root, text="上一张 (A)",
                  command=lambda: self.load(self.idx - 1)).grid(row=r, column=1, pady=2); r += 1
        tk.Button(self.root, text="下一张 (D)",
                  command=lambda: self.load(self.idx + 1)).grid(row=r, column=1, pady=2); r += 1
        tk.Button(self.root, text="跳到未标注",
                  command=self.goto_unannotated).grid(row=r, column=1, pady=2); r += 1
        tk.Button(self.root, text="下一张需增强 (N)",
                  command=self.goto_next_queue).grid(row=r, column=1, pady=2); r += 1
        self.status = tk.Label(self.root, text="", anchor="w", width=40)
        self.status.grid(row=r, column=1, sticky="w")
        self.root.bind("a", lambda e: self.load(self.idx - 1))
        self.root.bind("d", lambda e: self.load(self.idx + 1))
        self.root.bind("s", lambda e: self.save())
        self.root.bind("n", lambda e: self.goto_next_queue())

    def out_path(self, idx=None):
        i = self.idx if idx is None else idx
        model, path = self.images[i]
        fname = os.path.splitext(os.path.basename(path))[0]
        return os.path.join(OUT_DIR, f"{model}_{fname}.json")

    def first_unannotated(self):
        """第一个未标注（无 JSON）的图片索引，全标了返回 0。"""
        for i in range(len(self.images)):
            if not os.path.exists(self.out_path(i)):
                return i
        return 0

    def goto_unannotated(self):
        self.idx = self.first_unannotated()
        self.load(self.idx)

    def _load_queue(self):
        if os.path.exists(QUEUE_FILE):
            with open(QUEUE_FILE, encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_queue(self):
        with open(QUEUE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.queue, f, ensure_ascii=False, indent=2)

    def load_queue(self, entry):
        """从标注队列加载一张图（用队列的 vehicle_box，跳过重检测）。"""
        path = os.path.join(HERE, entry["path"])
        self.cur = {"image": path, "model": entry["model"],
                    "vehicle_box": entry.get("vehicle_box")}
        self.parts = []
        self.view_var.set(entry.get("view", ""))
        self.model_var.set(entry["model"])
        self.load_existing()
        self.render()

    def goto_next_queue(self):
        if not self.queue:
            self.status.config(text="标注队列为空（先跑 check_taillight_detection.py 生成）")
            return
        self.queue_idx = (self.queue_idx + 1) % len(self.queue)
        self.load_queue(self.queue[self.queue_idx])
        self.status.config(
            text=f"队列 {self.queue_idx+1}/{len(self.queue)}  "
                 f"{os.path.basename(self.cur['image'])}  部件:{len(self.parts)}")

    def load(self, idx):
        if not self.images:
            return
        idx = max(0, min(idx, len(self.images) - 1))
        self.idx = idx
        model, path = self.images[idx]
        self.model_var.set(model)
        try:
            det = detect(path)
        except Exception as e:
            self.status.config(text=f"检测失败: {e}")
            return
        self.cur = {"image": path, "model": model,
                    "vehicle_box": det["vehicle_box"]}
        self.parts = []
        self.view_var.set("")
        self.load_existing()
        self.render()

    def load_existing(self):
        out = self.out_path()
        if os.path.exists(out):
            with open(out, encoding="utf-8") as f:
                d = json.load(f)
            self.parts = d.get("parts", [])
            if d.get("view"):
                self.view_var.set(d["view"])

    def render(self):
        if not self.cur:
            return
        img = imread_unicode(self.cur["image"])
        if img is None:
            self.status.config(text=f"读图失败: {os.path.basename(self.cur['image'])}")
            return
        ih, iw = img.shape[:2]
        s = min(820 / iw, 620 / ih)
        self.scale = s
        img2 = cv2.resize(img, (int(iw * s), int(ih * s)))

        def sb(b):
            return tuple(int(v * s) for v in b)
        if self.cur["vehicle_box"]:
            x1, y1, x2, y2 = sb(self.cur["vehicle_box"])
            cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img2, "vehicle", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for p in self.parts:
            x1, y1, x2, y2 = sb(p["box"])
            c = PART_COLORS.get(p["label"], (255, 0, 0))
            cv2.rectangle(img2, (x1, y1), (x2, y2), c, 2)
            cv2.putText(img2, p["label"], (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.status.config(
            text=f"{self.idx + 1}/{len(self.images)}  "
                 f"{os.path.basename(self.cur['image'])}  部件:{len(self.parts)}")

    def on_down(self, e):
        self.drawing = True
        self.start = (e.x, e.y)
        self.rect_id = self.canvas.create_rectangle(
            e.x, e.y, e.x, e.y, outline="yellow", width=2)

    def on_move(self, e):
        if self.drawing:
            self.canvas.coords(self.rect_id, self.start[0], self.start[1], e.x, e.y)

    def on_up(self, e):
        if not self.drawing:
            return
        self.drawing = False
        x1, y1 = self.start
        x2, y2 = e.x, e.y
        self.canvas.delete(self.rect_id)
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            return
        s = self.scale
        box = [int(min(x1, x2) / s), int(min(y1, y2) / s),
               int(max(x1, x2) / s), int(max(y1, y2) / s)]
        sel = self.part_list.curselection()
        label = PART_LABELS[sel[0]] if sel else PART_LABELS[0]
        self.parts.append({"label": label, "box": box})
        self.render()

    def del_last(self):
        if self.parts:
            self.parts.pop()
            self.render()

    def save(self):
        if not self.cur:
            return
        cur_rel = os.path.relpath(self.cur["image"], HERE).replace("\\", "/")
        d = {
            "image": cur_rel,
            "model": self.cur["model"],
            "view": self.view_var.get(),
            "vehicle_box": self.cur["vehicle_box"],
            "parts": self.parts,
        }
        with open(self.out_path(), "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        # 出队：当前图在标注队列则移除（标注增强完成）
        before = len(self.queue)
        self.queue = [q for q in self.queue if q.get("path") != cur_rel]
        if len(self.queue) < before:
            self._save_queue()
            if self.queue_idx >= len(self.queue):
                self.queue_idx = -1
            self.status.config(
                text=f"已保存并出队，队列剩余 {len(self.queue)}  "
                     f"{os.path.basename(self.out_path())}")
        else:
            self.status.config(text=f"已保存: {os.path.basename(self.out_path())}")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("车辆部件标注工具")
    AnnoTool(root)
    root.mainloop()
