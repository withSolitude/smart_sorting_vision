"""SAM 3 交互标注：正负点、可选框、候选切换和手工多边形。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from sam3_annotation import AnnotationSession
from sam3_dataset import process_masks
from sam3_interface import BASE, Sam3Interface, load_config


class AnnotationWindow:
    """每张图只编码一次；点提示在后台运行，窗口保持可响应。"""

    def __init__(self, root: tk.Tk, config: dict):
        self.root, self.config = root, config
        ui = config["interactive"]
        output = Path(ui["output_dir"])
        self.session = AnnotationSession(
            config["dataset"]["source_dir"], output,
            config["dataset"]["splits"], config["export"],
            limit=config["dataset"]["limit"],
        )
        if not self.session.entries:
            raise ValueError("数据集中没有图片")
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.model = None
        self.index = 0
        self.objects = []
        self.points, self.labels, self.vertices = [], [], []
        self.box = None
        self.box_start = None
        self.candidates = []
        self.candidate_index = 0
        self.mode = "点选"
        self.dirty = False
        self.closed = False
        root.title("SAM 3 交互标注 — 左键目标 / 右键排除")
        width = min(int(ui["view_width"]), root.winfo_screenwidth() - 350)
        height = min(int(ui["view_height"]), root.winfo_screenheight() - 180)
        outer = ttk.Frame(root, padding=8)
        outer.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(outer, width=width, height=height,
                                background="#181818", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        panel = ttk.Frame(outer, padding=(12, 0, 0, 0), width=300)
        panel.pack(side="right", fill="y")
        self.title = tk.StringVar()
        ttk.Label(panel, textvariable=self.title, wraplength=290).pack(anchor="w")
        self.mode_label = tk.StringVar(value="模式：点选")
        ttk.Label(panel, textvariable=self.mode_label).pack(anchor="w", pady=6)
        for label, action in [
            ("点选模式（D）", lambda: self.set_mode("点选")),
            ("框选目标范围（B）", lambda: self.set_mode("框选")),
            ("手工多边形（P）", lambda: self.set_mode("多边形")),
            ("撤销最后一点（Backspace）", self.undo),
            ("清除当前提示（C）", self.clear),
        ]:
            ttk.Button(panel, text=label, command=action).pack(fill="x", pady=2)
        self.candidate_label = tk.StringVar(value="候选：无")
        ttk.Label(panel, textvariable=self.candidate_label).pack(anchor="w", pady=(8, 2))
        row = ttk.Frame(panel)
        row.pack(fill="x")
        for i in range(3):
            ttk.Button(row, text=str(i + 1), width=7,
                       command=lambda j=i: self.choose(j)).pack(side="left", padx=2)
        ttk.Button(panel, text="确认当前实例（Enter）", command=self.accept).pack(fill="x", pady=6)
        ttk.Label(panel, text="已确认实例（选中后可删除）：").pack(anchor="w")
        self.instance_list = tk.Listbox(panel, height=6, exportselection=False)
        self.instance_list.pack(fill="x")
        self.instance_list.bind("<<ListboxSelect>>", lambda _: self.render())
        ttk.Button(panel, text="删除选中实例（Delete）", command=self.delete).pack(fill="x", pady=3)
        for label, action in [
            ("保存本张（S）", self.save),
            ("保存并下一张（N）", self.next_image),
            ("上一张（A）", lambda: self.navigate(-1)),
            ("适应窗口（F）", self.fit),
            ("导出已保存图片的 CVAT 包（E）", self.export),
        ]:
            ttk.Button(panel, text=label, command=action).pack(fill="x", pady=2)
        ttk.Label(panel, text="左键：目标正点；右键：排除负点\n滚轮：缩放；中键拖动：平移\n多边形模式：左键添点、右键撤点\nEnter 确认一个实例，再点下一个物体\n无目标时直接保存，生成空标签\n框选是当前实例范围，不是相似物体示例\n这里不使用 YAML 的文字提示词", wraplength=290).pack(anchor="w", pady=8)
        self.status = tk.StringVar(value="正在加载本地模型，请稍候……")
        ttk.Label(root, textvariable=self.status, padding=8, wraplength=1100).pack(fill="x")
        self.canvas.bind("<Button-1>", self.left_click)
        self.canvas.bind("<Button-3>", self.right_click)
        self.canvas.bind("<B1-Motion>", self.box_motion)
        self.canvas.bind("<ButtonRelease-1>", self.box_release)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Button-4>", lambda event: self.zoom(event, 1))
        self.canvas.bind("<Button-5>", lambda event: self.zoom(event, -1))
        self.canvas.bind("<Button-2>", self.pan_start)
        self.canvas.bind("<B2-Motion>", self.pan)
        self.canvas.bind("<Configure>", lambda _: self.render())
        self.pan_anchor = None
        root.bind("<Key>", self.key)
        root.protocol("WM_DELETE_WINDOW", self.close)
        start = next((i for i in range(len(self.session.entries)) if not self.session.is_saved(i)), 0)
        self.open_image(start, encode=False)
        self.submit(self.load_model, self.model_loaded)

    def load_model(self):
        return Sam3Interface(self.config["model"], enable_interactivity=True)

    def submit(self, work, callback):
        if self.future is not None:
            return
        self.future = self.pool.submit(work)
        def poll():
            if self.closed:
                return
            if not self.future.done():
                self.root.after(50, poll)
                return
            future, self.future = self.future, None
            try:
                callback(future.result())
            except Exception as error:
                self.status.set(f"操作失败：{error}。标签未自动改写，可继续手工多边形标注。")
                messagebox.showerror("SAM 3", str(error), parent=self.root)
        self.root.after(50, poll)

    def model_loaded(self, model):
        self.model = model
        self.encode_image()

    def encode_image(self):
        self.status.set("正在编码当前图片……")
        self.submit(lambda: self.model.set_image(self.image),
                    lambda _: self.status.set("就绪：左键点顶面，右键点不需要的侧面；分不开时按 P 手工描边。"))

    def pending(self):
        return bool(self.points or self.box is not None or self.vertices or self.candidates)

    def open_image(self, index, encode=True):
        self.index = index
        split, path = self.session.entries[index]
        with Image.open(path) as opened:
            self.image = opened.convert("RGB")
        self.objects = self.session.load(index)
        self.dirty = False
        self.reset_prompts()
        self.title.set(f"{index + 1} / {len(self.session.entries)}\n{split} / {path.name}")
        self.refresh_list()
        self.root.update_idletasks()
        self.fit()
        if encode and self.model is not None:
            self.encode_image()

    def reset_prompts(self):
        self.points, self.labels, self.vertices = [], [], []
        self.box, self.box_start = None, None
        self.candidates, self.candidate_index = [], 0
        self.candidate_label.set("候选：无")

    def set_mode(self, mode):
        if self.future is not None:
            return
        if mode == "多边形" or self.mode == "多边形":
            self.reset_prompts()
        self.mode = mode
        self.mode_label.set("模式：" + mode)
        self.render()

    def image_point(self, event):
        x = (event.x - self.offset_x) / self.scale
        y = (event.y - self.offset_y) / self.scale
        if 0 <= x < self.image.width and 0 <= y < self.image.height:
            return [float(x), float(y)]
        return None

    def left_click(self, event):
        if self.future is not None:
            return
        point = self.image_point(event)
        if point is None:
            return
        self.canvas.focus_set()
        if self.mode == "多边形":
            self.vertices.append(point)
            self.render()
        elif self.mode == "框选":
            self.box_start = point
            self.box = [*point, *point]
            self.render()
        else:
            self.points.append(point)
            self.labels.append(1)
            self.predict()

    def right_click(self, event):
        if self.future is not None:
            return
        if self.mode == "多边形":
            self.undo()
            return
        point = self.image_point(event)
        if point is not None:
            self.points.append(point)
            self.labels.append(0)
            self.predict()

    def box_motion(self, event):
        point = self.image_point(event)
        if self.box_start is not None and point is not None:
            self.box = [*self.box_start, *point]
            self.render()

    def box_release(self, event):
        if self.box_start is None:
            return
        x1, y1, x2, y2 = self.box
        self.box_start = None
        self.box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        if self.box[2] - self.box[0] < 2 or self.box[3] - self.box[1] < 2:
            self.box = None
        self.mode = "点选"
        self.mode_label.set("模式：点选")
        self.predict()

    def predict(self):
        self.candidates = []
        self.candidate_label.set("候选：无")
        self.render()
        if self.model is None:
            self.status.set("模型尚未加载；可以按 P 使用手工多边形。")
            return
        if 1 not in self.labels and self.box is None:
            self.status.set("先左键点一个目标正点，或按 B 框住当前目标。")
            return
        points, labels = list(self.points), list(self.labels)
        box = list(self.box) if self.box is not None else None
        self.status.set("正在根据正负点分割当前实例……")
        self.submit(lambda: self.model.predict_points(points, labels, box=box), self.predicted)

    def predicted(self, predictions):
        # 三个结果是同一实例的备选，不作为三个物体同时导出。
        import cv2
        self.candidates = []
        for prediction in predictions:
            objects = process_masks([prediction], self.config["export"])
            if objects:
                obj = objects[0]
                # 以实际导出的轮廓检查正负点，不能只信模型预测的质量分数。
                obj["point_errors"] = sum(
                    (cv2.pointPolygonTest(obj["polygon"], tuple(point), False) >= 0) != bool(label)
                    for point, label in zip(self.points, self.labels)
                )
                self.candidates.append(obj)
        self.candidates.sort(key=lambda obj: (obj["point_errors"], -obj["score"]))
        self.candidate_index = 0
        self.choose(0)
        self.status.set("选择合适候选，Enter 确认；边界不对可继续补点，或按 P 手工描边。"
                        if self.candidates else "没有有效候选，继续补点、重新框选，或按 P 手工描边。")

    def choose(self, index):
        if 0 <= index < len(self.candidates):
            self.candidate_index = index
            obj = self.candidates[index]
            errors = obj.get("point_errors", 0)
            suffix = f"；{errors} 个点未满足" if errors else "；符合点选"
            if obj.get("disconnected_pixels") or obj.get("has_holes"):
                suffix += "；仅保留主轮廓，需检查"
            self.candidate_label.set(f"候选 {index + 1}/{len(self.candidates)}，质量 {obj['score']:.3f}" + suffix)
        self.render()

    def accept(self):
        if self.future is not None:
            return
        if self.mode == "多边形":
            polygon = np.asarray(self.vertices, dtype=np.float32)
            if len(polygon) < 3:
                self.status.set("手工多边形至少需要三个点。")
                return
            import cv2
            if abs(cv2.contourArea(polygon)) < float(self.config["export"]["min_area_px"]):
                self.status.set("多边形面积太小，请检查描边或 YAML 的 min_area_px。")
                return
            obj = {"polygon": polygon, "score": 1.0, "prompt": "manual polygon"}
        elif self.candidates:
            obj = self.candidates[self.candidate_index]
        else:
            self.status.set("还没有可确认的候选。")
            return
        # 只保留标签需要的信息，避免为全部已确认实例持有整张 mask。
        self.objects.append({"polygon": np.asarray(obj["polygon"]).copy(),
                             "score": float(obj["score"]), "prompt": obj.get("prompt", "interactive"),
                             "disconnected_pixels": obj.get("disconnected_pixels", 0),
                             "has_holes": obj.get("has_holes", False)})
        self.dirty = True
        self.reset_prompts()
        self.refresh_list()
        self.status.set("已加入一个实例；继续点下一个物体，或按 S 保存本张。")
        self.render()

    def refresh_list(self):
        self.instance_list.delete(0, tk.END)
        for i, _ in enumerate(self.objects, 1):
            self.instance_list.insert(tk.END, f"{i}. object")

    def delete(self):
        selected = self.instance_list.curselection()
        if selected:
            self.objects.pop(selected[0])
            self.dirty = True
            self.refresh_list()
            self.render()

    def undo(self):
        if self.future is not None:
            return
        if self.mode == "多边形":
            if self.vertices:
                self.vertices.pop()
            self.render()
        elif self.points:
            self.points.pop()
            self.labels.pop()
            self.predict()
        elif self.box is not None:
            self.clear()

    def clear(self):
        if self.future is None:
            self.reset_prompts()
            self.render()

    def save(self):
        if self.pending():
            self.status.set("当前提示尚未确认：先 Enter 加入实例，或 C 清除，再保存。")
            return False
        try:
            self.session.save(self.index, self.objects)
        except (OSError, ValueError) as error:
            messagebox.showerror("保存失败", str(error), parent=self.root)
            return False
        self.dirty = False
        self.status.set(f"已保存本张 {len(self.objects)} 个实例；按 N 保存并下一张，E 导出。")
        return True

    def next_image(self):
        if self.future is None and self.save():
            if self.index + 1 < len(self.session.entries):
                self.open_image(self.index + 1)
            else:
                self.status.set("最后一张已保存；按 E 导出已确认的数据。")

    def navigate(self, delta):
        if self.future is not None:
            return
        index = self.index + delta
        if 0 <= index < len(self.session.entries):
            if self.pending():
                self.status.set("先 Enter 确认当前实例，或 C 清除提示。")
                return
            if self.dirty and not self.save():
                return
            self.open_image(index)

    def export(self):
        if self.future is not None:
            return
        if self.pending():
            self.status.set("先 Enter 确认当前实例，或 C 清除，再导出。")
            return
        if self.dirty and not self.save():
            return
        # 切到下一张后直接导出，不能把尚未检查的新图保存成空背景。
        try:
            result = self.session.export()
        except (OSError, ValueError) as error:
            messagebox.showerror("导出失败", str(error), parent=self.root)
            return
        target = result.get("cvat_zip", result["dataset_dir"])
        self.status.set(f"已导出 {result['images']} 张、{result['instances']} 个实例：{target}")

    def fit(self):
        if not hasattr(self, "image"):
            return
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        self.scale = min(width / self.image.width, height / self.image.height)
        self.offset_x = (width - self.image.width * self.scale) / 2
        self.offset_y = (height - self.image.height * self.scale) / 2
        self.render()

    def zoom(self, event, direction=None):
        if direction is None:
            direction = 1 if event.delta > 0 else -1
        point = self.image_point(event)
        if point is None:
            return
        self.scale = min(8.0, max(0.05, self.scale * (1.2 if direction > 0 else 1 / 1.2)))
        self.offset_x = event.x - point[0] * self.scale
        self.offset_y = event.y - point[1] * self.scale
        self.render()

    def pan_start(self, event):
        self.pan_anchor = (event.x, event.y, self.offset_x, self.offset_y)

    def pan(self, event):
        if self.pan_anchor:
            x, y, ox, oy = self.pan_anchor
            self.offset_x, self.offset_y = ox + event.x - x, oy + event.y - y
            self.render()

    def render(self):
        if not hasattr(self, "scale") or not hasattr(self, "image"):
            return
        overlay = Image.new("RGBA", self.image.size)
        draw = ImageDraw.Draw(overlay)
        selected = self.instance_list.curselection()
        for i, obj in enumerate(self.objects):
            polygon = [tuple(p) for p in np.asarray(obj["polygon"]).tolist()]
            color = (255, 220, 30, 210) if selected and selected[0] == i else (30, 230, 90, 180)
            draw.polygon(polygon, fill=(*color[:3], 40), outline=color, width=2)
        if self.candidates:
            polygon = [tuple(p) for p in self.candidates[self.candidate_index]["polygon"].tolist()]
            draw.polygon(polygon, fill=(30, 150, 255, 90), outline=(40, 210, 255, 255), width=2)
        radius = max(2, 5 / self.scale)
        for (x, y), label in zip(self.points, self.labels):
            color = (30, 255, 60, 255) if label else (255, 40, 40, 255)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white")
        if self.box is not None:
            x1, y1, x2, y2 = self.box
            draw.rectangle((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), outline="yellow", width=2)
        if self.vertices:
            vertices = [tuple(p) for p in self.vertices]
            if len(vertices) >= 2:
                draw.line(vertices, fill=(255, 80, 230, 255), width=2)
            for x, y in vertices:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 80, 230, 255))
        image = Image.alpha_composite(self.image.convert("RGBA"), overlay).convert("RGB")
        size = (max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height()))
        viewport = image.transform(size, Image.Transform.AFFINE,
                                   (1 / self.scale, 0, -self.offset_x / self.scale,
                                    0, 1 / self.scale, -self.offset_y / self.scale),
                                   resample=Image.Resampling.BILINEAR, fillcolor="#181818")
        self.photo = ImageTk.PhotoImage(viewport)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

    def key(self, event):
        actions = {"Return": self.accept, "BackSpace": self.undo, "Delete": self.delete,
                   "s": self.save, "n": self.next_image, "a": lambda: self.navigate(-1),
                   "c": self.clear, "f": self.fit, "e": self.export,
                   "d": lambda: self.set_mode("点选"), "b": lambda: self.set_mode("框选"),
                   "p": lambda: self.set_mode("多边形")}
        key = event.keysym if event.keysym in actions else event.keysym.lower()
        if key in actions:
            actions[key]()
            return "break"
        if key in {"1", "2", "3"}:
            self.choose(int(key) - 1)
            return "break"

    def close(self):
        if self.dirty or self.pending():
            if not messagebox.askyesno("退出", "本张有未保存内容，仍要退出吗？", parent=self.root):
                return
        self.closed = True
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="SAM 3 正负点交互标注")
    parser.add_argument("--config", type=Path, default=BASE / "sam3_config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(config["interactive"]["output_dir"])
    if not output.is_absolute():
        output = args.config.resolve().parent / output
    config["interactive"]["output_dir"] = output.resolve()
    root = tk.Tk()
    try:
        AnnotationWindow(root, config)
    except Exception as error:
        messagebox.showerror("启动失败", str(error), parent=root)
        root.destroy()
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
