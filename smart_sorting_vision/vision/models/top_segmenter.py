from pathlib import Path
from typing import Union
import time

import numpy as np
from ultralytics import YOLO

class TopSegmenter:
    """
    顶部相机实例分割模型
    """
    
    def __init__(
        self,
        model_path: Union[str,Path],
        conf: float = 0.25,
        imgsz: int = 640,
        device: str = "cpu",):


        self.model_path = Path(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        
        # 检查模型文件是否存在
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"找不到模型文件:{self.model_path}"
            )
        print(
            f"[TopSegmenter]正在加载模型"
            f"{self.model_path}"
        )
        
        # 加载yolo模型
        self.model = YOLO(str(self.model_path))
        
        print("[TopSegmenter]模型加载成功")

    def warmup(self, image: np.ndarray, runs: int = 3) -> dict:
        """
        在正式 START 之前把 PyTorch/TensorRT 的惰性初始化全部完成。

        使用真实顶部相机帧，并沿用正式 predict() 的 conf/imgsz/device，
        因此固定尺寸 TensorRT Engine 的 shape 不匹配会在 WAIT START 前暴露。
        warmup 不修改 TopPickConsensus，也不会产生任何机械动作。
        """
        if image is None:
            raise ValueError("TopSegmenter warmup 输入图像为空")
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("TopSegmenter warmup 输入图像无效")

        runs = max(1, int(runs))
        timings_ms = []

        for index in range(runs):
            start = time.perf_counter()
            result = self.predict(image)

            # 强制走一遍 Boxes/Mask -> CPU/NumPy 路径，比赛开始后不再
            # 第一次触发相关初始化，同时也能提前发现 NumPy ABI 类问题。
            self.get_objects(result)

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timings_ms.append(elapsed_ms)
            print(
                f"[WARMUP][TOP] {index + 1}/{runs} "
                f"{elapsed_ms:.1f} ms"
            )

        steady = timings_ms[1:] if len(timings_ms) > 1 else timings_ms
        return {
            "runs": runs,
            "first_ms": float(timings_ms[0]),
            "steady_avg_ms": float(sum(steady) / len(steady)),
            "max_ms": float(max(timings_ms)),
        }
        
    def predict(self,image: np.ndarray):
        """
        对单张图像进行实例分割

        Args:
            image (np.ndarray): _description_ BGR图像
            
        return:
            yolo单张图片推理结果
        """
        
        if image is None:
            raise ValueError('输入图像为空')
        
        results = self.model.predict(
            source=image,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        
        return results[0]
    
    def get_objects(self,result):
        """
        从yolo结果中提取一个实例的信息

        return:
                类别编号
                类别名称
                置信度
                bbox
                polygon
        """
        
        objects = []
        
        # 没有检测到任何目标
        if result.boxes is None:
            return objects
        
        if len(result.boxes) == 0:
            return objects
        
        boxes = result.boxes
        
        # 
        masks = result.masks
        
        for i in range(len(boxes)):
            
            # 类别
            class_id = int(boxes.cls[i].item())
            
            class_name = result.names[class_id]
            
            confindence = float(boxes.conf[i].item())
            
            x1,y1,x2,y2 = (boxes.xyxy[i].cpu().numpy())
            
            bbox = (float(x1),float(y1),float(x2),float(y2))
            
            # 实例分割轮廓
            polygon = None
            
            if(masks is not None and i < len(masks.xy)):
                polygon = masks.xy[i]
                
            obj = {
                "index": i,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confindence,
                "bbox": bbox,
                "polygon": polygon,
            }
            
            objects.append(obj)
            
        return objects