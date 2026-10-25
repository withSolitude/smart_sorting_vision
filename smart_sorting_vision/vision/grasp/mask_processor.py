import math
import cv2
import numpy as np

class MaskProcessor:
    """
    对yolo实例分割得到的单个物体轮廓进行几何分析
    """
    
    def __init__(self,edge_margin: int = 10):
        """_summary_

        Args:
            edge_margin (int, optional): _description_. Defaults to 10.
        判断物体是否靠近图像边缘使用的安全距离
        """
        
        self.edge_margin = edge_margin
        
    def analyze(self,polygon: np.ndarray,image_shape,):
        """
        分析一个物体的mask轮廓

        Args:
            polygon (np.ndarray): yolo.result.masks.xy中的一个轮廓
            image_shape (_type_): 原始图像shape
            
        return:
        
        dict : 物体的几何信息
        """
        
        if polygon is None:
            raise ValueError("polygon为空")
        
        polygon = np.asarray(polygon,dtype=np.float32)
        
        if len(polygon) < 3:
            raise ValueError("轮廓点数量不足,无法形成封闭区域")
        
        # opencv轮廓常用形状
        # (N,1,2)
        contour = polygon.reshape(-1,1,2)
        
        # 计算面积
        area = float(cv2.contourArea(contour))
        
        # 计算质心
        moments = cv2.moments(contour)
        
        if abs(moments["m00"]) > 1e-6:
            
            center_x = (moments["m10"] / moments["m00"])
            
            center_y = (moments["m01"] / moments["m00"])
            
        else:
            
            # 极端情况下使用所有轮廓点平均值
            
            center_x = float(polygon[:,0].mean())
            
            center_y = float(polygon[:,1].mean())
            
        center = (float(center_x),float(center_y),)
        
        # 最小外接旋转矩阵
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.asarray(box,dtype=np.float32)
        
        # 根据矩形最长边计算方向
        longest_length = 0.0
        longest_vector = None
        
        for i in range(4):
            p1 = box[i]
            p2 = box[(i+1)%4]
            vector = p2-p1
            length = float(np.linalg.norm(vector))
            
            if length > longest_length:
                longest_length = length
                longest_vector = vector
                
        if longest_vector is None:
            angle = 0.0
        else:
            angle = math.degrees(math.atan2(float(longest_vector[1]),float(longest_vector[0]),))
        
        # 角度归一到【-90,90】
        while angle >= 90.0:
            angle -= 180.0
            
        while angle < -90.0:
            angle += 180.0
            
        # 矩形长宽
        side_1 = float(np.linalg.norm(box[1]-box[0]))
        side_2 = float(np.linalg.norm(box[2]-box[1]))
        
        width = max(side_1,side_2,)
        height = min(side_1,side_2)
        
        # 普通外接矩形
        x,y,w,h = cv2.boundingRect(contour)
        
        # 判断是否靠近图像边缘
        image_height = image_shape[0]
        image_width = image_shape[1]
        
        touches_edge = (
            x <= self.edge_margin
            or y <= self.edge_margin
            or (x + w) >= (image_width - self.edge_margin)
            or (y + h) >= (image_height -self.edge_margin)
            )
        
        # 返回结果
        
        return {
            "area": area,
            "center": center,
            "angle":angle,
            "width": width,
            "height": height,
            "rotated_box": box,
            "bbox": (int(x),int(y),int(w),int(h),),
            "touches_edge": touches_edge
        }
        