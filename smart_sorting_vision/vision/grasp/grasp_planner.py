import cv2
import numpy as np


class GraspPlanner:
    """
    根据实例分割 Mask 计算安全抓取点。
    """

    def __init__(self):
        pass

    def polygon_to_mask(
        self,
        polygon: np.ndarray,
        image_shape,
    ):
        """
        将 YOLO polygon 转换成二值 Mask。

        Args:
            polygon:
                YOLO result.masks.xy 中的轮廓点。

            image_shape:
                原图 image.shape。

        Returns:
            mask:
                二值图像。
                物体内部 = 255
                背景 = 0
        """

        if polygon is None:
            raise ValueError("polygon为空")

        polygon = np.asarray(
            polygon,
            dtype=np.float32,
        )

        if len(polygon) < 3:
            raise ValueError(
                "polygon轮廓点不足"
            )

        image_height = image_shape[0]
        image_width = image_shape[1]

        # 创建全黑Mask
        mask = np.zeros(
            (
                image_height,
                image_width,
            ),
            dtype=np.uint8,
        )

        # 转成OpenCV需要的整数坐标
        points = np.round(
            polygon
        ).astype(
            np.int32
        )

        # 防止坐标越出图像范围
        points[:, 0] = np.clip(
            points[:, 0],
            0,
            image_width - 1,
        )

        points[:, 1] = np.clip(
            points[:, 1],
            0,
            image_height - 1,
        )

        # 将polygon内部填成白色
        cv2.fillPoly(
            mask,
            [points],
            255,
        )

        return mask

    def find_safe_grasp_point(
        self,
        polygon: np.ndarray,
        image_shape,
    ):
        """
        使用距离变换寻找安全抓取点。

        Returns:
            dict:
                grasp_point:
                    安全抓取点 (u, v)

                clearance:
                    抓取点距离Mask边缘的像素距离

                mask:
                    二值Mask

                distance_map:
                    距离变换结果
        """

        # =====================================
        # 1. polygon → Mask
        # =====================================

        mask = self.polygon_to_mask(
            polygon=polygon,
            image_shape=image_shape,
        )

        # =====================================
        # 2. 给Mask外面增加一圈黑色边界
        # =====================================

        # 这样即使物体贴到图像边缘，
        # Distance Transform仍然能够正确
        # 把图像边缘当作物体边界。

        padded_mask = cv2.copyMakeBorder(
            mask,
            1,
            1,
            1,
            1,
            cv2.BORDER_CONSTANT,
            value=0,
        )

        # =====================================
        # 3. 距离变换
        # =====================================

        distance_map = cv2.distanceTransform(
            padded_mask,
            cv2.DIST_L2,
            5,
        )

        # 去掉刚才增加的一圈
        distance_map = distance_map[
            1:-1,
            1:-1,
        ]

        # =====================================
        # 4. 找距离最大的点
        # =====================================

        min_value, max_value, min_location, max_location = (
            cv2.minMaxLoc(
                distance_map
            )
        )

        if max_value <= 0:
            raise RuntimeError(
                "无法计算有效抓取点"
            )

        grasp_x = int(
            max_location[0]
        )

        grasp_y = int(
            max_location[1]
        )

        grasp_point = (
            grasp_x,
            grasp_y,
        )

        # =====================================
        # 5. 返回
        # =====================================

        return {
            "grasp_point": grasp_point,

            "clearance": float(
                max_value
            ),

            "mask": mask,

            "distance_map": distance_map,
        }