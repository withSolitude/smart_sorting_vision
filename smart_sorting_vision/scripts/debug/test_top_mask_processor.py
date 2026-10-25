from pathlib import Path
import cv2
from vision.models.top_segmenter import(TopSegmenter,)
from vision.grasp.mask_processor import(MaskProcessor,)

# 工程根目录
ROOT = Path(__file__).resolve().parents[1]

# 路径
MODEL_PATH = (
    ROOT
    /"weights"
    /"pretrained"
    /"yolo26n-seg.pt"
)

IMAGE_PATH = (
    ROOT
    /"outputs"
    /"debug"
    /"bus.jpg"
)

def main():
    print("======================")
    print("Mask几何处理测试")
    print("======================")
    
    # 加载模型
    segmenter = TopSegmenter(model_path=MODEL_PATH,
                             conf=0.25,
                             imgsz=640,
                             device='cpu',
                             )
    
    # 创建mask处理器
    processor = MaskProcessor(edge_margin=10)
    
    # 读取图片
    image = cv2.imread(str(IMAGE_PATH))
    
    if image is None:
        raise RuntimeError(f"无法读取图片:{IMAGE_PATH}")
    
    # YOLO推理
    result = segmenter.predict(image)
    
    objects = segmenter.get_objects(result)
    
    print(
        f"yolo检测到:"
        f"{len(objects)}个目标"
    )
    
    # 绘图
    display = image.copy()
    
    # 分析每一个mask
    for obj in objects:
        polygon = obj["polygon"]
        
        # if polygon is None:
        #     continue
        print(
            f"目标 {obj['index']}:"
            f"polygon是否存在 = {obj['polygon'] is not None}"
        )

        polygon = obj["polygon"]

        if polygon is None:
            print(
                f"目标 {obj['index']} 没有有效Mask，跳过几何分析"
            )
            continue
        
        geometry = processor.analyze(
            polygon=polygon,
            image_shape=image.shape,
        )
        
        print()
        print(f"目标{obj['index']}")
        print(
            f"类别:"
            f"{obj['class_name']}"
        )
        print(
            f"置信度:"
            f"{obj['confidence']}"
        )
        print(
            f"面积："
            f"{geometry['area']:.1f} px²"
        )

        center_x, center_y = (
            geometry["center"]
        )

        print(
            f"中心："
            f"({center_x:.1f}, "
            f"{center_y:.1f})"
        )

        print(
            f"方向角："
            f"{geometry['angle']:.1f}°"
        )

        print(
            f"尺寸："
            f"{geometry['width']:.1f}"
            f" × "
            f"{geometry['height']:.1f}"
        )

        print(
            f"靠近边缘："
            f"{geometry['touches_edge']}"
        )
        
        # 绘制mask轮廓
        contour = polygon.astype("int32").reshape(-1,1,2,)
        
        cv2.drawContours(
            display,
            [contour],
            -1,
            (255,255,255),
            2,
        )
        
        # 绘制中心点
        cx = int(center_x)
        cy = int(center_y)
        cv2.circle(
            display,
            (cx,cy),
            6,
            (255,255,2550),
            -1,
        )
        
        # 绘制旋转矩形
        rotated_box = (
            geometry['rotated_box'].astype('int32')
        )
        
        cv2.drawContours(
            display,
            [rotated_box],
            0,
            (255,255,255),
            2,
        )
        
        # 显示编号和方向
        text = (
            f"ID={obj['index']}"
            f"A={geometry['angle']:.1f}"
        )
        
        cv2.putText(
            display,
            text,
            (cx + 10,cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,255,255),
            1,
            cv2.LINE_AA,
        )
        
        # 显示
    cv2.imshow(
        "Mask Processor Test",
        display,
    )
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()