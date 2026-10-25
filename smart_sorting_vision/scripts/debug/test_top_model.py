from pathlib import Path
import cv2

from vision.models.top_segmenter import(TopSegmenter,)

# 工程根目录
ROOT = Path(__file__).resolve().parents[1]

# 文件路径
MODEL_PATH = (
    ROOT
    / "weights"
    / "pretrained"
    / "yolo26n-seg.pt"
)

IMAGE_PATH = (
    ROOT
    / "outputs"
    / "debug"
    / "bus.jpg"
)

def main():
    
    print("=============================")
    print("顶部实例分割模型测试")
    print("============================")
    
    # 创建模型
    segmenter = TopSegmenter(
        model_path=MODEL_PATH,
        conf=0.25,
        imgsz=640,
        device='cpu',
    )
    
    # 读取测试图片
    image = cv2.imread(str(IMAGE_PATH))
    
    if image is None:
        raise RuntimeError(f"无法读取图片:{IMAGE_PATH}")

    print(f"测试图片:{IMAGE_PATH}")
    print(
        f"图片尺寸:"
        f"{image.shape[1]}"
        f"x"
        f"{image.shape[0]}"
        )
    
    # yolo推理
    result = segmenter.predict(image)
    
    # 提取实例
    objects = (segmenter.get_objects(result))
    
    print()
    print(
        f"检测到目标数量:"
        f"{len(objects)}"
    )
    
    print("========================")
    
    # 输出每个实例的信息
    for obj in objects:
        
        print(
            f"目标{obj['index']}"
        )
        print(
            f"类别:"
            f"{obj['class_name']}"
        )
        
        print(
            f"类别ID:"
            f"{obj['class_id']}"
        )
        
        print(
            f"置信度:"
            f"{obj['confidence']:.3f}"
        )
        
        print(
            f"bbox"
            f"{obj['bbox']}"
        )
        
        polygon = obj['polygon']
        
        if polygon is not None:
            
            print(
                f"mask轮廓点数:"
                f"{len(polygon)}"
            )
        else:
            print('mask:无')
            
        print('=======================')
        
        # 绘制结果
        
        display_image = (result.plot())
        cv2.imshow('Top Yolo26 Segmentation', display_image,)
        
        cv2.waitKey(0)
        cv2.destroyAllwindows()
        
if __name__=="__main__":
    main()