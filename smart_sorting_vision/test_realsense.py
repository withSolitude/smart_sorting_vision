import pyrealsense2 as rs
import numpy as np
import cv2

# 1. 创建相机数据管道
pipeline = rs.pipeline()

# 2. 配置相机
config = rs.config()

# 开启深度图
config.enable_stream(
    rs.stream.depth,
    640, 480,
    rs.format.z16,
    30
)

# 开启彩色图
config.enable_stream(
    rs.stream.color,
    640, 480,
    rs.format.bgr8,
    30
)

# 3. 启动相机
pipeline.start(config)

try:
    while True:
        # 4. 获取相机数据
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # 5. 转换为 NumPy 数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 6. 获取中心点深度（单位：米）
        distance = depth_frame.get_distance(320, 240)

        print(f"中心点距离: {distance:.3f} m")

        # 7. 深度图伪彩色显示
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # 8. 显示图像
        cv2.imshow("D455 Color", color_image)
        cv2.imshow("D455 Depth", depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()