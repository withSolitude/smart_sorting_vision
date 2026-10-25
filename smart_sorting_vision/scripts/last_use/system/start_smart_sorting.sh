#!/bin/bash

PROJECT="/home/nvidia/smart_sorting_vision"
PYTHON="$PROJECT/.venv/bin/python"
LOG_DIR="$PROJECT/outputs/logs"
LOG_FILE="$LOG_DIR/autostart.log"

STM32="/dev/ttyCH341USB0"
TOP_CAMERA="/dev/v4l/by-id/usb-icSpring_icspring_camera_20240307110322-video-index0"
SIDE_CAMERA="/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0"

mkdir -p "$LOG_DIR"

echo "" >> "$LOG_FILE"
echo "============================================================" >> "$LOG_FILE"
echo "[AUTOSTART] $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "============================================================" >> "$LOG_FILE"

cd "$PROJECT" || {
    echo "[ERROR] Cannot cd to $PROJECT" >> "$LOG_FILE"
    exit 1
}

# 防止重复启动两个 main.py
exec 9>/tmp/smart_sorting.lock

if ! flock -n 9; then
    echo "[INFO] smart_sorting is already running, exit." >> "$LOG_FILE"
    exit 0
fi

# 等待桌面、USB、CUDA等初始化
echo "[WAIT] Initial system settle..." >> "$LOG_FILE"
sleep 2


wait_device()
{
    local DEV="$1"
    local NAME="$2"

    echo "[WAIT] $NAME : $DEV" >> "$LOG_FILE"

    # 设备枚举晚于桌面/USB初始化很常见。这里持续等待，不把暂时离线
    # 当成程序错误；设备出现后再启动主程序，由主程序负责运行中重连。
    local i=0
    while true; do
        i=$((i + 1))
        if [ -e "$DEV" ]; then
            echo "[OK] $NAME ready after ${i}s" >> "$LOG_FILE"
            return 0
        fi

        if [ $((i % 5)) -eq 0 ]; then
            echo "[WAIT] $NAME still offline after ${i}s" >> "$LOG_FILE"
        fi
        sleep 1
    done
}


# STM32
wait_device "$STM32" "STM32" || exit 1

# 顶部相机
wait_device "$TOP_CAMERA" "TOP CAMERA" || exit 1

# 侧面相机
wait_device "$SIDE_CAMERA" "SIDE CAMERA" || exit 1


if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] Python not found: $PYTHON" >> "$LOG_FILE"
    exit 1
fi


echo "[START] $PYTHON $PROJECT/main.py" >> "$LOG_FILE"
echo "[TIME] $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"

# 用 venv 中的 Python 直接运行正式程序
exec "$PYTHON" "$PROJECT/main.py" >> "$LOG_FILE" 2>&1
