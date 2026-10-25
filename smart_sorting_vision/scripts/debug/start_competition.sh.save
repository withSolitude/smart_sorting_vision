#!/bin/bash

PROJECT=/home/nvidia/smart_sorting_vision
LOGDIR=$PROJECT/outputs/logs

mkdir -p "$LOGDIR"

echo "====================================" >> "$LOGDIR/autostart.log"
echo "START $(date)" >> "$LOGDIR/autostart.log"

# 等系统和 USB 设备起来
for i in $(seq 1 30); do
    if [ -e /dev/ttyCH341USB0 ] \
       && [ -e /dev/v4l/by-id/usb-icSpring_icspring_camera_20240307110322-video-index0 ] \
       && [ -e /dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0 ]; then
        echo "Devices ready after ${i}s" >> "$LOGDIR/autostart.log"
        break
    fi

    echo "Waiting devices... ${i}/30" >> "$LOGDIR/autostart.log"
    sleep 1
done

cd "$PROJECT" || exit 1

source "$PROJECT/.venv/bin/activate"

# 防止 OpenCV 再次抢 Qt 插件路径
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/qt5/plugins/platforms
export QT_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/qt5/plugins
unset QT_QPA_FONTDIR

echo "Launching Smart Sorting..." >> "$LOGDIR/autostart.log"

exec python "$PROJECT/main.py" >> "$LOGDIR/autostart.log" 2>&1
