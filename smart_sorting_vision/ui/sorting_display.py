from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


@dataclass
class SortingDisplayRecord:
    """入盒完成后，用于更新对应储物盒一行的记录。"""

    box_id: int
    product_name: str
    image_bgr: Optional[np.ndarray]
    box_count: int


class BoxThumbnail(QLabel):
    """按当前单元格大小等比缩放照片。"""

    def __init__(self, pixmap: QPixmap):
        super().__init__()
        self._photo = pixmap
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.setPixmap(self._photo.scaled(
            max(1, self.width() - 8),
            max(1, self.height() - 8),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        ))


class SortingDisplayWindow(QMainWindow):
    """智能分拣比赛显示屏。每个储物盒对应一行，数量以入盒确认为准。"""

    add_record_signal = pyqtSignal(object)
    reset_signal = pyqtSignal()

    def __init__(
        self,
        fullscreen: bool = True,
        box_count: int = 6,
    ):
        super().__init__()

        self.fullscreen = fullscreen
        self.box_count = box_count

        self.setWindowTitle(
            "智能分拣系统"
        )

        self.setMinimumSize(
            800,
            480,
        )

        self.add_record_signal.connect(
            self._update_box
        )

        self.reset_signal.connect(
            self._reset_table
        )

        self._build_ui()
        self._apply_style()
        self._reset_table()

        if fullscreen:
            self.showFullScreen()
        else:
            self.resize(
                1024,
                600,
            )
            self.show()

    # ============================================================
    # UI
    # ============================================================

    def _build_ui(self):
        central = QWidget()

        self.setCentralWidget(
            central
        )

        layout = QVBoxLayout(
            central
        )

        layout.setContentsMargins(
            20,
            16,
            20,
            16,
        )

        layout.setSpacing(
            12
        )

        # --------------------------------------------------------
        # 标题
        # --------------------------------------------------------

        self.title_label = QLabel(
            "智能分拣系统"
        )

        self.title_label.setAlignment(
            Qt.AlignCenter
        )

        self.title_label.setObjectName(
            "Title"
        )

        layout.addWidget(
            self.title_label
        )

        # --------------------------------------------------------
        # 表格
        # --------------------------------------------------------

        self.table = QTableWidget(
            self.box_count,
            4,
        )

        self.table.setHorizontalHeaderLabels(
            [
                "序号",
                "货物名称",
                "图片",
                "分拣成功总数量",
            ]
        )

        self.table.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )

        self.table.setSelectionMode(
            QAbstractItemView.NoSelection
        )

        self.table.setFocusPolicy(
            Qt.NoFocus
        )

        self.table.setShowGrid(
            True
        )

        self.table.verticalHeader().setVisible(
            False
        )
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

        header = self.table.horizontalHeader()

        header.setSectionResizeMode(
            0,
            QHeaderView.ResizeToContents,
        )

        header.setSectionResizeMode(
            1,
            QHeaderView.Stretch,
        )

        header.setSectionResizeMode(
            2,
            QHeaderView.Stretch,
        )

        header.setSectionResizeMode(
            3,
            QHeaderView.Stretch,
        )

        self.table.setColumnWidth(
            0,
            100,
        )

        layout.addWidget(
            self.table,
            1,
        )

        # --------------------------------------------------------
        # 底部提示
        # --------------------------------------------------------

        self.status_label = QLabel(
            "等待分拣结果"
        )

        self.status_label.setAlignment(
            Qt.AlignCenter
        )

        self.status_label.setObjectName(
            "Status"
        )

        layout.addWidget(
            self.status_label
        )

    # ============================================================
    # 样式
    # ============================================================

    def _apply_style(self):
        self.setFont(
            QFont(
                "Noto Sans CJK SC",
                14,
            )
        )

        self.setStyleSheet(
            """
            QMainWindow {
                background-color: white;
            }

            #Title {
                color: #111111;
                font-size: 30px;
                font-weight: bold;
                padding: 8px;
            }

            QTableWidget {
                background-color: white;
                color: black;

                gridline-color: #222222;

                border: 2px solid #222222;

                font-size: 20px;
            }

            QHeaderView::section {
                background-color: #f1f1f1;
                color: black;

                border: 1px solid #222222;

                padding: 8px;

                font-size: 20px;
                font-weight: bold;
            }

            QTableWidget::item {
                border: 1px solid #333333;

                padding: 5px;

                color: black;
            }

            #Status {
                color: #333333;

                font-size: 17px;

                padding: 4px;
            }
            """
        )

    # ============================================================
    # 对外接口
    # ============================================================

    def add_finished_item(
        self,
        *,
        box_id: int,
        product_name: str,
        image_bgr: Optional[np.ndarray],
        box_count: int,
    ):
        """入盒确认后提交对应盒号和盒内数量。可从运行线程调用。"""
        self.add_record_signal.emit(SortingDisplayRecord(
            box_id=box_id,
            product_name=product_name,
            image_bgr=image_bgr.copy() if image_bgr is not None else None,
            box_count=box_count,
        ))

    def reset(self):
        """
        新一轮比赛开始时清空。
        """

        self.reset_signal.emit()

    def set_waiting(
        self,
        text: str = "等待分拣结果",
    ):
        self.status_label.setText(
            text
        )

    # ============================================================
    # 更新储物盒
    # ============================================================

    def _update_box(self, record: SortingDisplayRecord):
        row = record.box_id - 1
        if not 0 <= row < self.table.rowCount():
            return

        count_item = self.table.item(row, 3)
        previous_count = int(count_item.text())
        if record.box_count <= previous_count:
            return

        if previous_count == 0:
            self.table.item(row, 1).setText(record.product_name)

        # 偶尔没有侧面图，等该盒下次成功入盒且有图时再补首张照片。
        if self.table.cellWidget(row, 2) is None:
            pixmap = self._bgr_to_pixmap(record.image_bgr)
            if pixmap is not None and not pixmap.isNull():
                self.table.setCellWidget(row, 2, BoxThumbnail(pixmap))

        count_item.setText(str(record.box_count))
        self.status_label.setText(
            f"{record.box_id} 号储物盒分拣完成，盒内共 {record.box_count} 件"
        )

    # ============================================================
    # BGR ndarray -> QPixmap
    # ============================================================

    @staticmethod
    def _bgr_to_pixmap(
        image_bgr: np.ndarray,
    ) -> Optional[QPixmap]:

        if image_bgr is None:
            return None

        if not isinstance(
            image_bgr,
            np.ndarray,
        ):
            return None

        if image_bgr.ndim != 3:
            return None

        if image_bgr.shape[2] != 3:
            return None

        # BGR -> RGB
        rgb = (
            image_bgr[
                :,
                :,
                ::-1
            ]
            .copy()
        )

        height, width, channel = (
            rgb.shape
        )

        bytes_per_line = (
            channel
            *
            width
        )

        qimage = QImage(
            rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        )

        qimage = qimage.copy()

        return QPixmap.fromImage(
            qimage
        )

    # ============================================================
    # Reset
    # ============================================================

    def _reset_table(self):
        for row in range(self.box_count):
            self.table.removeCellWidget(row, 2)
            for column, value in ((0, str(row + 1)), (1, "—"), (3, "0")):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, column, item)
        self.status_label.setText("等待分拣结果")

    # ============================================================
    # 键盘
    # ============================================================

    def keyPressEvent(
        self,
        event,
    ):
        if event.key() == Qt.Key_Escape:
            self.close()
            return

        if event.key() == Qt.Key_F11:

            if self.isFullScreen():
                self.showNormal()

            else:
                self.showFullScreen()

            return

        super().keyPressEvent(
            event
        )
