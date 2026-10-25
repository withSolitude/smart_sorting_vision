from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
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
    """
    一件已经完成分拣的货物记录。
    """

    sequence: int
    product_name: str
    image_bgr: Optional[np.ndarray]
    success_count: int


class SortingDisplayWindow(QMainWindow):
    """
    智能分拣比赛显示屏。

    按竞赛规则显示：

        序号
        货物名称
        图片
        分拣成功总数量

    只有物体真正完成分拣后，
    才通过 add_finished_item() 新增记录。
    """

    add_record_signal = pyqtSignal(object)
    reset_signal = pyqtSignal()

    def __init__(
        self,
        fullscreen: bool = True,
        max_visible_rows: int = 6,
    ):
        super().__init__()

        self.fullscreen = fullscreen
        self.max_visible_rows = max_visible_rows

        self.records: list[SortingDisplayRecord] = []

        self.setWindowTitle(
            "智能分拣系统"
        )

        self.setMinimumSize(
            800,
            480,
        )

        self.add_record_signal.connect(
            self._add_record
        )

        self.reset_signal.connect(
            self._reset_table
        )

        self._build_ui()
        self._apply_style()

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
            0,
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
        sequence: int,
        product_name: str,
        image_bgr: Optional[np.ndarray],
        success_count: int,
    ):
        """
        一件货物真正完成分拣后调用。

        可以从其他线程调用。
        """

        if image_bgr is not None:
            image_bgr = image_bgr.copy()

        record = SortingDisplayRecord(
            sequence=sequence,
            product_name=product_name,
            image_bgr=image_bgr,
            success_count=success_count,
        )

        self.add_record_signal.emit(
            record
        )

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
    # 内部添加记录
    # ============================================================

    def _add_record(
        self,
        record: SortingDisplayRecord,
    ):
        self.records.append(
            record
        )

        row = self.table.rowCount()

        self.table.insertRow(
            row
        )

        # --------------------------------------------------------
        # 序号
        # --------------------------------------------------------

        sequence_item = QTableWidgetItem(
            str(
                record.sequence
            )
        )

        sequence_item.setTextAlignment(
            Qt.AlignCenter
        )

        self.table.setItem(
            row,
            0,
            sequence_item,
        )

        # --------------------------------------------------------
        # 货物名称
        # --------------------------------------------------------

        name_item = QTableWidgetItem(
            record.product_name
        )

        name_item.setTextAlignment(
            Qt.AlignCenter
        )

        self.table.setItem(
            row,
            1,
            name_item,
        )

        # --------------------------------------------------------
        # 图片
        # --------------------------------------------------------

        image_label = QLabel()

        image_label.setAlignment(
            Qt.AlignCenter
        )

        image_label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        if record.image_bgr is not None:

            pixmap = self._bgr_to_pixmap(
                record.image_bgr
            )

            if pixmap is not None:

                pixmap = pixmap.scaled(
                    180,
                    120,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )

                image_label.setPixmap(
                    pixmap
                )

        self.table.setCellWidget(
            row,
            2,
            image_label,
        )

        # --------------------------------------------------------
        # 成功数量
        # --------------------------------------------------------

        count_item = QTableWidgetItem(
            str(
                record.success_count
            )
        )

        count_item.setTextAlignment(
            Qt.AlignCenter
        )

        self.table.setItem(
            row,
            3,
            count_item,
        )

        # --------------------------------------------------------
        # 行高度
        # --------------------------------------------------------

        self.table.setRowHeight(
            row,
            135,
        )

        # --------------------------------------------------------
        # 自动滚动到最新结果
        # --------------------------------------------------------

        self.table.scrollToBottom()

        self.status_label.setText(
            f"第 {record.sequence} 件货物分拣完成"
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
        self.records.clear()

        self.table.setRowCount(
            0
        )

        self.status_label.setText(
            "等待分拣结果"
        )

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
