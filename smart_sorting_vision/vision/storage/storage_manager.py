"""StorageManager

六箱动态分配。

正式运行使用两阶段事务：
    reserve -> commit / cancel
只有收到与当前周期匹配的 MCU_SORT_DONE 后才真正增加库存数量。

allocate() 保留为兼容接口，内部执行 reserve + commit；正式 Runtime 不使用它。
"""

from dataclasses import dataclass

from .object_info import ObjectInfo
from .storage_box import StorageBox


@dataclass(frozen=True)
class StorageResult:
    success: bool
    box_id: int
    is_new_box: bool
    count: int
    reason: str


@dataclass(frozen=True)
class StorageReservation:
    success: bool
    reservation_id: int | None
    box_id: int
    is_new_box: bool
    projected_count: int
    reason: str
    obj: ObjectInfo | None = None


class StorageManager:
    def __init__(self, box_count: int = 6):
        self.boxes = [StorageBox(i + 1) for i in range(box_count)]
        self._reservations: dict[int, StorageReservation] = {}
        self._next_reservation_id = 1

    def _new_reservation(
        self,
        *,
        box: StorageBox,
        obj: ObjectInfo,
        is_new_box: bool,
        reason: str,
    ) -> StorageReservation:
        reservation_id = self._next_reservation_id
        self._next_reservation_id += 1

        reservation = StorageReservation(
            success=True,
            reservation_id=reservation_id,
            box_id=box.box_id,
            is_new_box=is_new_box,
            projected_count=box.count + 1,
            reason=reason,
            obj=obj,
        )
        self._reservations[reservation_id] = reservation
        return reservation

    def reserve(self, obj: ObjectInfo) -> StorageReservation:
        """为当前物体预留目标盒，但不修改真实库存。"""
        # 1) 已有同类箱优先。
        for box in self.boxes:
            if box.match(obj):
                return self._new_reservation(
                    box=box,
                    obj=obj,
                    is_new_box=False,
                    reason="existing_box_reserved",
                )

        # 2) 新类别只能占用未被其它挂起事务预留的空箱。
        reserved_new_box_ids = {
            item.box_id
            for item in self._reservations.values()
            if item.success and item.is_new_box
        }

        for box in self.boxes:
            if box.empty() and box.box_id not in reserved_new_box_ids:
                return self._new_reservation(
                    box=box,
                    obj=obj,
                    is_new_box=True,
                    reason="new_box_reserved",
                )

        return StorageReservation(
            success=False,
            reservation_id=None,
            box_id=0,
            is_new_box=False,
            projected_count=0,
            reason="no_available_box",
            obj=None,
        )

    def commit(self, reservation_id: int) -> StorageResult:
        """提交一个预留事务；重复/未知事务不会重复计数。"""
        reservation = self._reservations.pop(int(reservation_id), None)
        if reservation is None or reservation.obj is None:
            return StorageResult(False, 0, False, 0, "unknown_reservation")

        box = next(
            (item for item in self.boxes if item.box_id == reservation.box_id),
            None,
        )
        if box is None:
            return StorageResult(False, 0, False, 0, "box_not_found")

        # 提交时再次验证目标箱没有被错误改成其它类别。
        if not box.empty() and not box.match(reservation.obj):
            return StorageResult(
                False,
                box.box_id,
                reservation.is_new_box,
                box.count,
                "box_content_changed",
            )

        box.add(reservation.obj)
        return StorageResult(
            True,
            box.box_id,
            reservation.is_new_box,
            box.count,
            "committed",
        )

    def cancel(self, reservation_id: int | None) -> bool:
        """取消尚未提交的预留。"""
        if reservation_id is None:
            return False
        return self._reservations.pop(int(reservation_id), None) is not None

    def allocate(self, obj: ObjectInfo) -> StorageResult:
        """兼容旧调试脚本：立即 reserve + commit。正式 Runtime 不使用。"""
        reservation = self.reserve(obj)
        if not reservation.success or reservation.reservation_id is None:
            return StorageResult(False, 0, False, 0, reservation.reason)
        return self.commit(reservation.reservation_id)

    def get_status(self):
        result = []
        for box in self.boxes:
            if box.empty():
                result.append({"box_id": box.box_id, "empty": True})
            else:
                result.append(
                    {
                        "box_id": box.box_id,
                        "empty": False,
                        "object": box.object_key,
                        "count": box.count,
                    }
                )
        return result

    def get_pending_reservations(self) -> list[StorageReservation]:
        return list(self._reservations.values())
