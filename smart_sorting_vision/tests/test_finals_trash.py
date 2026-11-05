"""决赛模式：前 5 类进分类盒，第 6 类起全部进垃圾桶。"""
from __future__ import annotations

import unittest

from vision.storage.object_info import ObjectInfo
from vision.storage.storage_manager import StorageManager


def obj(index: int) -> ObjectInfo:
    return ObjectInfo(color=f"c{index}", shape="正方体", qr_present=False)


def allocate(manager: StorageManager, info: ObjectInfo):
    reservation = manager.reserve(info)
    if reservation.success and reservation.reservation_id is not None:
        manager.commit(reservation.reservation_id)
    return reservation


class FinalsTrashTests(unittest.TestCase):
    def test_first_five_classes_fill_boxes_one_to_five(self):
        manager = StorageManager(box_count=6, assignable_count=5)
        for index in range(5):
            reservation = allocate(manager, obj(index))
            self.assertTrue(reservation.success)
            self.assertEqual(reservation.box_id, index + 1)
            self.assertEqual(reservation.reason, "new_box_reserved")
        self.assertIsNotNone(manager.trash_box_id)

    def test_sixth_class_goes_to_trash(self):
        manager = StorageManager(box_count=6, assignable_count=5)
        for index in range(5):
            allocate(manager, obj(index))
        reservation = allocate(manager, obj(5))
        self.assertTrue(reservation.success)
        self.assertEqual(reservation.box_id, 6)
        self.assertEqual(reservation.reason, "trash_box_reserved")

    def test_all_classes_after_five_reuse_trash(self):
        manager = StorageManager(box_count=6, assignable_count=5)
        for index in range(5):
            allocate(manager, obj(index))
        allocate(manager, obj(5))
        for index in range(6, 10):
            reservation = allocate(manager, obj(index))
            self.assertTrue(reservation.success)
            self.assertEqual(reservation.box_id, 6)
            self.assertEqual(reservation.reason, "trash_box_reserved")
        self.assertEqual(manager.boxes[5].count, 5)

    def test_existing_class_still_uses_its_own_box(self):
        manager = StorageManager(box_count=6, assignable_count=5)
        for index in range(5):
            allocate(manager, obj(index))
        reservation = allocate(manager, obj(1))
        self.assertTrue(reservation.success)
        self.assertEqual(reservation.box_id, 2)
        self.assertEqual(reservation.reason, "existing_box_reserved")

    def test_non_finals_mode_keeps_six_boxes_and_fails_on_seventh(self):
        manager = StorageManager(box_count=6)
        self.assertIsNone(manager.trash_box_id)
        for index in range(6):
            self.assertTrue(allocate(manager, obj(index)).success)
        seventh = allocate(manager, obj(6))
        self.assertFalse(seventh.success)
        self.assertEqual(seventh.reason, "no_available_box")


if __name__ == "__main__":
    unittest.main()
