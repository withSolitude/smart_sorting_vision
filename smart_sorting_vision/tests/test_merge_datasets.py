"""YOLO 数据集合并回归；只使用临时文件，不读取或修改实际采集数据。"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from merge_datasets import main, merge_datasets


POLYGON = "0 0.1 0.1 0.8 0.1 0.8 0.8\n"


class DatasetMergeTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.base = Path(folder.name)
        self.output = self.base / "merged"

    def dataset(self, name, names, records, *, nested=False, split_specs=None):
        """图片内容仅供哈希区分，不要求安装图像解码库。"""
        source = self.base / name
        root = source / "dataset" if nested else source
        for split in ("train", "val", "test"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
        for split, filename, image_bytes, label in records:
            image = root / "images" / split / filename
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(image_bytes)
            if label is not None:
                target = root / "labels" / split / Path(filename).with_suffix(".txt")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(label, encoding="utf-8")
        for split in ("train", "val", "test"):
            paths = sorted((root / "images" / split).rglob("*.jpg"))
            (root / f"{split}.txt").write_text(
                "".join(f"./{p.relative_to(root).as_posix()}\n" for p in paths),
                encoding="utf-8",
            )
        config = {"names": names, **{s: f"images/{s}" for s in ("train", "val", "test")}}
        config.update(split_specs or {})
        (root / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
        return source

    @staticmethod
    def snapshot(root):
        return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}

    def exported(self):
        result = {}
        for image in (self.output / "images").rglob("*.jpg"):
            label = self.output / "labels" / image.relative_to(self.output / "images").with_suffix(".txt")
            self.assertTrue(label.is_file())
            result[image.read_bytes()] = (image, label.read_text(encoding="utf-8"))
        return result

    def test_cli_uses_dataset_names_and_creates_new_timestamped_output(self):
        self.dataset("旧版本", ["object"], [("train", "old.jpg", b"old", POLYGON)])
        self.dataset("补充版本", ["object"], [("val", "new.jpg", b"new", POLYGON)], nested=True)
        config = self.base / "merge.yaml"
        config.write_text(yaml.safe_dump({
            "dataset_root": str(self.base), "dataset_a": "旧版本", "dataset_b": "补充版本",
        }, allow_unicode=True), encoding="utf-8")
        with patch("sys.argv", ["merge_datasets.py", "--config", str(config)]):
            self.assertEqual(main(), 0)
        outputs = list(self.output.iterdir())
        self.assertEqual(len(outputs), 1)
        self.assertTrue(outputs[0].name.startswith("旧版本_plus_补充版本_"))
        self.assertEqual(len(list((outputs[0] / "images").rglob("*.jpg"))), 2)

    def test_same_names_are_kept_and_class_ids_follow_names(self):
        a = self.dataset("旧数据", ["方块", "球"], [("train", "scene/same.jpg", b"old", POLYGON)])
        b = self.dataset("新标注", {0: "球", 1: "方块"}, [
            ("train", "scene/same.jpg", b"new", POLYGON),
        ], nested=True, split_specs={"train": "train.txt", "val": "val.txt", "test": "test.txt"})
        before = (self.snapshot(a), self.snapshot(b))
        merge_datasets(a, b, self.output)
        records = self.exported()
        self.assertEqual(set(records), {b"old", b"new"})
        self.assertNotEqual(records[b"old"][0].name, records[b"new"][0].name)
        config = yaml.safe_load((self.output / "data.yaml").read_text(encoding="utf-8"))
        names = config["names"]
        names = [names[i] for i in sorted(names)] if isinstance(names, dict) else names
        self.assertEqual(names[int(records[b"old"][1].split()[0])], "方块")
        self.assertEqual(names[int(records[b"new"][1].split()[0])], "球")
        self.assertEqual((self.snapshot(a), self.snapshot(b)), before)
        self.assertTrue((self.output / "merge_manifest.csv").is_file())
        self.assertTrue((self.output / "summary.json").is_file())

    def test_identical_train_images_with_remapped_equal_labels_are_deduplicated(self):
        a = self.dataset("a", ["cube", "sphere"], [("train", "old.jpg", b"same", POLYGON)])
        b = self.dataset("b", ["sphere", "cube"], [
            ("train", "new.jpg", b"same", POLYGON.replace("0 ", "1 ", 1)),
        ])
        merge_datasets(a, b, self.output)
        self.assertEqual(len(list((self.output / "images").rglob("*.jpg"))), 1)

    def test_cross_split_duplicates_and_conflicting_labels_fail_before_writing(self):
        a = self.dataset("a", ["cube", "sphere"], [("train", "old.jpg", b"same", POLYGON)])
        for reason, split, label in (("split", "val", POLYGON),
                                     ("label", "train", POLYGON.replace("0 ", "1 ", 1))):
            with self.subTest(reason=reason):
                b = self.dataset(reason, ["cube", "sphere"], [(split, "new.jpg", b"same", label)])
                with self.assertRaises(ValueError):
                    merge_datasets(a, b, self.output)
                self.assertFalse(self.output.exists())

    def test_missing_label_is_not_silently_treated_as_empty_background(self):
        a = self.dataset("a", ["object"], [("train", "a.jpg", b"a", POLYGON)])
        b = self.dataset("b", ["object"], [("train", "missing.jpg", b"missing", None)])
        with self.assertRaises(ValueError):
            merge_datasets(a, b, self.output)
        self.assertFalse(self.output.exists())

    def test_detection_and_segmentation_cannot_be_mixed(self):
        a = self.dataset("segments", ["object"], [("train", "a.jpg", b"seg", POLYGON)])
        b = self.dataset("boxes", ["object"], [("train", "b.jpg", b"det", "0 0.5 0.5 0.2 0.2\n")])
        with self.assertRaises(ValueError):
            merge_datasets(a, b, self.output)
        self.assertFalse(self.output.exists())

    def test_empty_labels_and_split_lists_are_exported_with_all_new_images(self):
        a = self.dataset("a", ["object"], [
            ("train", "a.jpg", b"train", POLYGON), ("val", "empty.jpg", b"empty", ""),
        ])
        b = self.dataset("b", ["object"], [
            ("train", "scene/b.jpg", b"new-train", POLYGON), ("test", "c.jpg", b"test", POLYGON),
        ], split_specs={"train": ["train.txt"], "val": "val.txt", "test": ["images/test"]})
        merge_datasets(a, b, self.output)
        self.assertEqual(self.exported()[b"empty"][1], "")
        for split, expected in (("train", 2), ("val", 1), ("test", 1)):
            lines = (self.output / f"{split}.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), expected)
            for line in lines:
                image = Path(line)
                image = image if image.is_absolute() else self.output / image
                self.assertTrue(image.is_file())
                self.assertIn(f"images/{split}/", image.as_posix())

    def test_existing_output_directory_is_never_overwritten(self):
        a = self.dataset("a", ["object"], [("train", "a.jpg", b"a", POLYGON)])
        b = self.dataset("b", ["object"], [("train", "b.jpg", b"b", POLYGON)])
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            merge_datasets(a, b, self.output)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_yaml_file_input_and_detection_labels_are_supported(self):
        a = self.dataset("a", {0: "object"}, [("train", "a.jpg", b"a", "0 0.5 0.5 0.2 0.2\n")])
        b = self.dataset("b", ["object"], [("val", "b.jpg", b"b", "0 0.5 0.5 0.3 0.3\n")])
        merge_datasets(a / "data.yaml", b, self.output)
        self.assertEqual(len(self.exported()), 2)


if __name__ == "__main__":
    unittest.main()
