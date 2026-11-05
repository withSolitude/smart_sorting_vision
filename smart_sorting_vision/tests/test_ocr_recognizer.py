"""OCR 文字识别：默认关闭、缺引擎不崩、与侧面识别结果对接。"""
from __future__ import annotations

import unittest

import numpy as np

from vision.pipeline.side_recognizer import SideRecognizer
from vision.text.ocr_recognizer import TextRecognizer


class FakeColor:
    def classify(self, crop, mask):
        return {"color": "red", "confidence": 0.9}


class FakeQR:
    def __init__(self, found=False):
        self.found = found

    def detect(self, crop):
        return {"found": self.found, "text": "QR" if self.found else ""}


class FakeOCR:
    def __init__(self, result=None):
        self.result = result or {
            "found": True,
            "text": "ABC123",
            "confidence": 0.88,
            "engine": "fake",
            "reason": "ok",
            "box": (1, 2, 30, 20),
        }
        self.calls = 0

    def recognize(self, crop, mask):
        self.calls += 1
        return dict(self.result)


def image():
    crop = np.full((40, 40, 3), 200, dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[8:32, 8:32] = 255
    return crop, mask


class TextRecognizerTests(unittest.TestCase):
    def test_unknown_engine_is_reported_not_raised(self):
        recognizer = TextRecognizer({"engine": "does_not_exist"})
        crop, mask = image()
        result = recognizer.recognize(crop, mask)
        self.assertFalse(result["found"])
        self.assertIn("engine_missing", result["reason"])
        self.assertEqual(result["text"], "")

    def test_auto_without_engine_is_graceful(self):
        recognizer = TextRecognizer({"engine": "auto"})
        crop, mask = image()
        result = recognizer.recognize(crop, mask)
        self.assertFalse(result["found"])
        self.assertIn("engine_missing", result["reason"])

    def test_empty_mask_returns_reason(self):
        recognizer = TextRecognizer({"engine": "auto"})
        crop, _ = image()
        mask = np.zeros((40, 40), dtype=np.uint8)
        result = recognizer.recognize(crop, mask)
        self.assertFalse(result["found"])
        self.assertIn(result["reason"], ("empty_mask", "engine_missing:auto"))


class SideRecognizerOCRTests(unittest.TestCase):
    def make(self, *, qr_found=False, run_when_qr_found=False):
        ocr = FakeOCR()
        recognizer = SideRecognizer(
            color_classifier=FakeColor(),
            qr_detector=FakeQR(qr_found),
            shape_classifier=None,
            ocr_recognizer=ocr,
            enable_color=True,
            enable_qr=True,
            enable_shape=False,
            enable_ocr=True,
            ocr_run_when_qr_found=run_when_qr_found,
        )
        return recognizer, ocr

    def test_text_is_returned_with_details(self):
        recognizer, ocr = self.make()
        crop, mask = image()
        result = recognizer.recognize(crop, mask)
        self.assertEqual(result["text"], "ABC123")
        self.assertAlmostEqual(result["text_confidence"], 0.88)
        self.assertTrue(result["details"]["ocr"]["found"])
        self.assertEqual(ocr.calls, 1)

    def test_ocr_skipped_when_qr_found_by_default(self):
        recognizer, ocr = self.make(qr_found=True)
        crop, mask = image()
        result = recognizer.recognize(crop, mask)
        self.assertEqual(result["text"], "")
        self.assertEqual(result["details"]["ocr"]["reason"], "skipped_qr_present")
        self.assertEqual(ocr.calls, 0)

    def test_ocr_runs_with_qr_when_configured(self):
        recognizer, ocr = self.make(qr_found=True, run_when_qr_found=True)
        crop, mask = image()
        result = recognizer.recognize(crop, mask)
        self.assertEqual(result["text"], "ABC123")
        self.assertEqual(ocr.calls, 1)

    def test_disabled_ocr_returns_empty_text(self):
        recognizer = SideRecognizer(
            color_classifier=FakeColor(),
            qr_detector=FakeQR(False),
            shape_classifier=None,
            enable_color=True,
            enable_qr=True,
            enable_shape=False,
            enable_ocr=False,
        )
        crop, mask = image()
        result = recognizer.recognize(crop, mask)
        self.assertEqual(result["text"], "")
        self.assertEqual(result["details"]["ocr"]["reason"], "disabled")


if __name__ == "__main__":
    unittest.main()
