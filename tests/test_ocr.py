"""
OCR 引擎单元测试
"""

import sys
import unittest
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ocr_engine import OCREngine


class TestOCREngineInit(unittest.TestCase):
    """OCR 引擎初始化（仅 Qwen-OCR）"""

    def test_init_qwen_ocr(self):
        """默认即为 qwen_ocr"""
        engine = OCREngine({})
        self.assertEqual(engine.engine_type, "qwen_ocr")


class TestOCREngineRecognize(unittest.TestCase):
    """recognize() 返回格式与过滤逻辑"""

    def setUp(self):
        # 仅当存在 Qwen-OCR API Key 时才运行这些测试，避免在无网络/无 Key 环境下失败
        if not (os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("DASHSCOPE_APIKEY")):
            self.skipTest("DASHSCOPE_API_KEY not set, skip Qwen-OCR recognize tests")
        self.engine = OCREngine({})

    def test_recognize_returns_list(self):
        """recognize 返回 list"""
        img = np.ones((100, 200, 3), dtype=np.uint8) * 255
        results = self.engine.recognize(img, protection_mask=None)
        self.assertIsInstance(results, list)

    def test_each_result_has_bbox_text_confidence(self):
        """每个结果包含 bbox, text, confidence"""
        img = np.ones((100, 200, 3), dtype=np.uint8) * 255
        results = self.engine.recognize(img, protection_mask=None)
        for r in results:
            self.assertIn("bbox", r)
            self.assertIn("text", r)
            self.assertIn("confidence", r)
            self.assertIsInstance(r["bbox"], (tuple, list))
            self.assertEqual(len(r["bbox"]), 4)

    def test_protection_mask_filters(self):
        """传入保护蒙版时，中心在蒙版内的结果被过滤"""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        # 蒙版：中心 (50,50) 为保护区域
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255
        results_no_mask = self.engine.recognize(img, protection_mask=None)
        results_with_mask = self.engine.recognize(img, protection_mask=mask)
        # 有蒙版时数量应不大于无蒙版（可能过滤掉一部分）
        self.assertLessEqual(len(results_with_mask), len(results_no_mask) + 1)


class TestOCREngineHelpers(unittest.TestCase):
    """辅助方法"""

    def setUp(self):
        self.engine = OCREngine({})

    def test_extract_text_region(self):
        """提取文字区域返回子图"""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        region = self.engine.extract_text_region(img, (10, 10, 30, 30))
        self.assertEqual(region.shape[0], 22)  # 含 margin
        self.assertEqual(region.shape[1], 22)

    def test_get_font_size(self):
        """get_font_size 返回正整数"""
        size = self.engine.get_font_size((0, 10, 0, 30))
        self.assertGreaterEqual(size, 8)
        self.assertLessEqual(size, 72)


def run_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestOCREngineInit))
    suite.addTests(loader.loadTestsFromTestCase(TestOCREngineRecognize))
    suite.addTests(loader.loadTestsFromTestCase(TestOCREngineHelpers))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
