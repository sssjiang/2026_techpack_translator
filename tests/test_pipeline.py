"""
Tech Pack Translator Tests
"""

import unittest
import numpy as np
import cv2
from pathlib import Path
import sys

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessor import ImagePreprocessor
from src.design_detector import DesignPackDetector
from src.ocr_engine import OCREngine
from src.translator import Translator
from src.renderer import ImageRenderer
from src.pipeline import TechPackTranslator


class TestImagePreprocessor(unittest.TestCase):
    """测试图像预处理器"""
    
    def setUp(self):
        self.preprocessor = ImagePreprocessor()
        # 创建测试图像
        self.test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    def test_grayscale_conversion(self):
        """测试灰度转换"""
        gray = self.preprocessor.get_grayscale(self.test_image)
        self.assertEqual(len(gray.shape), 2)
    
    def test_binary_conversion(self):
        """测试二值化"""
        binary = self.preprocessor.get_binary(self.test_image)
        self.assertEqual(len(binary.shape), 2)
        self.assertTrue(np.all((binary == 0) | (binary == 255)))


class TestDesignDetector(unittest.TestCase):
    """测试设计图案检测器"""
    
    def setUp(self):
        self.detector = DesignPackDetector()
        self.test_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
    
    def test_is_design_pack_label(self):
        """测试标注识别"""
        self.assertTrue(self.detector._is_design_pack_label("design pack image"))
        self.assertTrue(self.detector._is_design_pack_label("Design Pack"))
        self.assertFalse(self.detector._is_design_pack_label("fabric description"))
    
    def test_create_protection_mask(self):
        """测试保护蒙版创建"""
        regions = [(10, 10, 50, 50), (100, 100, 150, 150)]
        mask = self.detector._create_protection_mask((200, 200), regions)
        self.assertEqual(mask.shape, (200, 200))
        self.assertTrue(np.any(mask > 0))


class TestTranslator(unittest.TestCase):
    """测试翻译器"""
    
    def setUp(self):
        self.translator = Translator()
    
    def test_should_not_translate(self):
        """测试不翻译规则"""
        self.assertTrue(self.translator._should_not_translate("123"))
        self.assertTrue(self.translator._should_not_translate("YKK 316"))
        self.assertTrue(self.translator._should_not_translate("N/A"))
        self.assertFalse(self.translator._should_not_translate("Cotton fabric"))
    
    def test_terminology_lookup(self):
        """测试术语查找"""
        # 这个测试假设术语库已加载
        if self.translator.terminology:
            result = self.translator._check_terminology("Cotton")
            self.assertIsNotNone(result)


class TestRenderer(unittest.TestCase):
    """测试渲染器"""
    
    def setUp(self):
        self.renderer = ImageRenderer()
        self.test_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
    
    def test_detect_background_color(self):
        """测试背景色检测"""
        from PIL import Image
        pil_image = Image.fromarray(self.test_image)
        bbox = (10, 10, 50, 50)
        color = self.renderer._detect_background_color(pil_image, bbox)
        self.assertEqual(len(color), 3)


class TestPipeline(unittest.TestCase):
    """测试完整流程"""
    
    def setUp(self):
        # 注意：这需要配置文件
        try:
            self.translator = TechPackTranslator('config/config.yaml')
        except:
            self.skipTest("Config file not available")
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.translator.preprocessor)
        self.assertIsNotNone(self.translator.design_detector)
        self.assertIsNotNone(self.translator.ocr_engine)
        self.assertIsNotNone(self.translator.translator)
        self.assertIsNotNone(self.translator.renderer)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试
    suite.addTests(loader.loadTestsFromTestCase(TestImagePreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestDesignDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestTranslator))
    suite.addTests(loader.loadTestsFromTestCase(TestRenderer))
    suite.addTests(loader.loadTestsFromTestCase(TestPipeline))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
