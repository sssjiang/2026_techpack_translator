"""
翻译器单元测试
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.translator import Translator


class TestTranslatorInit(unittest.TestCase):
    """翻译器初始化与语言代码"""

    def test_normalize_lang_code_zh(self):
        """语言代码 zh/zh-cn 规范为 zh-CN"""
        t = Translator({"target_lang": "zh"})
        self.assertEqual(t.target_lang, "zh-CN")
        t2 = Translator({"target_lang": "zh-cn"})
        self.assertEqual(t2.target_lang, "zh-CN")

    def test_normalize_lang_code_en(self):
        """英语代码保持不变"""
        t = Translator({"source_lang": "en"})
        self.assertEqual(t.source_lang, "en")

    def test_default_config(self):
        """无配置时引擎固定为 deepl"""
        t = Translator({})
        self.assertEqual(t.engine_type, "deepl")


class TestTranslatorShouldNotTranslate(unittest.TestCase):
    """不翻译规则"""

    def setUp(self):
        self.translator = Translator({})

    def test_pure_digits(self):
        self.assertTrue(self.translator._should_not_translate("123"))
        self.assertTrue(self.translator._should_not_translate("18\"L"))

    def test_code_like(self):
        self.assertTrue(self.translator._should_not_translate("YKK 316"))
        self.assertTrue(self.translator._should_not_translate("CB"))
        self.assertTrue(self.translator._should_not_translate("DTM"))

    def test_na(self):
        self.assertTrue(self.translator._should_not_translate("N/A"))
        self.assertTrue(self.translator._should_not_translate("TBD"))

    def test_normal_text(self):
        self.assertFalse(self.translator._should_not_translate("Cotton fabric"))
        self.assertFalse(self.translator._should_not_translate("Main Fabric"))


class TestTranslatorTerminology(unittest.TestCase):
    """术语库查找（依赖 config/terminology.json）"""

    def setUp(self):
        self.translator = Translator({
            "terminology_file": str(ROOT / "config" / "terminology.json"),
        })

    def test_terminology_loaded(self):
        """术语库应被加载"""
        self.assertIsInstance(self.translator.terminology, dict)

    def test_check_terminology_exact(self):
        """精确匹配术语"""
        if "Cotton" in self.translator.terminology:
            result = self.translator._check_terminology("Cotton")
            self.assertIsNotNone(result)
            self.assertEqual(result, self.translator.terminology["Cotton"])

    def test_check_terminology_no_match(self):
        """无关词不匹配"""
        result = self.translator._check_terminology("xyznonexistent123")
        self.assertIsNone(result)


class TestTranslatorTranslate(unittest.TestCase):
    """translate() 返回结构与缓存"""

    def setUp(self):
        # 为避免测试时真实调用 DeepL，将 translator.translator 置为 None
        self.translator = Translator({})
        self.translator.translator = None

    def test_empty_text(self):
        """空文本返回原样"""
        out = self.translator.translate("")
        self.assertIsInstance(out, dict)
        self.assertEqual(out["original"], "")
        self.assertEqual(out["translated"], "")
        self.assertEqual(out["method"], "empty")

    def test_result_structure(self):
        """返回包含 original, translated, confidence, method"""
        out = self.translator.translate("hello")
        self.assertIn("original", out)
        self.assertIn("translated", out)
        self.assertIn("confidence", out)
        self.assertIn("method", out)
        self.assertEqual(out["original"], "hello")

    def test_cache_used(self):
        """相同文本第二次调用应命中缓存"""
        self.translator.translate("cache me")
        out2 = self.translator.translate("cache me")
        self.assertIn("cache me", self.translator.cache)
        self.assertEqual(out2["original"], "cache me")


def run_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestTranslatorInit))
    suite.addTests(loader.loadTestsFromTestCase(TestTranslatorShouldNotTranslate))
    suite.addTests(loader.loadTestsFromTestCase(TestTranslatorTerminology))
    suite.addTests(loader.loadTestsFromTestCase(TestTranslatorTranslate))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
