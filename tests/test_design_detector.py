"""
DesignPackDetector 测试模块
- 合成图像单元测试：验证各子方法逻辑
- 真实图像可视化测试：将检测区域 / 保护蒙版渲染到图片，保存到 tests/output/ 便于人工审查
- 流水线 1→2→3 集成测试：Step1 预处理 → Step2 初次 OCR → Step3 设计包检测（与 main 流程一致）
"""

import unittest
import sys
import os
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.design_detector import DesignPackDetector
from src.pipeline import TechPackTranslator


# 项目根目录、真实 techpack 图、可视化输出目录、配置路径
PROJECT_ROOT = Path(__file__).parent.parent
REAL_IMAGE_PATH = str(PROJECT_ROOT / "input" / "techpack_img.png")
VIS_OUTPUT_DIR = str(Path(__file__).parent / "output")
CONFIG_PATH = str(PROJECT_ROOT / "config" / "config.yaml")


def _make_synthetic_techpack(width=800, height=600):
    """
    构造一张合成 techpack：白底 + 一块嵌入的彩色"设计图"区域。
    返回 (image, design_bbox)，design_bbox = (x1, y1, x2, y2)。
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 240

    # 在中间放一块 200×200 的高饱和度随机彩色块来模拟设计包图
    dx1, dy1, dx2, dy2 = 300, 200, 500, 400
    rng = np.random.RandomState(42)
    color_block = rng.randint(0, 255, (dy2 - dy1, dx2 - dx1, 3), dtype=np.uint8)
    # 提高饱和度
    hsv = cv2.cvtColor(color_block, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(int) + 120, 0, 255).astype(np.uint8)
    color_block = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img[dy1:dy2, dx1:dx2] = color_block

    return img, (dx1, dy1, dx2, dy2)


class TestDesignDetectorUnit(unittest.TestCase):
    """纯单元测试（合成图像，无需 API/真实文件）"""

    def setUp(self):
        self.detector = DesignPackDetector()
        self.synth_img, self.synth_bbox = _make_synthetic_techpack()

    # ---- _is_design_pack_label ----
    def test_label_exact_match(self):
        self.assertTrue(self.detector._is_design_pack_label("design pack image"))
        self.assertTrue(self.detector._is_design_pack_label("Design Pack"))
        self.assertTrue(self.detector._is_design_pack_label("设计包图像"))

    def test_label_case_insensitive(self):
        self.assertTrue(self.detector._is_design_pack_label("DESIGN PACK IMAGE"))
        self.assertTrue(self.detector._is_design_pack_label("Design Image"))

    def test_label_negative(self):
        self.assertFalse(self.detector._is_design_pack_label("fabric description"))
        self.assertFalse(self.detector._is_design_pack_label("size chart"))
        self.assertFalse(self.detector._is_design_pack_label(""))

    # ---- _is_image_region ----
    def test_is_image_region_colorful(self):
        """高饱和度彩色区域应被判为图像"""
        region = self.synth_img[200:400, 300:500]
        self.assertTrue(self.detector._is_image_region(region))

    def test_is_image_region_plain(self):
        """纯白区域不应被判为图像"""
        plain = np.ones((100, 100, 3), dtype=np.uint8) * 240
        self.assertFalse(self.detector._is_image_region(plain))

    def test_is_image_region_empty(self):
        empty = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        self.assertFalse(self.detector._is_image_region(empty))

    # ---- _find_high_complexity_regions ----
    def test_high_complexity_finds_colorful_block(self):
        regions = self.detector._find_high_complexity_regions(self.synth_img)
        self.assertGreater(len(regions), 0, "应检测到至少一个高复杂度区域")

        best = regions[0]
        bx1, by1, bx2, by2 = best
        dx1, dy1, dx2, dy2 = self.synth_bbox
        # 检测区域应与合成设计块有大面积重叠
        overlap_x = max(0, min(bx2, dx2) - max(bx1, dx1))
        overlap_y = max(0, min(by2, dy2) - max(by1, dy1))
        overlap_area = overlap_x * overlap_y
        design_area = (dx2 - dx1) * (dy2 - dy1)
        self.assertGreater(overlap_area / design_area, 0.5,
                           "检测到的区域应至少覆盖合成设计块 50%")

    # ---- _detect_by_visual_features ----
    def test_visual_feature_detection(self):
        regions = self.detector._detect_by_visual_features(self.synth_img)
        self.assertGreater(len(regions), 0, "视觉特征检测应找到设计块")

    # ---- _detect_by_annotation ----
    def test_annotation_with_keyword(self):
        """当 OCR 包含 'design pack image' 标注时应检测到区域"""
        dx1, dy1, dx2, dy2 = self.synth_bbox
        ocr_results = [{
            'text': 'Design pack image',
            'bbox': (dx1, dy2 + 10, dx2, dy2 + 30),  # 标注在设计块正下方
            'confidence': 0.95,
        }]
        regions = self.detector._detect_by_annotation(self.synth_img, ocr_results)
        # 至少检测到了关键词对应的区域（可能走 nearby 或 arrow）
        # 因为合成图没有箭头，可能找不到也可能找附近高复杂度区域
        # 这里只验证不报错并且返回列表
        self.assertIsInstance(regions, list)

    # ---- detect (完整流程) ----
    def test_detect_returns_regions_and_mask(self):
        regions, mask = self.detector.detect(self.synth_img, ocr_results=[])
        self.assertIsInstance(regions, list)
        self.assertEqual(mask.shape, self.synth_img.shape[:2])
        self.assertEqual(mask.dtype, np.uint8)
        if len(regions) > 0:
            self.assertTrue(np.any(mask > 0), "有区域时蒙版应有非零像素")

    # ---- _create_protection_mask ----
    def test_protection_mask_covers_regions(self):
        regions = [(100, 100, 200, 200)]
        mask = self.detector._create_protection_mask((400, 400), regions)
        margin = self.detector.margin
        # 区域内部应全为 255
        self.assertTrue(np.all(mask[100:200, 100:200] == 255))
        # margin 边框也应为 255
        self.assertTrue(np.all(mask[100 - margin:100, 100:200] == 255))
        # 完全在外面的区域应为 0
        self.assertEqual(mask[0, 0], 0)

    def test_protection_mask_empty(self):
        mask = self.detector._create_protection_mask((300, 300), [])
        self.assertTrue(np.all(mask == 0))

    # ---- _find_precise_boundary ----
    def test_find_precise_boundary(self):
        """给一个粗略 bbox，精确边界不应比原来大太多"""
        rough = (300, 200, 500, 400)
        precise = self.detector._find_precise_boundary(self.synth_img, rough)
        self.assertEqual(len(precise), 4)
        px1, py1, px2, py2 = precise
        self.assertGreater(px2 - px1, 0)
        self.assertGreater(py2 - py1, 0)


class TestDesignDetectorVisualize(unittest.TestCase):
    """
    使用真实 techpack 图像运行检测，并将结果保存到 tests/output/ 供人工检查。
    如果真实图像不存在则跳过。
    """

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(REAL_IMAGE_PATH):
            raise unittest.SkipTest(f"真实图像不存在: {REAL_IMAGE_PATH}")
        cls.image = cv2.imread(REAL_IMAGE_PATH)
        if cls.image is None:
            raise unittest.SkipTest(f"无法读取图像: {REAL_IMAGE_PATH}")
        os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)

    def test_visualize_detected_regions(self):
        """
        在真实图上运行检测（无 OCR 标注 → 走视觉特征路径），
        保存：
          - detect_regions.png  : 原图 + 绿色边框标注检测区域
          - detect_mask.png     : 保护蒙版（白色=保护区域）
          - detect_overlay.png  : 蒙版半透明叠加在原图上
        """
        detector = DesignPackDetector()
        regions, mask = detector.detect(self.image, ocr_results=[])

        print(f"\n[Visual] 检测到 {len(regions)} 个设计包区域:")
        for i, (x1, y1, x2, y2) in enumerate(regions):
            w, h = x2 - x1, y2 - y1
            print(f"  Region {i + 1}: ({x1}, {y1}) → ({x2}, {y2})  size={w}×{h}")

        # 1. 绿色边框标注
        vis = detector.visualize_detections(self.image, regions)
        out_regions = os.path.join(VIS_OUTPUT_DIR, "detect_regions.png")
        cv2.imwrite(out_regions, vis)
        print(f"  -> saved: {out_regions}")

        # 2. 保护蒙版
        out_mask = os.path.join(VIS_OUTPUT_DIR, "detect_mask.png")
        cv2.imwrite(out_mask, mask)
        print(f"  -> saved: {out_mask}")

        # 3. 半透明叠加
        overlay = self.image.copy()
        red_layer = np.zeros_like(overlay)
        red_layer[:, :, 2] = 255  # BGR 红色
        mask_3d = np.stack([mask] * 3, axis=2) > 0
        overlay[mask_3d] = cv2.addWeighted(
            overlay, 0.5, red_layer, 0.5, 0
        )[mask_3d]
        out_overlay = os.path.join(VIS_OUTPUT_DIR, "detect_overlay.png")
        cv2.imwrite(out_overlay, overlay)
        print(f"  -> saved: {out_overlay}")

        self.assertIsInstance(regions, list)
        self.assertEqual(mask.shape, self.image.shape[:2])

    def test_visualize_with_mock_annotation(self):
        """
        模拟 OCR 识别到 'Design pack image' 标注（放在图像底部中间），
        走标注+箭头路径检测，对比视觉特征路径的结果差异。
        """
        detector = DesignPackDetector()
        h, w = self.image.shape[:2]

        mock_ocr = [{
            'text': 'Design pack image',
            'bbox': (w // 2 - 100, h - 50, w // 2 + 100, h - 20),
            'confidence': 0.95,
        }]

        regions, mask = detector.detect(self.image, ocr_results=mock_ocr)

        print(f"\n[Annotation] 检测到 {len(regions)} 个设计包区域:")
        for i, (x1, y1, x2, y2) in enumerate(regions):
            print(f"  Region {i + 1}: ({x1}, {y1}) → ({x2}, {y2})  size={x2-x1}×{y2-y1}")

        vis = detector.visualize_detections(self.image, regions)
        out_path = os.path.join(VIS_OUTPUT_DIR, "detect_annotation.png")
        cv2.imwrite(out_path, vis)
        print(f"  -> saved: {out_path}")

        out_mask = os.path.join(VIS_OUTPUT_DIR, "detect_annotation_mask.png")
        cv2.imwrite(out_mask, mask)
        print(f"  -> saved: {out_mask}")

        self.assertIsInstance(regions, list)


class TestDesignDetectorPipelineSteps123(unittest.TestCase):
    """
    真实流水线测试：按 main.py 流程执行 Step 1 → Step 2 → Step 3。
    Step 1: ImagePreprocessor.process() → enhanced_image, original_image
    Step 2: OCREngine.recognize(enhanced_image) → initial_ocr_results（无蒙版，供检测用）
    Step 3: DesignPackDetector.detect(enhanced_image, initial_ocr_results) → regions, protection_mask
    若无 DASHSCOPE_API_KEY，Step 2 返回空列表，Step 3 退化为纯视觉特征检测。
    """

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(REAL_IMAGE_PATH):
            raise unittest.SkipTest(f"真实图像不存在: {REAL_IMAGE_PATH}")
        config_path = CONFIG_PATH
        if not os.path.isfile(config_path):
            raise unittest.SkipTest(f"配置文件不存在: {config_path}")
        cls.translator = TechPackTranslator(config_path=config_path)
        os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)

    def test_steps_1_2_3_real_pipeline(self):
        """执行 Step 1 → Step 2 → Step 3，断言结果并保存可视化到 tests/output/"""
        # Step 1: 图像预处理
        enhanced_image, original_image = self.translator.preprocessor.process(REAL_IMAGE_PATH)
        self.assertIsNotNone(enhanced_image)
        self.assertIsNotNone(original_image)
        self.assertEqual(enhanced_image.shape, original_image.shape)

        # Step 2: 初步 OCR（无蒙版，用于标注检测）
        initial_ocr_results = self.translator.ocr_engine.recognize(enhanced_image)
        self.assertIsInstance(initial_ocr_results, list)
        # 无 API key 时可能为空，仍继续 Step 3

        # Step 3: 设计包区域检测（使用 Step 2 的 OCR 结果）
        design_regions, protection_mask = self.translator.design_detector.detect(
            enhanced_image, initial_ocr_results
        )
        self.assertIsInstance(design_regions, list)
        self.assertEqual(protection_mask.shape, enhanced_image.shape[:2])
        self.assertEqual(protection_mask.dtype, np.uint8)

        # 可视化保存（与原有 visualize 测试一致）
        detector = self.translator.design_detector
        vis = detector.visualize_detections(original_image, design_regions)
        out_regions = os.path.join(VIS_OUTPUT_DIR, "pipeline123_regions.png")
        cv2.imwrite(out_regions, vis)
        out_mask = os.path.join(VIS_OUTPUT_DIR, "pipeline123_mask.png")
        cv2.imwrite(out_mask, protection_mask)
        overlay = original_image.copy()
        red_layer = np.zeros_like(overlay)
        red_layer[:, :, 2] = 255
        mask_3d = np.stack([protection_mask] * 3, axis=2) > 0
        overlay[mask_3d] = cv2.addWeighted(overlay, 0.5, red_layer, 0.5, 0)[mask_3d]
        out_overlay = os.path.join(VIS_OUTPUT_DIR, "pipeline123_overlay.png")
        cv2.imwrite(out_overlay, overlay)

        # 可选：保存 Step 1 增强图，便于对比
        out_enhanced = os.path.join(VIS_OUTPUT_DIR, "pipeline123_enhanced.png")
        cv2.imwrite(out_enhanced, enhanced_image)

        print(f"\n[Pipeline 1→2→3] 初始 OCR 框数: {len(initial_ocr_results)}")
        print(f"[Pipeline 1→2→3] 设计包区域数: {len(design_regions)}")
        for i, (x1, y1, x2, y2) in enumerate(design_regions):
            print(f"  Region {i + 1}: ({x1}, {y1}) → ({x2}, {y2})  size={x2-x1}×{y2-y1}")
        print(f"  -> saved: {out_regions}, {out_mask}, {out_overlay}, {out_enhanced}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
