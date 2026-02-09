"""
图像渲染模块
负责将翻译后的文字渲染回图像
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional
from loguru import logger
import os


class ImageRenderer:
    """图像渲染器"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.font_config = self.config.get('fonts', {})
        self.default_font = self.font_config.get('default', 'SimHei')
        self.preserve_layout = self.config.get('preserve_layout', True)
        self.auto_resize = self.config.get('auto_resize', True)
        
        # 加载字体
        self.fonts = self._load_fonts()
        
    def _load_fonts(self) -> Dict:
        """加载字体"""
        fonts = {}
        self._cjk_font_path = None  # 记录找到的 CJK 字体路径

        # 常用中文字体路径
        font_paths = [
            '/usr/share/fonts/opentype/noto/',
            '/usr/share/fonts/truetype/noto/',
            '/usr/share/fonts/truetype/',
            '/usr/share/fonts/opentype/',
            '/System/Library/Fonts/',
            'C:/Windows/Fonts/',
            'fonts/',  # 项目fonts目录
        ]

        # 尝试加载常用中文字体
        font_names = [
            'NotoSansCJK-Regular.ttc',  # 思源黑体
            'NotoSerifCJK-Regular.ttc', # 思源宋体
            'SimHei.ttf',      # 黑体
            'SimSun.ttf',      # 宋体
            'Arial.ttf',       # Arial (备用)
        ]

        for base_path in font_paths:
            if not os.path.exists(base_path):
                continue

            for font_name in font_names:
                font_path = os.path.join(base_path, font_name)

                if os.path.exists(font_path):
                    try:
                        # 尝试加载不同大小
                        for size in [12, 16, 20, 24, 32]:
                            font = ImageFont.truetype(font_path, size)
                            key = f"{font_name}_{size}"
                            fonts[key] = font

                        # 记录第一个成功加载的 CJK 字体名
                        if self._cjk_font_path is None:
                            self._cjk_font_path = font_path
                            self.default_font = font_name

                        logger.info(f"Loaded font: {font_name} from {font_path}")
                    except Exception as e:
                        logger.debug(f"Failed to load {font_path}: {e}")

        if not fonts:
            logger.warning("No fonts loaded, using default")
            # 使用默认字体
            for size in [12, 16, 20, 24, 32]:
                fonts[f"default_{size}"] = ImageFont.load_default()

        return fonts
    
    def render(self, 
               original_image: np.ndarray,
               ocr_results: List[Dict],
               translation_results: List[Dict],
               protection_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        渲染翻译后的图像
        
        Args:
            original_image: 原始图像
            ocr_results: OCR结果
            translation_results: 翻译结果
            protection_mask: 保护蒙版
            
        Returns:
            渲染后的图像
        """
        logger.info("Rendering translated image...")
        
        # 转换为PIL Image
        output_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(output_image)
        
        # 遍历每个翻译结果
        for i, (ocr_result, trans_result) in enumerate(zip(ocr_results, translation_results)):
            bbox = ocr_result['bbox']
            original_text = trans_result['original']
            translated_text = trans_result['translated']
            
            # 如果翻译和原文相同，跳过
            if original_text == translated_text:
                continue
            
            # 清除原文字区域
            self._clear_text_region(output_image, bbox)
            
            # 渲染新文字
            self._render_text(draw, bbox, translated_text, ocr_result)
        
        # 恢复保护区域
        if protection_mask is not None:
            output_image = self._restore_protected_regions(
                output_image, original_image, protection_mask
            )
        
        # 转换回OpenCV格式
        output_cv = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)
        
        logger.info("Rendering completed")
        return output_cv
    
    def _clear_text_region(self, image: Image.Image, bbox: Tuple):
        """清除文字区域"""
        x1, y1, x2, y2 = bbox
        
        # 使用白色填充（或检测背景色）
        background_color = self._detect_background_color(image, bbox)
        
        draw = ImageDraw.Draw(image)
        draw.rectangle([x1, y1, x2, y2], fill=background_color)
    
    def _detect_background_color(self, image: Image.Image, 
                                 bbox: Tuple) -> Tuple[int, int, int]:
        """检测背景色"""
        x1, y1, x2, y2 = bbox
        
        # 扩展区域以获取背景
        margin = 5
        bg_x1 = max(0, x1 - margin)
        bg_y1 = max(0, y1 - margin)
        bg_x2 = min(image.width, x1 + margin)
        bg_y2 = min(image.height, y2 + margin)
        
        # 采样边缘像素
        img_array = np.array(image)
        
        # 获取顶部边缘
        if bg_y1 < y1:
            top_pixels = img_array[bg_y1:y1, x1:x2]
            if top_pixels.size > 0:
                avg_color = np.mean(top_pixels, axis=(0, 1))
                return tuple(avg_color.astype(int))
        
        # 默认白色
        return (255, 255, 255)
    
    def _render_text(self, draw: ImageDraw.Draw, bbox: Tuple,
                    text: str, ocr_result: Dict):
        """渲染文字"""
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        # 动态选择字体大小，适配 bbox
        font, font_size = self._select_font(text, bbox_width, bbox_height)

        # 用实际 textbbox 计算位置
        try:
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except:
            text_width = len(text) * font_size
            text_height = font_size

        # 居中对齐
        text_x = x1 + (bbox_width - text_width) / 2
        text_y = y1 + (bbox_height - text_height) / 2

        draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))

    def _select_font(self, text: str, bbox_width: int, bbox_height: int
                    ) -> Tuple[ImageFont.FreeTypeFont, int]:
        """动态选择合适的字体大小，确保文字完全在 bbox 内"""
        if not self._cjk_font_path:
            return ImageFont.load_default(), 12

        # 目标：字体高度不超过 bbox 高度，文字总宽度不超过 bbox 宽度
        # 从 bbox 高度的 70% 开始尝试，逐步缩小
        max_size = max(8, int(bbox_height * 0.7))
        min_size = 6

        for size in range(max_size, min_size - 1, -1):
            try:
                font = ImageFont.truetype(self._cjk_font_path, size)
                # 用 getlength 精确测量文字宽度
                text_width = font.getlength(text)
                if text_width <= bbox_width * 0.95:
                    return font, size
            except:
                continue

        # 用最小字号
        try:
            font = ImageFont.truetype(self._cjk_font_path, min_size)
            return font, min_size
        except:
            return ImageFont.load_default(), min_size
    
    def _restore_protected_regions(self, output_image: Image.Image,
                                  original_image: np.ndarray,
                                  protection_mask: np.ndarray) -> Image.Image:
        """恢复保护区域"""
        # 转换为numpy数组
        output_array = np.array(output_image)
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # 应用蒙版
        mask_3d = np.stack([protection_mask] * 3, axis=2) > 0
        output_array[mask_3d] = original_rgb[mask_3d]
        
        # 转换回PIL Image
        return Image.fromarray(output_array)
    
    def create_comparison(self, original: np.ndarray, 
                         translated: np.ndarray,
                         mode: str = 'side_by_side') -> np.ndarray:
        """
        创建对比图
        
        Args:
            original: 原图
            translated: 翻译后的图
            mode: 对比模式 ('side_by_side', 'overlay')
            
        Returns:
            对比图
        """
        if mode == 'side_by_side':
            # 并排显示
            height = max(original.shape[0], translated.shape[0])
            width = original.shape[1] + translated.shape[1]
            
            comparison = np.zeros((height, width, 3), dtype=np.uint8)
            comparison.fill(255)  # 白色背景
            
            # 放置原图
            comparison[:original.shape[0], :original.shape[1]] = original
            
            # 放置翻译图
            comparison[:translated.shape[0], 
                      original.shape[1]:original.shape[1]+translated.shape[1]] = translated
            
            # 添加标签
            cv2.putText(comparison, "Original", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(comparison, "Translated", 
                       (original.shape[1] + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return comparison
        
        elif mode == 'overlay':
            # 半透明叠加
            alpha = 0.5
            comparison = cv2.addWeighted(original, alpha, translated, 1-alpha, 0)
            return comparison
        
        else:
            return translated
