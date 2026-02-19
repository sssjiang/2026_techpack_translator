"""
OCR引擎模块
当前只保留 Qwen-OCR（阿里云百炼）用于文字识别与定位
"""

import base64
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from loguru import logger
import os


class OCREngine:
    """OCR识别引擎（仅 Qwen-OCR）"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        # 目前仅保留 Qwen-OCR
        self.engine_type = 'qwen_ocr'
        
        # 初始化OCR引擎（主要是存储 Qwen-OCR 相关配置）
        self.ocr = self._init_qwen_ocr()
    
    def _init_qwen_ocr(self):
        """初始化 Qwen-OCR（阿里云百炼 高精识别）"""
        try:
            import dashscope
            api_key = self.config.get('api_key') or os.getenv('DASHSCOPE_API_KEY')
            if not api_key or api_key == 'XXX':
                logger.warning(
                    "Qwen-OCR: Set DASHSCOPE_API_KEY or ocr.api_key in config. "
                    "Get key: https://help.aliyun.com/zh/model-studio/get-api-key"
                )
            self._qwen_base_url = self.config.get(
                'base_url', 'https://dashscope.aliyuncs.com/api/v1'
            )
            self._qwen_model = self.config.get('model', 'qwen-vl-ocr-latest')
            self._qwen_min_pixels = self.config.get('min_pixels', 32 * 32 * 3)
            self._qwen_max_pixels = self.config.get('max_pixels', 32 * 32 * 8192)
            logger.info("Qwen-OCR (advanced_recognition) initialized successfully")
            return None  # 无本地引擎实例，调用时走 API
        except ImportError:
            logger.error("dashscope not installed. Install with: pip install dashscope")
            raise

    def recognize(self, image: np.ndarray, 
                 protection_mask: Optional[np.ndarray] = None) -> List[Dict]:
        """
        识别图像中的文字
        
        Args:
            image: 输入图像
            protection_mask: 保护蒙版（这些区域不进行OCR）
            
        Returns:
            OCR结果列表
        """
        logger.info("Running OCR recognition (Qwen-OCR)...")
        results = self._recognize_qwen_ocr(image)
        
        # 过滤掉保护区域中的文字
        if protection_mask is not None:
            results = self._filter_protected_regions(results, protection_mask)
        
        logger.info(f"OCR completed: {len(results)} text regions found")
        return results

    def _recognize_qwen_ocr(self, image: np.ndarray) -> List[Dict]:
        """使用阿里云 Qwen-OCR 高精识别（带文字定位）"""
        import dashscope

        api_key = self.config.get('api_key') or os.getenv('DASHSCOPE_API_KEY')
        if not api_key or api_key == 'XXX':
            logger.error("Qwen-OCR requires DASHSCOPE_API_KEY or ocr.api_key")
            return []

        # 图像转 Base64
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        _, buf = cv2.imencode('.png', image)
        b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
        data_url = f"data:image/png;base64,{b64}"

        # 临时切换 base_url（若配置了地域）
        old_base = getattr(dashscope, 'base_http_api_url', None)
        try:
            dashscope.base_http_api_url = self._qwen_base_url
            response = dashscope.MultiModalConversation.call(
                api_key=api_key,
                model=self._qwen_model,
                messages=[{
                    'role': 'user',
                    'content': [{
                        'image': data_url,
                        'min_pixels': self._qwen_min_pixels,
                        'max_pixels': self._qwen_max_pixels,
                        'enable_rotate': self.config.get('enable_rotate', False),
                    }]
                }],
                ocr_options={'task': 'advanced_recognition'},
            )
        finally:
            if old_base is not None:
                dashscope.base_http_api_url = old_base

        ocr_results = []
        if not response or 'output' not in response:
            logger.warning("Qwen-OCR returned no output")
            return ocr_results

        choices = response.get('output', {}).get('choices', [])
        if not choices:
            return ocr_results

        content = choices[0].get('message', {}).get('content', [])
        for block in content:
            ocr_result = block.get('ocr_result')
            if not ocr_result:
                continue
            words_info = ocr_result.get('words_info', [])
            for w in words_info:
                text = w.get('text', '').strip()
                if not text:
                    continue
                location = w.get('location')  # [x1,y1, x2,y2, x3,y3, x4,y4]
                if location and len(location) >= 8:
                    xs = [location[i] for i in range(0, 8, 2)]
                    ys = [location[i] for i in range(1, 8, 2)]
                    bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                else:
                    bbox = (0, 0, 0, 0)
                ocr_results.append({
                    'bbox': bbox,
                    'text': text,
                    'confidence': 1.0,
                    'engine': 'qwen_ocr',
                })

        return ocr_results

    def _filter_protected_regions(self, results: List[Dict],
                                  mask: np.ndarray) -> List[Dict]:
        """过滤掉保护区域中的文字"""
        filtered = []
        
        for result in results:
            bbox = result['bbox']
            x1, y1, x2, y2 = bbox
            
            # 检查bbox中心点是否在保护区域内
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            if center_y < mask.shape[0] and center_x < mask.shape[1]:
                if mask[center_y, center_x] == 0:  # 不在保护区域
                    filtered.append(result)
            else:
                filtered.append(result)
        
        removed_count = len(results) - len(filtered)
        if removed_count > 0:
            logger.info(f"Filtered {removed_count} text regions in protected areas")
        
        return filtered
    
    def extract_text_region(self, image: np.ndarray, bbox: Tuple) -> np.ndarray:
        """提取文字区域"""
        x1, y1, x2, y2 = bbox
        
        # 添加小边距
        margin = 2
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(image.shape[1], x2 + margin)
        y2 = min(image.shape[0], y2 + margin)
        
        return image[y1:y2, x1:x2]
    
    def enhance_text_region(self, region: np.ndarray) -> np.ndarray:
        """增强文字区域以提高识别率"""
        # 转为灰度
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 去噪
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        
        return denoised
    
    def get_font_size(self, bbox: Tuple) -> int:
        """估算字体大小"""
        _, y1, _, y2 = bbox
        height = y2 - y1
        
        # 简单估算：文字高度约等于字号
        return max(8, min(72, height))
    
    def visualize_ocr_results(self, image: np.ndarray, 
                             results: List[Dict]) -> np.ndarray:
        """可视化OCR结果"""
        vis_image = image.copy()
        
        for result in results:
            bbox = result['bbox']
            text = result['text']
            confidence = result['confidence']
            
            x1, y1, x2, y2 = bbox
            
            # 绘制边界框
            color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # 添加文本和置信度
            label = f"{text[:20]} ({confidence:.2f})"
            cv2.putText(vis_image, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis_image
