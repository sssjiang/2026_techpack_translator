"""
OCR引擎模块
支持多种OCR引擎：PaddleOCR, Tesseract, EasyOCR
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from loguru import logger
import os


class OCREngine:
    """OCR识别引擎"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.engine_type = self.config.get('engine', 'paddleocr')
        self.languages = self.config.get('languages', ['en', 'ch'])
        self.use_gpu = self.config.get('use_gpu', False)
        
        # 初始化OCR引擎
        self.ocr = self._init_engine()
        
    def _init_engine(self):
        """初始化OCR引擎"""
        logger.info(f"Initializing OCR engine: {self.engine_type}")
        
        if self.engine_type == 'paddleocr':
            return self._init_paddle_ocr()
        elif self.engine_type == 'tesseract':
            return self._init_tesseract()
        elif self.engine_type == 'easyocr':
            return self._init_easyocr()
        else:
            raise ValueError(f"Unknown OCR engine: {self.engine_type}")
    
    def _init_paddle_ocr(self):
        """初始化PaddleOCR"""
        try:
            from paddleocr import PaddleOCR
            
            ocr = PaddleOCR(
                use_textline_orientation=True,
                lang='en',  # 主要语言
                text_det_thresh=self.config.get('det_db_thresh', 0.3),
                text_det_box_thresh=self.config.get('det_db_box_thresh', 0.6),
                text_recognition_batch_size=self.config.get('rec_batch_num', 6)
            )
            
            logger.info("PaddleOCR initialized successfully")
            return ocr
            
        except ImportError:
            logger.error("PaddleOCR not installed. Install with: pip install paddleocr")
            raise
    
    def _init_tesseract(self):
        """初始化Tesseract"""
        try:
            import pytesseract
            
            # 检查Tesseract是否安装
            try:
                pytesseract.get_tesseract_version()
            except:
                logger.error("Tesseract not found. Install from: https://github.com/tesseract-ocr/tesseract")
                raise
            
            logger.info("Tesseract initialized successfully")
            return pytesseract
            
        except ImportError:
            logger.error("pytesseract not installed. Install with: pip install pytesseract")
            raise
    
    def _init_easyocr(self):
        """初始化EasyOCR"""
        try:
            import easyocr
            
            reader = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu,
                verbose=False
            )
            
            logger.info("EasyOCR initialized successfully")
            return reader
            
        except ImportError:
            logger.error("EasyOCR not installed. Install with: pip install easyocr")
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
        logger.info("Running OCR recognition...")
        
        if self.engine_type == 'paddleocr':
            results = self._recognize_paddle(image)
        elif self.engine_type == 'tesseract':
            results = self._recognize_tesseract(image)
        elif self.engine_type == 'easyocr':
            results = self._recognize_easyocr(image)
        else:
            results = []
        
        # 过滤掉保护区域中的文字
        if protection_mask is not None:
            results = self._filter_protected_regions(results, protection_mask)
        
        logger.info(f"OCR completed: {len(results)} text regions found")
        return results
    
    def _recognize_paddle(self, image: np.ndarray) -> List[Dict]:
        """使用PaddleOCR识别"""
        result = self.ocr.predict(image)

        ocr_results = []

        if result:
            for res in result:
                if hasattr(res, 'rec_texts') and hasattr(res, 'rec_boxes'):
                    # 新版 PaddleOCR predict() 返回格式
                    for i, (text, score, box) in enumerate(
                        zip(res.rec_texts, res.rec_scores, res.rec_boxes)
                    ):
                        x_coords = [box[0], box[2]]
                        y_coords = [box[1], box[3]]
                        bbox = (
                            int(min(x_coords)),
                            int(min(y_coords)),
                            int(max(x_coords)),
                            int(max(y_coords))
                        )
                        ocr_results.append({
                            'bbox': bbox,
                            'text': text,
                            'confidence': float(score),
                            'engine': 'paddleocr'
                        })
                elif isinstance(res, list):
                    # 兼容旧版 ocr() 返回格式
                    for line in res:
                        if line:
                            box = line[0]
                            text_info = line[1]
                            text = text_info[0]
                            confidence = text_info[1]
                            x_coords = [p[0] for p in box]
                            y_coords = [p[1] for p in box]
                            bbox = (
                                int(min(x_coords)),
                                int(min(y_coords)),
                                int(max(x_coords)),
                                int(max(y_coords))
                            )
                            ocr_results.append({
                                'bbox': bbox,
                                'text': text,
                                'confidence': confidence,
                                'engine': 'paddleocr'
                            })
        
        return ocr_results
    
    def _recognize_tesseract(self, image: np.ndarray) -> List[Dict]:
        """使用Tesseract识别"""
        import pytesseract
        
        # 获取详细信息
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        ocr_results = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            if int(data['conf'][i]) > 0:  # 过滤掉置信度为-1的
                text = data['text'][i].strip()
                if text:  # 过滤掉空文本
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    bbox = (x, y, x + w, y + h)
                    confidence = int(data['conf'][i]) / 100.0
                    
                    ocr_results.append({
                        'bbox': bbox,
                        'text': text,
                        'confidence': confidence,
                        'engine': 'tesseract'
                    })
        
        return ocr_results
    
    def _recognize_easyocr(self, image: np.ndarray) -> List[Dict]:
        """使用EasyOCR识别"""
        results = self.ocr.readtext(image)
        
        ocr_results = []
        
        for detection in results:
            box, text, confidence = detection
            
            # 转换box格式
            x_coords = [p[0] for p in box]
            y_coords = [p[1] for p in box]
            bbox = (
                int(min(x_coords)),
                int(min(y_coords)),
                int(max(x_coords)),
                int(max(y_coords))
            )
            
            ocr_results.append({
                'bbox': bbox,
                'text': text,
                'confidence': confidence,
                'engine': 'easyocr'
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
