"""
图像预处理模块
负责图像加载、增强、校正等预处理操作
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
from loguru import logger


class ImagePreprocessor:
    """图像预处理器"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        
    def process(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            (enhanced_image, original_image)
        """
        logger.info(f"Loading image: {image_path}")
        
        # 加载图像
        original = self._load_image(image_path)
        
        # 验证图像
        self._validate_image(original)
        
        # 图像增强
        enhanced = self._enhance_image(original.copy())
        
        logger.info(f"Image preprocessed: {original.shape}")
        return enhanced, original
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """加载图像"""
        try:
            # 使用PIL加载以支持更多格式
            pil_image = Image.open(image_path)
            
            # 转换为RGB
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # 转换为numpy数组
            image = np.array(pil_image)
            
            # PIL使用RGB，OpenCV使用BGR，需要转换
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise
    
    def _validate_image(self, image: np.ndarray) -> None:
        """验证图像质量"""
        height, width = image.shape[:2]
        
        # 检查尺寸
        if width < 500 or height < 500:
            logger.warning(f"Image size is small: {width}x{height}")
        
        # 检查是否为空
        if image.size == 0:
            raise ValueError("Image is empty")
        
        logger.debug(f"Image validated: {width}x{height}, channels: {image.shape[2]}")
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """增强图像以提高OCR准确率"""
        
        # 1. 去噪
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        # 2. 锐化（可选）
        if self.config.get('sharpen', False):
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            denoised = cv2.filter2D(denoised, -1, kernel)
        
        # 3. 对比度增强
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def correct_skew(self, image: np.ndarray) -> np.ndarray:
        """校正图像倾斜"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 霍夫变换检测直线
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is None:
            return image
        
        # 计算角度
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            angles.append(angle)
        
        # 使用中位数角度
        median_angle = np.median(angles)
        
        # 只有在倾斜明显时才校正
        if abs(median_angle) > 0.5:
            logger.info(f"Correcting skew: {median_angle:.2f} degrees")
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            image = cv2.warpAffine(image, matrix, (width, height),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        
        return image
    
    def resize_if_needed(self, image: np.ndarray, 
                        max_dimension: int = 4000) -> np.ndarray:
        """
        如果图像过大则调整大小
        
        Args:
            image: 输入图像
            max_dimension: 最大维度
            
        Returns:
            调整后的图像
        """
        height, width = image.shape[:2]
        
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            logger.info(f"Resizing image: {width}x{height} -> {new_width}x{new_height}")
            
            image = cv2.resize(image, (new_width, new_height),
                             interpolation=cv2.INTER_AREA)
        
        return image
    
    def get_grayscale(self, image: np.ndarray) -> np.ndarray:
        """获取灰度图"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def get_binary(self, image: np.ndarray) -> np.ndarray:
        """获取二值图"""
        gray = self.get_grayscale(image)
        
        # 自适应阈值
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        return binary
