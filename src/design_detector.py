"""
设计包图像检测模块
负责检测和保护技术包中的设计图案区域
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from loguru import logger
import re


class DesignPackDetector:
    """设计包图像检测器"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.keywords = self.config.get('keywords', [
            'design pack image',
            'design pack',
            'design image',
            '设计包图像'
        ])
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.margin = self.config.get('protection_margin', 10)
        # 检测策略: visual_features（仅视觉） | annotation_first（先标注再回退视觉）
        self.detection_mode = (self.config.get('design_pack_detection_mode') or 'visual_features').strip().lower()
        if self.detection_mode not in ('visual_features', 'annotation_first'):
            self.detection_mode = 'visual_features'

    def detect(self, image: np.ndarray,
               ocr_results: List[Dict]) -> Tuple[List[Tuple], np.ndarray]:
        """
        检测设计包图像区域

        Args:
            image: 输入图像
            ocr_results: OCR识别结果列表（annotation_first 时用于标注定位）

        Returns:
            (design_pack_regions, protection_mask)
        """
        logger.info("Detecting design pack image regions...")
        regions = []

        if self.detection_mode == 'visual_features':
            # 方案二：仅用视觉特征（默认）
            logger.info("Using visual feature detection (config: design_pack_detection_mode=visual_features)")
            regions = self._detect_by_visual_features(image)
        else:
            # 方案一：先标注再回退视觉
            logger.info("Using annotation-first detection (config: design_pack_detection_mode=annotation_first)")
            text_based_regions = self._detect_by_annotation(image, ocr_results)
            regions.extend(text_based_regions)
            if len(regions) == 0:
                logger.info("No annotation found, falling back to visual feature detection...")
                regions = self._detect_by_visual_features(image)

        mask = self._create_protection_mask(image.shape[:2], regions)
        logger.info(f"Found {len(regions)} design pack region(s)")
        return regions, mask
    
    def _detect_by_annotation(self, image: np.ndarray, 
                              ocr_results: List[Dict]) -> List[Tuple]:
        """通过文本标注检测"""
        regions = []
        
        for result in ocr_results:
            text = result.get('text', '').lower().strip()
            bbox = result.get('bbox')
            
            # 检查是否包含关键词
            if self._is_design_pack_label(text):
                logger.info(f"Found design pack label: '{text}' at {bbox}")
                
                # 尝试追踪箭头指向的区域
                target_region = self._trace_arrow_target(image, bbox, ocr_results)
                
                if target_region:
                    regions.append(target_region)
                else:
                    # 如果没有箭头，尝试查找附近的图像区域
                    nearby_region = self._find_nearby_image_region(image, bbox)
                    if nearby_region:
                        regions.append(nearby_region)
        
        return regions
    
    def _is_design_pack_label(self, text: str) -> bool:
        """判断文本是否为设计包标注"""
        text = text.lower().strip()
        
        for keyword in self.keywords:
            # 使用模糊匹配
            if keyword.lower() in text:
                return True
            
            # 也接受部分匹配，如 "design pack" 在 "design pack image" 中
            words = text.split()
            keyword_words = keyword.lower().split()
            if all(kw in words for kw in keyword_words):
                return True
        
        return False
    
    def _trace_arrow_target(self, image: np.ndarray, 
                           label_bbox: Tuple,
                           ocr_results: List[Dict]) -> Optional[Tuple]:
        """追踪箭头指向的目标区域"""
        x1, y1, x2, y2 = label_bbox
        
        # 在标注附近寻找箭头或线条
        # 扩展搜索区域
        search_margin = 200
        search_x1 = max(0, x1 - search_margin)
        search_y1 = max(0, y1 - search_margin)
        search_x2 = min(image.shape[1], x2 + search_margin)
        search_y2 = min(image.shape[0], y2 + search_margin)
        
        search_region = image[search_y1:search_y2, search_x1:search_x2]
        
        # 检测线条
        gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                               minLineLength=30, maxLineGap=10)
        
        if lines is not None and len(lines) > 0:
            # 找到最长的线
            longest_line = max(lines, key=lambda l: 
                             np.sqrt((l[0][2]-l[0][0])**2 + (l[0][3]-l[0][1])**2))
            
            # 线的终点（相对于搜索区域）
            _, _, end_x, end_y = longest_line[0]
            
            # 转换为图像坐标
            target_x = search_x1 + end_x
            target_y = search_y1 + end_y
            
            # 在目标点附近查找图像区域
            target_region = self._find_image_at_point(image, target_x, target_y)
            if target_region:
                return target_region
        
        return None
    
    def _find_nearby_image_region(self, image: np.ndarray, 
                                  bbox: Tuple, 
                                  search_radius: int = 300) -> Optional[Tuple]:
        """在标注附近查找图像区域"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # 定义搜索区域
        search_x1 = max(0, center_x - search_radius)
        search_y1 = max(0, center_y - search_radius)
        search_x2 = min(image.shape[1], center_x + search_radius)
        search_y2 = min(image.shape[0], center_y + search_radius)
        
        search_region = image[search_y1:search_y2, search_x1:search_x2]
        
        # 查找高色彩复杂度区域
        regions = self._find_high_complexity_regions(search_region)
        
        if regions:
            # 转换为图像坐标
            best_region = regions[0]
            rx1, ry1, rx2, ry2 = best_region
            return (
                search_x1 + rx1,
                search_y1 + ry1,
                search_x1 + rx2,
                search_y1 + ry2
            )
        
        return None
    
    def _find_image_at_point(self, image: np.ndarray, 
                            x: int, y: int,
                            size: int = 100) -> Optional[Tuple]:
        """在指定点查找图像区域"""
        # 从点开始扩展查找区域
        x1 = max(0, x - size)
        y1 = max(0, y - size)
        x2 = min(image.shape[1], x + size)
        y2 = min(image.shape[0], y + size)
        
        region = image[y1:y2, x1:x2]
        
        # 检查是否为图像区域（高色彩复杂度）
        if self._is_image_region(region):
            # 尝试找到精确边界
            precise_bbox = self._find_precise_boundary(image, (x1, y1, x2, y2))
            return precise_bbox
        
        return None
    
    def _detect_by_visual_features(self, image: np.ndarray) -> List[Tuple]:
        """基于视觉特征检测设计图像"""
        regions = self._find_high_complexity_regions(image)
        
        # 过滤掉太小的区域（可能是表格中的小图标）
        min_area = 10000  # 最小100x100像素
        filtered_regions = []
        
        for region in regions:
            x1, y1, x2, y2 = region
            area = (x2 - x1) * (y2 - y1)
            if area >= min_area:
                filtered_regions.append(region)
        
        return filtered_regions
    
    def _find_high_complexity_regions(self, image: np.ndarray, 
                                     threshold: float = 0.15) -> List[Tuple]:
        """查找高复杂度区域（可能是设计图案）"""
        # 转换为HSV以分析颜色
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 计算色彩丰富度
        h, s, v = cv2.split(hsv)
        
        # 高饱和度区域（彩色图案）
        high_saturation = (s > 50).astype(np.uint8) * 255
        
        # 形态学操作连接区域
        kernel = np.ones((20, 20), np.uint8)
        closed = cv2.morphologyEx(high_saturation, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # 检查区域是否足够大且饱和度足够高
            region = s[y:y+h, x:x+w]
            avg_saturation = np.mean(region)
            
            if avg_saturation > 30 and w * h > 5000:
                regions.append((x, y, x+w, y+h))
        
        # 按面积排序，大的在前
        regions.sort(key=lambda r: (r[2]-r[0])*(r[3]-r[1]), reverse=True)
        
        return regions
    
    def _is_image_region(self, region: np.ndarray) -> bool:
        """判断区域是否为图像（vs文字/表格）"""
        if region.size == 0:
            return False
        
        # 转换为HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 计算平均饱和度
        avg_saturation = np.mean(s)
        
        # 计算颜色种类
        unique_colors = len(np.unique(region.reshape(-1, 3), axis=0))
        total_pixels = region.shape[0] * region.shape[1]
        color_diversity = unique_colors / total_pixels
        
        # 图像通常有较高的饱和度和颜色多样性
        return avg_saturation > 20 and color_diversity > 0.1
    
    def _find_precise_boundary(self, image: np.ndarray, 
                              rough_bbox: Tuple) -> Tuple:
        """精确查找图像边界"""
        x1, y1, x2, y2 = rough_bbox
        
        # 扩展区域进行边缘检测
        margin = 20
        exp_x1 = max(0, x1 - margin)
        exp_y1 = max(0, y1 - margin)
        exp_x2 = min(image.shape[1], x2 + margin)
        exp_y2 = min(image.shape[0], y2 + margin)
        
        region = image[exp_y1:exp_y2, exp_x1:exp_x2]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找最大轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # 转换回图像坐标
            return (
                exp_x1 + x,
                exp_y1 + y,
                exp_x1 + x + w,
                exp_y1 + y + h
            )
        
        return rough_bbox
    
    def _create_protection_mask(self, image_shape: Tuple, 
                               regions: List[Tuple]) -> np.ndarray:
        """创建保护蒙版"""
        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for region in regions:
            x1, y1, x2, y2 = region
            
            # 添加保护边距
            x1 = max(0, x1 - self.margin)
            y1 = max(0, y1 - self.margin)
            x2 = min(width, x2 + self.margin)
            y2 = min(height, y2 + self.margin)
            
            # 在蒙版上标记保护区域
            mask[y1:y2, x1:x2] = 255
        
        return mask
    
    def visualize_detections(self, image: np.ndarray, 
                           regions: List[Tuple]) -> np.ndarray:
        """可视化检测结果"""
        vis_image = image.copy()
        
        for i, region in enumerate(regions):
            x1, y1, x2, y2 = region
            
            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # 添加标签
            label = f"Design Pack {i+1}"
            cv2.putText(vis_image, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_image
