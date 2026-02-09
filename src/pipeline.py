"""
主流程控制模块
整合所有组件完成翻译流程
"""

import cv2
import numpy as np
import yaml
from typing import Dict, Optional, Tuple
from loguru import logger
import os
import time

from .preprocessor import ImagePreprocessor
from .design_detector import DesignPackDetector
from .ocr_engine import OCREngine
from .translator import Translator
from .renderer import ImageRenderer


class TechPackTranslator:
    """技术包翻译器主类"""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        初始化翻译器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化各模块
        logger.info("Initializing Tech Pack Translator...")
        
        self.preprocessor = ImagePreprocessor(
            self.config.get('preprocessing', {})
        )
        
        self.design_detector = DesignPackDetector(
            self.config.get('detection', {})
        )
        
        self.ocr_engine = OCREngine(
            self.config.get('ocr', {})
        )
        
        self.translator = Translator(
            self.config.get('translation', {})
        )
        
        self.renderer = ImageRenderer(
            self.config.get('rendering', {})
        )
        
        logger.info("Tech Pack Translator initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Config loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def translate_image(self, 
                       input_path: str,
                       output_path: str,
                       save_intermediate: bool = False) -> Dict:
        """
        翻译技术包图像
        
        Args:
            input_path: 输入图像路径
            output_path: 输出图像路径
            save_intermediate: 是否保存中间结果
            
        Returns:
            处理结果统计信息
        """
        start_time = time.time()
        logger.info(f"Starting translation: {input_path} -> {output_path}")
        
        stats = {
            'input_file': input_path,
            'output_file': output_path,
            'status': 'processing'
        }
        
        try:
            # 步骤1: 图像预处理
            logger.info("Step 1/6: Image preprocessing...")
            enhanced_image, original_image = self.preprocessor.process(input_path)
            stats['image_size'] = original_image.shape[:2]
            
            if save_intermediate:
                cv2.imwrite('debug_enhanced.png', enhanced_image)
            
            # 步骤2: OCR初步识别（用于检测标注）
            logger.info("Step 2/6: Initial OCR for annotation detection...")
            initial_ocr_results = self.ocr_engine.recognize(enhanced_image)
            stats['initial_text_regions'] = len(initial_ocr_results)
            
            if save_intermediate:
                vis_ocr = self.ocr_engine.visualize_ocr_results(
                    enhanced_image.copy(), initial_ocr_results
                )
                cv2.imwrite('debug_ocr.png', vis_ocr)
            
            # 步骤3: 检测设计包图像区域
            logger.info("Step 3/6: Detecting design pack image regions...")
            design_regions, protection_mask = self.design_detector.detect(
                enhanced_image, initial_ocr_results
            )
            stats['design_pack_regions'] = len(design_regions)
            
            if save_intermediate:
                vis_detection = self.design_detector.visualize_detections(
                    enhanced_image.copy(), design_regions
                )
                cv2.imwrite('debug_detection.png', vis_detection)
                cv2.imwrite('debug_mask.png', protection_mask)
            
            # 步骤4: 完整OCR识别（排除保护区域）
            logger.info("Step 4/6: Full OCR recognition...")
            ocr_results = self.ocr_engine.recognize(
                enhanced_image, 
                protection_mask
            )
            stats['text_regions_to_translate'] = len(ocr_results)
            
            # 步骤5: 翻译文字
            logger.info("Step 5/6: Translating text...")
            translation_results = []
            
            for ocr_result in ocr_results:
                text = ocr_result['text']
                trans_result = self.translator.translate(text)
                translation_results.append(trans_result)
            
            # 统计翻译结果
            stats['translated_count'] = sum(
                1 for r in translation_results 
                if r['original'] != r['translated']
            )
            stats['avg_confidence'] = sum(
                r['confidence'] for r in translation_results
            ) / len(translation_results) if translation_results else 0
            
            # 步骤6: 渲染翻译结果
            logger.info("Step 6/6: Rendering translated image...")
            output_image = self.renderer.render(
                original_image,
                ocr_results,
                translation_results,
                protection_mask
            )
            
            # 保存输出
            cv2.imwrite(output_path, output_image)
            logger.info(f"Output saved to: {output_path}")
            
            # 生成对比图（如果配置启用）
            if self.config.get('output', {}).get('generate_preview', False):
                comparison_path = output_path.replace('.', '_comparison.')
                comparison = self.renderer.create_comparison(
                    original_image, output_image,
                    mode=self.config.get('output', {}).get('comparison_mode', 'side_by_side')
                )
                cv2.imwrite(comparison_path, comparison)
                logger.info(f"Comparison saved to: {comparison_path}")
            
            # 完成
            elapsed_time = time.time() - start_time
            stats['status'] = 'success'
            stats['elapsed_time'] = f"{elapsed_time:.2f}s"
            
            logger.info(f"Translation completed in {elapsed_time:.2f}s")
            self._print_stats(stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Translation failed: {e}", exc_info=True)
            stats['status'] = 'failed'
            stats['error'] = str(e)
            return stats
    
    def translate_batch(self, input_dir: str, output_dir: str) -> Dict:
        """
        批量翻译目录中的所有图像
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            
        Returns:
            批量处理统计
        """
        logger.info(f"Batch translation: {input_dir} -> {output_dir}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 支持的图像格式
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        
        # 查找所有图像文件
        image_files = []
        for filename in os.listdir(input_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                image_files.append(filename)
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # 处理每个文件
        results = []
        for i, filename in enumerate(image_files):
            logger.info(f"\nProcessing {i+1}/{len(image_files)}: {filename}")
            
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            result = self.translate_image(input_path, output_path)
            results.append(result)
        
        # 统计
        batch_stats = {
            'total_files': len(image_files),
            'successful': sum(1 for r in results if r['status'] == 'success'),
            'failed': sum(1 for r in results if r['status'] == 'failed'),
            'results': results
        }
        
        logger.info(f"\nBatch translation completed:")
        logger.info(f"  Total: {batch_stats['total_files']}")
        logger.info(f"  Success: {batch_stats['successful']}")
        logger.info(f"  Failed: {batch_stats['failed']}")
        
        return batch_stats
    
    def _print_stats(self, stats: Dict):
        """打印统计信息"""
        logger.info("\n" + "="*60)
        logger.info("Translation Statistics:")
        logger.info("="*60)
        logger.info(f"Input: {stats['input_file']}")
        logger.info(f"Output: {stats['output_file']}")
        logger.info(f"Image Size: {stats.get('image_size', 'N/A')}")
        logger.info(f"Design Pack Regions: {stats.get('design_pack_regions', 0)}")
        logger.info(f"Total Text Regions: {stats.get('text_regions_to_translate', 0)}")
        logger.info(f"Translated: {stats.get('translated_count', 0)}")
        logger.info(f"Average Confidence: {stats.get('avg_confidence', 0):.2%}")
        logger.info(f"Time Elapsed: {stats.get('elapsed_time', 'N/A')}")
        logger.info(f"Status: {stats['status'].upper()}")
        logger.info("="*60 + "\n")
