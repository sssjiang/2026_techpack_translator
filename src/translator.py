"""
翻译引擎模块
支持多种翻译引擎和专业术语库
"""

import json
import re
from typing import Dict, List, Optional
from loguru import logger
import os


class Translator:
    """翻译引擎"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.engine_type = self.config.get('engine', 'google')
        self.source_lang = self.config.get('source_lang', 'en')
        self.target_lang = self.config.get('target_lang', 'zh')
        self.use_cache = self.config.get('use_cache', True)
        
        # 加载专业术语库
        terminology_file = self.config.get('terminology_file', 
                                          'config/terminology.json')
        self.terminology = self._load_terminology(terminology_file)
        
        # 翻译缓存
        self.cache = {}
        
        # 初始化翻译引擎
        self.translator = self._init_translator()
        
    def _load_terminology(self, filepath: str) -> Dict:
        """加载专业术语库"""
        if not os.path.exists(filepath):
            logger.warning(f"Terminology file not found: {filepath}")
            return {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                terminology = json.load(f)
            
            # 扁平化术语库
            flat_terms = {}
            for category, terms in terminology.items():
                if isinstance(terms, dict):
                    flat_terms.update(terms)
                elif isinstance(terms, list):
                    # do_not_translate列表
                    for term in terms:
                        flat_terms[term] = term  # 保持不变
            
            logger.info(f"Loaded {len(flat_terms)} terminology entries")
            return flat_terms
            
        except Exception as e:
            logger.error(f"Failed to load terminology: {e}")
            return {}
    
    def _init_translator(self):
        """初始化翻译引擎"""
        logger.info(f"Initializing translator: {self.engine_type}")
        
        if self.engine_type == 'google':
            return self._init_google_translator()
        elif self.engine_type == 'deepl':
            return self._init_deepl_translator()
        elif self.engine_type == 'local':
            return self._init_local_translator()
        else:
            logger.warning(f"Unknown translator: {self.engine_type}, using Google")
            return self._init_google_translator()
    
    def _init_google_translator(self):
        """初始化Google翻译"""
        try:
            from googletrans import Translator
            translator = Translator()
            logger.info("Google Translator initialized")
            return translator
        except ImportError:
            logger.warning("googletrans not installed, trying deep-translator")
            try:
                from deep_translator import GoogleTranslator
                translator = GoogleTranslator(
                    source=self.source_lang,
                    target=self.target_lang
                )
                logger.info("Deep Translator (Google) initialized")
                return translator
            except ImportError:
                logger.error("No translation library available")
                return None
    
    def _init_deepl_translator(self):
        """初始化DeepL翻译"""
        try:
            from deep_translator import DeeplTranslator
            
            api_key = self.config.get('api_key') or os.getenv('DEEPL_API_KEY')
            if not api_key:
                logger.error("DeepL API key not found")
                return None
            
            translator = DeeplTranslator(
                api_key=api_key,
                source=self.source_lang,
                target=self.target_lang
            )
            
            logger.info("DeepL Translator initialized")
            return translator
            
        except ImportError:
            logger.error("deep-translator not installed")
            return None
    
    def _init_local_translator(self):
        """初始化本地翻译模型"""
        try:
            from transformers import MarianMTModel, MarianTokenizer
            
            # 使用Marian模型
            model_name = f'Helsinki-NLP/opus-mt-{self.source_lang}-{self.target_lang}'
            
            logger.info(f"Loading local model: {model_name}")
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            
            logger.info("Local translator initialized")
            return {'tokenizer': tokenizer, 'model': model}
            
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            return None
    
    def translate(self, text: str, context: Optional[str] = None) -> Dict:
        """
        翻译文本
        
        Args:
            text: 要翻译的文本
            context: 上下文信息（例如：表格列名）
            
        Returns:
            {
                'original': 原文,
                'translated': 译文,
                'confidence': 置信度,
                'method': 翻译方法
            }
        """
        # 清理文本
        text = text.strip()
        
        if not text:
            return {
                'original': text,
                'translated': text,
                'confidence': 1.0,
                'method': 'empty'
            }
        
        # 检查缓存
        if self.use_cache and text in self.cache:
            logger.debug(f"Cache hit: {text}")
            return self.cache[text]
        
        # 1. 检查是否应该保持不变（数字、代码、品牌名等）
        if self._should_not_translate(text):
            result = {
                'original': text,
                'translated': text,
                'confidence': 1.0,
                'method': 'no_translation_needed'
            }
            self.cache[text] = result
            return result
        
        # 2. 检查术语库
        terminology_result = self._check_terminology(text)
        if terminology_result:
            result = {
                'original': text,
                'translated': terminology_result,
                'confidence': 1.0,
                'method': 'terminology'
            }
            self.cache[text] = result
            return result
        
        # 3. 使用翻译引擎
        translated = self._translate_with_engine(text)
        
        # 4. 后处理
        translated = self._post_process(text, translated)
        
        result = {
            'original': text,
            'translated': translated,
            'confidence': 0.85,  # 机器翻译置信度
            'method': self.engine_type
        }
        
        self.cache[text] = result
        return result
    
    def _should_not_translate(self, text: str) -> bool:
        """判断是否应该翻译"""
        # 纯数字
        if re.match(r'^[\d\s.,%-]+$', text):
            return True
        
        # 代码/编号（如 YKK 316, CB, DTM）
        if re.match(r'^[A-Z]{2,}\s*\d*$', text):
            return True
        
        # N/A
        if text.upper() in ['N/A', 'TBD', 'TBA']:
            return True
        
        # 单位符号
        if text in ['"', '%', '°', '±']:
            return True
        
        return False
    
    def _check_terminology(self, text: str) -> Optional[str]:
        """检查专业术语库"""
        # 精确匹配
        if text in self.terminology:
            return self.terminology[text]
        
        # 不区分大小写匹配
        text_lower = text.lower()
        for term, translation in self.terminology.items():
            if term.lower() == text_lower:
                return translation
        
        # 部分匹配（包含术语）
        for term, translation in self.terminology.items():
            if term.lower() in text_lower:
                # 替换术语部分
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                result = pattern.sub(translation, text)
                if result != text:
                    return result
        
        return None
    
    def _translate_with_engine(self, text: str) -> str:
        """使用翻译引擎翻译"""
        if self.translator is None:
            logger.warning("Translator not available, returning original text")
            return text
        
        try:
            if self.engine_type == 'google':
                if hasattr(self.translator, 'translate'):
                    # googletrans
                    result = self.translator.translate(text, 
                                                      src=self.source_lang,
                                                      dest=self.target_lang)
                    return result.text
                else:
                    # deep_translator
                    return self.translator.translate(text)
            
            elif self.engine_type == 'deepl':
                return self.translator.translate(text)
            
            elif self.engine_type == 'local':
                tokenizer = self.translator['tokenizer']
                model = self.translator['model']
                
                # 编码
                inputs = tokenizer(text, return_tensors="pt", padding=True)
                
                # 翻译
                translated = model.generate(**inputs)
                
                # 解码
                result = tokenizer.decode(translated[0], skip_special_tokens=True)
                return result
            
            else:
                return text
                
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text
    
    def _post_process(self, original: str, translated: str) -> str:
        """后处理翻译结果"""
        # 保留原文的格式（冒号、括号等）
        if ':' in original and ':' not in translated and '：' not in translated:
            # 尝试在合适的位置添加冒号
            parts = original.split(':', 1)
            if len(parts) == 2:
                trans_parts = translated.split(' ', 1)
                if len(trans_parts) == 2:
                    translated = trans_parts[0] + '：' + trans_parts[1]
        
        # 保留百分号
        if '%' in original and '%' not in translated:
            translated = translated + '%'
        
        # 移除多余空格
        translated = ' '.join(translated.split())
        
        return translated
    
    def translate_batch(self, texts: List[str]) -> List[Dict]:
        """批量翻译"""
        results = []
        
        for text in texts:
            result = self.translate(text)
            results.append(result)
        
        return results
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        logger.info("Translation cache cleared")
