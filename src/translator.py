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
        # 目前仅保留 DeepL 翻译引擎
        self.engine_type = 'deepl'
        self.source_lang = self._normalize_lang_code(self.config.get('source_lang', 'en'))
        self.target_lang = self._normalize_lang_code(self.config.get('target_lang', 'zh'))
        self.use_cache = self.config.get('use_cache', True)
        
        # 加载专业术语库
        terminology_file = self.config.get('terminology_file', 
                                          'config/terminology.json')
        self.terminology = self._load_terminology(terminology_file)
        
        # 翻译缓存
        self.cache = {}
        
        # 初始化翻译引擎
        self.translator = self._init_translator()
        
    def _normalize_lang_code(self, code: str) -> str:
        """规范语言代码（根据引擎不同会有不同处理）"""
        if not code:
            return 'en'
        c = code.strip().lower().replace('_', '-')
        # 常见简写/变体 -> 标准码
        # 注意：DeepL 使用 'ZH'，Google/deep-translator 使用 'zh-CN'
        mapping = {
            'zh': 'zh-CN', 'zh-cn': 'zh-CN', 'chinese': 'zh-CN',
            'zh-tw': 'zh-TW', 'zh-hk': 'zh-TW', 'zh-hant': 'zh-TW',
        }
        return mapping.get(c, code)

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
        """初始化翻译引擎（目前仅 DeepL）"""
        logger.info("Initializing translator: deepl")
        return self._init_deepl_translator()

    def _init_deepl_translator(self):
        """初始化DeepL翻译（免费版，直接使用REST API）"""
        try:
            import requests
            
            api_key = self.config.get('api_key') or os.getenv('DEEPL_API_KEY')
            if not api_key:
                logger.error("DeepL API key not found")
                logger.info("Get your free API key from: https://www.deepl.com/pro-api")
                return None
            
            # DeepL 使用小写语言代码：zh 而不是 zh-CN（DeepL 不区分简繁体）
            source = 'ZH' if self.source_lang.startswith('zh') else self.source_lang.upper()
            target = 'ZH' if self.target_lang.startswith('zh') else self.target_lang.upper()
            
            # 免费版 API 端点
            translator_config = {
                'api_key': api_key,
                'api_url': 'https://api-free.deepl.com/v2/translate',  # 免费版端点
                'source': source,
                'target': target
            }
            
            logger.info(f"DeepL Translator initialized (Free API: api-free.deepl.com, source={source}, target={target})")
            return translator_config
            
        except ImportError:
            logger.error("requests not installed. Install with: pip install requests")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize DeepL Translator: {e}")
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

        # 如果使用 DeepL，引擎优先：不走术语库 / 不翻译规则，所有内容都交给 DeepL
        if self.engine_type == 'deepl':
            translated = self._translate_with_engine(text)
        else:
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
            # 目前仅支持 Deepl：engine_type 恒为 'deepl'
            import requests
            config = self.translator  # 包含 api_key, api_url, source, target

            headers = {
                'Authorization': f'DeepL-Auth-Key {config["api_key"]}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }

            data = {
                'text': text,
                'source_lang': config['source'],
                'target_lang': config['target']
            }

            response = requests.post(config['api_url'], headers=headers, data=data, timeout=10)
            response.raise_for_status()
            result = response.json()

            if 'translations' in result and len(result['translations']) > 0:
                return result['translations'][0]['text']
            else:
                logger.error(f"Unexpected API response: {result}")
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
