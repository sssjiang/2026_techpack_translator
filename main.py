"""
Tech Pack Translator - 命令行入口
"""

import argparse
import sys
from loguru import logger
from pathlib import Path

from src.pipeline import TechPackTranslator


def setup_logging(log_level: str = "INFO"):
    """配置日志"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level
    )
    
    # 也写入文件
    logger.add(
        "logs/translator.log",
        rotation="10 MB",
        retention="7 days",
        level=log_level
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Tech Pack Translator - 服装技术包图像翻译系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 翻译单个文件
  python main.py input.png output.png
  
  # 翻译并保存中间结果（用于调试）
  python main.py input.png output.png --debug
  
  # 批量翻译目录
  python main.py --batch input_dir/ output_dir/
  
  # 使用自定义配置
  python main.py input.png output.png --config my_config.yaml
  
  # 指定目标语言
  python main.py input.png output.png --target-lang zh
        """
    )
    
    # 位置参数
    parser.add_argument('input', nargs='?', help='输入图像文件或目录')
    parser.add_argument('output', nargs='?', help='输出图像文件或目录')
    
    # 可选参数
    parser.add_argument('--batch', action='store_true',
                       help='批量处理模式')
    parser.add_argument('--config', '-c', default='config/config.yaml',
                       help='配置文件路径 (默认: config/config.yaml)')
    parser.add_argument('--target-lang', '-t', default='zh',
                       help='目标语言 (默认: zh)')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='保存中间结果用于调试')
    parser.add_argument('--log-level', '-l', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别 (默认: INFO)')
    parser.add_argument('--version', '-v', action='version',
                       version='Tech Pack Translator v1.0.0')
    
    args = parser.parse_args()
    
    # 配置日志
    setup_logging(args.log_level)
    
    # 检查输入输出
    if not args.input or not args.output:
        parser.print_help()
        sys.exit(1)
    
    # 检查输入路径
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path not found: {args.input}")
        sys.exit(1)
    
    # 初始化翻译器
    try:
        translator = TechPackTranslator(config_path=args.config)
    except Exception as e:
        logger.error(f"Failed to initialize translator: {e}")
        sys.exit(1)
    
    # 执行翻译
    try:
        if args.batch:
            # 批量模式
            logger.info("Running in batch mode")
            stats = translator.translate_batch(args.input, args.output)
            
            if stats['failed'] > 0:
                sys.exit(1)
        else:
            # 单文件模式
            logger.info("Running in single file mode")
            stats = translator.translate_image(
                args.input,
                args.output,
                save_intermediate=args.debug
            )
            
            if stats['status'] != 'success':
                sys.exit(1)
        
        logger.info("Translation completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("Translation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Translation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
