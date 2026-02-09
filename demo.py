#!/usr/bin/env python
"""
Tech Pack Translator - 演示脚本
展示如何使用API进行翻译
"""

from src.pipeline import TechPackTranslator
import cv2
import numpy as np
from pathlib import Path


def create_sample_techpack():
    """创建示例技术包图像（用于演示）"""
    # 创建一个简单的技术包样本
    img = np.ones((800, 1200, 3), dtype=np.uint8) * 255
    
    # 添加标题
    cv2.putText(img, "Tech Pack Sample", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    
    # 添加表格
    # 表头
    cv2.putText(img, "Description", (50, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Placement", (400, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Color", (700, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # 表格内容
    content = [
        ("Fabric: 97% Cotton, 3% Spandex", "Main Fabric", "Blue Light"),
        ("Thread: 100% Polyester", "All Seams", "DTM"),
        ("Zipper: #5 Close End", "CB", "YKK 316"),
        ("Main Label: Hemp Screen Printed", "CB", "Natural"),
    ]
    
    y = 180
    for desc, place, color in content:
        cv2.putText(img, desc, (50, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(img, place, (400, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(img, color, (700, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y += 50
    
    # 添加设计包图像标注
    cv2.putText(img, "Design pack image -->", (50, 500),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # 画一个彩色区域代表设计图案
    design_area = img[450:650, 700:1100]
    # 填充渐变色
    for i in range(design_area.shape[0]):
        for j in range(design_area.shape[1]):
            design_area[i, j] = [
                (i * 255) // design_area.shape[0],
                (j * 255) // design_area.shape[1],
                128
            ]
    
    # 画边框
    cv2.rectangle(img, (700, 450), (1100, 650), (0, 0, 0), 2)
    
    return img


def demo_basic_usage():
    """演示基本用法"""
    print("=" * 60)
    print("Tech Pack Translator - 基本用法演示")
    print("=" * 60)
    print()
    
    # 1. 创建示例图像
    print("步骤1: 创建示例技术包图像...")
    sample_img = create_sample_techpack()
    
    # 确保目录存在
    Path("input").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    
    input_path = "input/demo_sample.png"
    cv2.imwrite(input_path, sample_img)
    print(f"✓ 示例图像已保存到: {input_path}")
    print()
    
    # 2. 初始化翻译器
    print("步骤2: 初始化翻译器...")
    translator = TechPackTranslator('config/config.yaml')
    print("✓ 翻译器初始化完成")
    print()
    
    # 3. 翻译图像
    print("步骤3: 翻译图像...")
    output_path = "output/demo_translated.png"
    
    stats = translator.translate_image(
        input_path,
        output_path,
        save_intermediate=True  # 保存调试信息
    )
    
    print()
    if stats['status'] == 'success':
        print("✓ 翻译成功!")
        print(f"  输出文件: {output_path}")
        print(f"  处理时间: {stats['elapsed_time']}")
        print(f"  翻译区域: {stats.get('translated_count', 0)}/{stats.get('text_regions_to_translate', 0)}")
        print(f"  设计包区域: {stats.get('design_pack_regions', 0)}")
    else:
        print("✗ 翻译失败")
        print(f"  错误: {stats.get('error', 'Unknown error')}")
    
    print()
    print("调试文件已生成:")
    debug_files = [
        "debug_enhanced.png",
        "debug_ocr.png", 
        "debug_detection.png",
        "debug_mask.png"
    ]
    
    for f in debug_files:
        if Path(f).exists():
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} (未生成)")
    
    print()
    print("=" * 60)


def demo_api_usage():
    """演示API用法"""
    print()
    print("=" * 60)
    print("API 使用示例")
    print("=" * 60)
    print()
    
    print("Python代码示例:")
    print()
    print("""
from src.pipeline import TechPackTranslator

# 初始化
translator = TechPackTranslator()

# 翻译单个文件
stats = translator.translate_image(
    'input.png',
    'output.png'
)

# 批量翻译
batch_stats = translator.translate_batch(
    'input_dir/',
    'output_dir/'
)

print(f"成功: {batch_stats['successful']}")
print(f"失败: {batch_stats['failed']}")
    """)
    
    print()
    print("cURL API示例:")
    print()
    print("""
# 启动API服务
python api.py

# 翻译并下载
curl -X POST \\
  -F "file=@input.png" \\
  http://localhost:8000/translate/download \\
  -o output.png
    """)
    
    print()
    print("=" * 60)


def demo_configuration():
    """演示配置用法"""
    print()
    print("=" * 60)
    print("配置示例")
    print("=" * 60)
    print()
    
    print("config/config.yaml:")
    print("""
ocr:
  engine: paddleocr  # 使用PaddleOCR
  use_gpu: false     # 不使用GPU

translation:
  engine: google     # 使用Google翻译
  target_lang: zh    # 目标语言：中文
  
detection:
  keywords:
    - "design pack image"
    - "设计包图像"
  protection_margin: 10  # 保护边距10像素
    """)
    
    print()
    print("config/terminology.json:")
    print("""
{
  "fabric_materials": {
    "Cotton": "棉",
    "Polyester": "聚酯纤维",
    "Spandex": "氨纶"
  },
  "do_not_translate": [
    "YKK",
    "DTM"
  ]
}
    """)
    
    print()
    print("=" * 60)


if __name__ == '__main__':
    # 运行演示
    try:
        demo_basic_usage()
        demo_api_usage()
        demo_configuration()
        
        print()
        print("演示完成！")
        print()
        print("下一步:")
        print("1. 查看生成的示例文件")
        print("2. 尝试翻译自己的技术包图像")
        print("3. 查看 USAGE.md 了解更多用法")
        print("4. 查看 ARCHITECTURE.md 了解系统设计")
        print()
        
    except Exception as e:
        print(f"演示过程中出错: {e}")
        print("请确保已安装所有依赖: pip install -r requirements.txt")
