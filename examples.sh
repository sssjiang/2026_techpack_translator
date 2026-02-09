#!/bin/bash
# Tech Pack Translator - 示例脚本

echo "Tech Pack Translator - 示例演示"
echo "================================"
echo ""

# 检查是否有示例文件
if [ ! -f "input/sample.png" ]; then
    echo "错误: 未找到示例文件 input/sample.png"
    echo "请将您的技术包图像放入 input/ 目录"
    exit 1
fi

echo "示例1: 单文件翻译"
echo "--------------------------------"
python main.py input/sample.png output/sample_zh.png

echo ""
echo "示例2: 调试模式（生成中间文件）"
echo "--------------------------------"
python main.py --debug input/sample.png output/sample_debug.png

echo ""
echo "生成的调试文件："
ls -lh debug_*.png 2>/dev/null || echo "  无调试文件"

echo ""
echo "示例3: 批量翻译"
echo "--------------------------------"
echo "翻译 input/ 目录下的所有图像..."
python main.py --batch input/ output/

echo ""
echo "完成！查看结果："
ls -lh output/

echo ""
echo "示例4: 查看对比图"
echo "--------------------------------"
# 生成对比图（如果配置启用）
if [ -f "output/sample_zh_comparison.png" ]; then
    echo "对比图已生成: output/sample_zh_comparison.png"
else
    echo "提示: 在 config/config.yaml 中启用 generate_preview 可生成对比图"
fi

echo ""
echo "所有示例完成！"
