# Tech Pack Translator - 快速开始

## 5分钟上手指南

### 步骤1: 获取代码

```bash
git clone https://github.com/your-repo/techpack-translator.git
cd techpack-translator
```

### 步骤2: 选择安装方式

#### 方式A: Docker（推荐，最简单）

```bash
# 一条命令完成所有安装
docker-compose build

# 测试是否成功
docker-compose run --rm translator --version
```

#### 方式B: 本地安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 步骤3: 准备示例文件

```bash
# 创建输入输出目录
mkdir -p input output

# 将你的技术包图像放入input目录
cp your_techpack.png input/
```

### 步骤4: 运行翻译

#### Docker方式
```bash
docker-compose run --rm translator \
  input/your_techpack.png \
  output/translated.png
```

#### 本地方式
```bash
python main.py input/your_techpack.png output/translated.png
```

### 步骤5: 查看结果

```bash
# 输出文件在
ls -lh output/

# 如果开启了预览，还会有对比图
open output/translated_comparison.png  # Mac
# 或 xdg-open output/translated_comparison.png  # Linux
```

## 常用命令

### 翻译单个文件
```bash
python main.py input.png output.png
```

### 批量翻译
```bash
python main.py --batch input/ output/
```

### 调试模式（生成中间文件）
```bash
python main.py --debug input.png output.png
# 会生成: debug_enhanced.png, debug_ocr.png, debug_detection.png
```

### 启动API服务
```bash
# Docker
docker-compose up api

# 本地
python api.py

# 然后访问 http://localhost:8000/docs
```

### 运行演示
```bash
python demo.py
```

## 配置翻译引擎

编辑 `config/config.yaml`:

```yaml
translation:
  engine: google  # 改为 deepl 或 local
  target_lang: zh # 目标语言
```

## 添加专业术语

编辑 `config/terminology.json`:

```json
{
  "fabric_materials": {
    "你的术语": "翻译"
  }
}
```

## 下一步

- 📖 查看 [USAGE.md](USAGE.md) 了解详细用法
- 🏗️ 查看 [ARCHITECTURE.md](ARCHITECTURE.md) 了解系统架构
- 🐛 遇到问题? 查看 [常见问题](USAGE.md#常见问题)
- 💡 想贡献代码? 欢迎提交 Pull Request!

## 需要帮助？

- 📧 Email: your-email@example.com
- 🐛 问题反馈: https://github.com/your-repo/issues
- 💬 讨论: https://github.com/your-repo/discussions

## 最小示例代码

```python
from src.pipeline import TechPackTranslator

# 初始化
translator = TechPackTranslator()

# 翻译
stats = translator.translate_image(
    'input.png',
    'output.png'
)

print(f"状态: {stats['status']}")
print(f"翻译了 {stats['translated_count']} 个文本区域")
```

就这么简单！🎉
