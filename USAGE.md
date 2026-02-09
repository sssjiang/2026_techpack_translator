# Tech Pack Translator - 使用指南

## 目录
- [系统要求](#系统要求)
- [安装](#安装)
- [快速开始](#快速开始)
- [详细使用](#详细使用)
- [配置说明](#配置说明)
- [常见问题](#常见问题)
- [API使用](#api使用)

## 系统要求

### 硬件要求
- CPU: 4核或以上（推荐）
- 内存: 8GB或以上
- 硬盘: 5GB可用空间（含模型文件）
- GPU: 可选，可加速OCR和翻译

### 软件要求
- Python 3.8-3.10
- Docker (可选，推荐)
- Git

## 安装

### 方式1: Docker安装（推荐）

```bash
# 1. 克隆项目
git clone https://github.com/your-repo/techpack-translator.git
cd techpack-translator

# 2. 构建Docker镜像
docker-compose build

# 3. 完成！可以使用了
docker-compose run --rm translator --help
```

### 方式2: 本地安装

```bash
# 1. 克隆项目
git clone https://github.com/your-repo/techpack-translator.git
cd techpack-translator

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载中文字体（如果系统没有）
mkdir -p fonts
# 将SimHei.ttf或其他中文字体放到fonts目录

# 5. 测试安装
python main.py --help
```

## 快速开始

### 1. 翻译单个文件

```bash
# Docker方式
docker-compose run --rm translator input/sample.png output/sample_zh.png

# 本地方式
python main.py input/sample.png output/sample_zh.png
```

### 2. 批量翻译

```bash
# 准备输入目录
mkdir -p input output
# 将所有技术包图像放入input目录

# Docker方式
docker-compose run --rm translator --batch input/ output/

# 本地方式
python main.py --batch input/ output/
```

### 3. 启动API服务

```bash
# Docker方式
docker-compose up api

# 本地方式
python api.py
```

访问 http://localhost:8000/docs 查看API文档

## 详细使用

### 命令行参数

```
python main.py [OPTIONS] INPUT OUTPUT

位置参数:
  INPUT                 输入图像文件或目录
  OUTPUT                输出图像文件或目录

可选参数:
  --batch               批量处理模式
  --config PATH         配置文件路径 (默认: config/config.yaml)
  --target-lang LANG    目标语言 (默认: zh)
  --debug               保存中间结果用于调试
  --log-level LEVEL     日志级别 (DEBUG|INFO|WARNING|ERROR)
  --version             显示版本信息
  --help                显示帮助信息
```

### 使用示例

#### 示例1: 基本翻译
```bash
python main.py techpack.png techpack_zh.png
```

#### 示例2: 调试模式
```bash
# 会生成debug_*.png中间文件
python main.py --debug techpack.png techpack_zh.png
```

生成的调试文件：
- `debug_enhanced.png` - 增强后的图像
- `debug_ocr.png` - OCR识别结果可视化
- `debug_detection.png` - 设计图案检测结果
- `debug_mask.png` - 保护蒙版

#### 示例3: 使用自定义配置
```bash
python main.py --config my_config.yaml input.png output.png
```

#### 示例4: 批量翻译并查看详细日志
```bash
python main.py --batch --log-level DEBUG input/ output/
```

## 配置说明

### 配置文件结构

`config/config.yaml`:

```yaml
# OCR配置
ocr:
  engine: paddleocr        # OCR引擎: paddleocr, tesseract, easyocr
  languages: [en, ch]      # 支持的语言
  use_gpu: false           # 是否使用GPU
  det_db_thresh: 0.3       # 文本检测阈值
  det_db_box_thresh: 0.6   # 边界框阈值

# 翻译配置
translation:
  engine: google           # 翻译引擎: google, deepl, local
  source_lang: en          # 源语言
  target_lang: zh          # 目标语言
  api_key: null            # API密钥（如需要）
  use_cache: true          # 使用翻译缓存
  terminology_file: config/terminology.json  # 术语库文件

# 设计图案检测配置
detection:
  methods:
    - text_annotation      # 通过文本标注检测
    - visual_features      # 通过视觉特征检测
  keywords:
    - "design pack image"
    - "design pack"
  confidence_threshold: 0.7
  protection_margin: 10    # 保护边距（像素）

# 渲染配置
rendering:
  fonts:
    default: SimHei        # 默认中文字体
    fallback: Arial        # 备用字体
  font_size:
    auto: true             # 自动检测字号
    min: 8
    max: 72
  preserve_layout: true    # 保持布局
  auto_resize: true        # 自动调整文字大小
  inpaint_method: telea    # 背景修复方法

# 输出配置
output:
  format: png              # 输出格式
  quality: 95              # 图像质量
  dpi: 300                 # DPI
  generate_preview: true   # 生成预览对比图
  comparison_mode: side_by_side  # 对比模式
```

### 专业术语库

编辑 `config/terminology.json` 添加行业术语：

```json
{
  "fabric_materials": {
    "Cotton": "棉",
    "Polyester": "聚酯纤维"
  },
  "garment_parts": {
    "Main Fabric": "主面料",
    "Lining": "里料"
  },
  "do_not_translate": [
    "YKK",
    "DTM"
  ]
}
```

## API使用

### 启动API服务

```bash
docker-compose up api
```

或

```bash
python api.py
```

### API端点

#### 1. 上传并翻译
```bash
curl -X POST \
  -F "file=@input.png" \
  -F "target_lang=zh" \
  http://localhost:8000/translate
```

响应：
```json
{
  "status": "success",
  "message": "Translation completed successfully",
  "output_file": "output/translated_input.png",
  "stats": {
    "translated_count": 156,
    "avg_confidence": 0.89
  }
}
```

#### 2. 翻译并下载
```bash
curl -X POST \
  -F "file=@input.png" \
  http://localhost:8000/translate/download \
  -o output.png
```

#### 3. 获取已翻译文件
```bash
curl http://localhost:8000/outputs/translated_input.png -o result.png
```

#### 4. 健康检查
```bash
curl http://localhost:8000/health
```

### Python客户端示例

```python
import requests

# 翻译文件
with open('input.png', 'rb') as f:
    files = {'file': f}
    response = requests.post(
        'http://localhost:8000/translate',
        files=files,
        data={'target_lang': 'zh'}
    )
    
result = response.json()
print(f"Status: {result['status']}")
print(f"Output: {result['output_file']}")
```

## 常见问题

### Q1: OCR识别率低怎么办？

**解决方案：**
1. 提高输入图像分辨率（至少300 DPI）
2. 确保图像清晰，文字对比度高
3. 尝试不同的OCR引擎（paddleocr, tesseract, easyocr）
4. 在config.yaml中调整OCR参数

### Q2: 设计图案被误翻译了？

**解决方案：**
1. 确保在技术包上明确标注了"Design Pack Image"
2. 使用 `--debug` 模式查看检测结果
3. 调整 `config.yaml` 中的检测参数
4. 手动在术语库中添加保护规则

### Q3: 翻译结果不准确？

**解决方案：**
1. 在 `config/terminology.json` 中添加行业专业术语
2. 更换翻译引擎（google -> deepl）
3. 使用API密钥获得更好的翻译质量

### Q4: 字体显示问题？

**解决方案：**
1. 安装中文字体到系统或放入 `fonts/` 目录
2. 在 `config.yaml` 中指定正确的字体名称
3. Docker用户：确保镜像包含中文字体

### Q5: 处理速度慢？

**解决方案：**
1. 启用GPU加速：设置 `ocr.use_gpu: true`
2. 减小图像分辨率
3. 使用批量处理模式
4. 启用翻译缓存

### Q6: Docker容器无法访问文件？

**解决方案：**
```bash
# 确保文件在挂载的目录中
cp your_file.png input/
docker-compose run --rm translator input/your_file.png output/result.png
```

## 性能优化建议

1. **批量处理**: 使用 `--batch` 模式处理多个文件
2. **GPU加速**: 在有GPU的机器上启用GPU支持
3. **缓存**: 启用翻译缓存减少重复翻译
4. **并行处理**: 配置 `performance.max_workers`

## 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 许可证

MIT License - 详见LICENSE文件

## 联系方式

- 项目主页: https://github.com/your-repo/techpack-translator
- 问题反馈: https://github.com/your-repo/techpack-translator/issues
