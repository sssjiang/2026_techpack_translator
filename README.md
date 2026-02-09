# Tech Pack 翻译系统

自动化服装技术包图像翻译系统，支持保护设计图案同时翻译所有文字信息。

## 功能特性

- ✅ 自动检测并保护"设计包图像"区域
- ✅ 智能OCR文字识别
- ✅ 多语言翻译支持（默认英文→中文）
- ✅ 保持表格结构完整
- ✅ 自适应字体渲染
- ✅ Docker容器化部署

## 项目结构

```
techpack-translator/
├── src/
│   ├── __init__.py
│   ├── preprocessor.py          # 图像预处理
│   ├── design_detector.py       # 设计图案检测
│   ├── text_extractor.py        # 文字提取
│   ├── ocr_engine.py            # OCR识别
│   ├── translator.py            # 翻译引擎
│   ├── renderer.py              # 图像重构
│   └── pipeline.py              # 主流程
├── config/
│   ├── terminology.json         # 专业术语库
│   └── config.yaml              # 配置文件
├── tests/
│   └── test_pipeline.py
├── fonts/                       # 中文字体
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── main.py                      # 入口文件
```

## 快速开始

### 使用Docker（推荐）

```bash
# 构建镜像
docker-compose build

# 运行翻译
docker-compose run translator python main.py input.png output.png

# 或使用API服务
docker-compose up
curl -X POST -F "file=@input.png" http://localhost:8000/translate -o output.png
```

### 本地安装

```bash
# 安装依赖
pip install -r requirements.txt

# 运行
python main.py input.png output.png --target-lang zh
```

## 配置说明

编辑 `config/config.yaml`:

```yaml
ocr:
  engine: paddleocr  # 可选: paddleocr, google_vision, tesseract
  languages: [en, ch]

translation:
  engine: google  # 可选: google, deepl, local
  api_key: YOUR_API_KEY
  source_lang: en
  target_lang: zh

rendering:
  font_family: SimHei
  auto_resize: true
  preserve_layout: true
```

## API文档

启动服务后访问: http://localhost:8000/docs

## 许可证

MIT License
