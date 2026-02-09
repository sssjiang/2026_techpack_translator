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

### 使用Docker

```bash
# 构建镜像
docker-compose build

# 运行翻译
docker-compose run --rm translator input/techpack_img.png output/techpack_img_zh.png
```

## 许可证

MIT License
