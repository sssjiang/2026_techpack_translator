# Tech Pack 翻译系统

自动化服装技术包图像翻译系统，支持保护设计图案同时翻译所有文字信息。

## 功能特性

- ✅ 自动检测并保护"设计包图像"区域
- ✅ Qwen-OCR 文字识别与定位（阿里云百炼）
- ✅ DeepL 翻译（默认英文→中文）
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
│   ├── ocr_engine.py            # OCR（仅 Qwen-OCR）
│   ├── translator.py            # 翻译（仅 DeepL）
│   ├── renderer.py              # 图像重构
│   └── pipeline.py              # 主流程
├── config/
│   ├── terminology.json         # 专业术语库
│   ├── config.yaml              # 配置文件（本地使用）
│   ├── config example.yaml      # 配置模板
│   └── README.md                # 配置说明
├── tests/
│   ├── test_pipeline.py
│   ├── test_ocr.py
│   └── test_translator.py
├── fonts/                       # 中文字体
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── main.py                      # 入口文件
```

## 配置说明

### 1. 复制配置模板

```bash
cp "config/config example.yaml" config/config.yaml
```

### 2. 填写 API Key

打开 `config/config.yaml`，找到以下两处并填写你的 API Key：

**Qwen-OCR（阿里云百炼）：**
```yaml
ocr:
  api_key: XXX  # 改为你的 DASHSCOPE_API_KEY，例如：sk-xxxxxxxxxxxxx
```

**DeepL 翻译：**
```yaml
translation:
  api_key: XXX  # 改为你的 DeepL API Key，例如：xxxxx:fx（免费版以 :fx 结尾）
```

> 💡 **获取 API Key：**
> - Qwen-OCR: https://help.aliyun.com/zh/model-studio/get-api-key
> - DeepL: https://www.deepl.com/pro-api

## 快速开始

### 使用Docker

```bash
# 构建镜像
docker-compose build

# 运行翻译
docker-compose run --rm translator input/techpack_img.png output/techpack_img_zh.png
```

### 本地运行（可选）

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 main.py input/techpack_img.png output/techpack_img_zh.png
```

### 运行单元测试

```bash
python -m unittest discover -s tests -v
```

## 许可证

MIT License
