FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    fonts-noto-cjk \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 安装Tesseract OCR（可选）
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements
COPY requirements.txt .

# 安装Python依赖（使用国内镜像源，增加超时时间）
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    --timeout 300 \
    && pip install --no-cache-dir --no-deps craft-text-detector \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn

# 下载PaddleOCR模型（如果使用PaddleOCR）
RUN PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    python -c "from paddleocr import PaddleOCR; ocr = PaddleOCR(use_textline_orientation=True, lang='en')"

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p logs fonts

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 默认命令
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
