"""
Tech Pack Translator - FastAPI Web Service
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import tempfile
import shutil
from pathlib import Path
from loguru import logger

from src.pipeline import TechPackTranslator


# 初始化FastAPI应用
app = FastAPI(
    title="Tech Pack Translator API",
    description="服装技术包图像翻译API服务",
    version="1.0.0"
)

# 添加CORS支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化翻译器
translator = None


class TranslationRequest(BaseModel):
    """翻译请求模型"""
    target_lang: str = "zh"
    save_intermediate: bool = False


class TranslationResponse(BaseModel):
    """翻译响应模型"""
    status: str
    message: str
    output_file: Optional[str] = None
    stats: Optional[dict] = None


@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    global translator
    logger.info("Initializing translator...")
    translator = TechPackTranslator(config_path='config/config.yaml')
    logger.info("API service ready")


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "Tech Pack Translator API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "translator_ready": translator is not None
    }


@app.post("/translate", response_model=TranslationResponse)
async def translate_image(
    file: UploadFile = File(...),
    target_lang: str = "zh",
    save_intermediate: bool = False
):
    """
    翻译技术包图像
    
    Args:
        file: 上传的图像文件
        target_lang: 目标语言（默认：zh）
        save_intermediate: 是否保存中间结果
        
    Returns:
        翻译结果
    """
    if translator is None:
        raise HTTPException(status_code=503, detail="Translator not initialized")
    
    # 验证文件类型
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
        )
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 保存上传的文件
        input_path = os.path.join(temp_dir, f"input{file_ext}")
        output_path = os.path.join(temp_dir, f"output{file_ext}")
        
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing uploaded file: {file.filename}")
        
        try:
            # 执行翻译
            stats = translator.translate_image(
                input_path,
                output_path,
                save_intermediate=save_intermediate
            )
            
            if stats['status'] == 'success':
                # 保存到output目录
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                
                final_output = output_dir / f"translated_{file.filename}"
                shutil.copy(output_path, final_output)
                
                return TranslationResponse(
                    status="success",
                    message="Translation completed successfully",
                    output_file=str(final_output),
                    stats=stats
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Translation failed: {stats.get('error', 'Unknown error')}"
                )
                
        except Exception as e:
            logger.error(f"Translation error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate/download")
async def translate_and_download(
    file: UploadFile = File(...),
    target_lang: str = "zh"
):
    """
    翻译并直接下载结果
    """
    if translator is None:
        raise HTTPException(status_code=503, detail="Translator not initialized")
    
    # 验证文件类型
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}"
        )
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, f"input{file_ext}")
        output_path = os.path.join(temp_dir, f"output{file_ext}")
        
        # 保存上传的文件
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            # 执行翻译
            stats = translator.translate_image(input_path, output_path)
            
            if stats['status'] == 'success':
                # 返回文件
                return FileResponse(
                    output_path,
                    media_type=f"image/{file_ext[1:]}",
                    filename=f"translated_{file.filename}"
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Translation failed: {stats.get('error', 'Unknown error')}"
                )
                
        except Exception as e:
            logger.error(f"Translation error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/outputs/{filename}")
async def get_output_file(filename: str):
    """获取输出文件"""
    file_path = Path("output") / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)


@app.get("/stats")
async def get_stats():
    """获取系统统计"""
    output_dir = Path("output")
    
    if not output_dir.exists():
        total_files = 0
    else:
        total_files = len(list(output_dir.glob("*")))
    
    return {
        "total_translations": total_files,
        "translator_status": "ready" if translator else "not initialized"
    }


if __name__ == "__main__":
    # 配置日志
    logger.add(
        "logs/api.log",
        rotation="10 MB",
        retention="7 days",
        level="INFO"
    )
    
    # 启动服务
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
