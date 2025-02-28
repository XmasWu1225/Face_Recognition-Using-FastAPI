from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from fastapi import UploadFile
from pydantic import field_validator
import cv2
import numpy as np
from server.logger import logger

class BaseResponse(BaseModel):
    success: bool = True
    message: str = "Operation successful"
    data: Optional[Any] = None

class ErrorResponse(BaseResponse):
    success: bool = False
    message: str = "An error occurred"

class ListResponse(BaseModel):
    items: List[Dict[str, Any]]

class TaskResponse(BaseResponse):
    task_id: str
    status: str

class FileResponse(BaseModel):
    file_name: str
    file_size: int
    file_path: str

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]

class PaginatedUsersResponse(BaseModel): 
    results: List[Dict[str, Any]]
    total: int

class ImageUpload(BaseModel):
    file: UploadFile

    @field_validator("file")
    def validate_image(cls, file: UploadFile):
        # 检查文件类型
        allowed_types = ["image/jpeg", "image/png"]  # 支持 .jpg 和 .png
        if file.content_type not in allowed_types:
            logger.warning(f"Invalid file type: {file.content_type}")
            raise ValueError(f"Only JPEG or PNG images are accepted, got {file.content_type}")
        
        # 读取并解码文件
        contents = file.file.read()  # 注意：这里是同步读取，需调整为异步
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            logger.error("Failed to decode image")
            raise ValueError("Invalid image content")
        
        # 转换为 RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 检查尺寸
        if img.shape != (112, 112, 3):
            logger.warning(f"Invalid image shape: {img.shape}")
            raise ValueError("Image must be 112x112 pixels with 3 channels")
        
        # 将图像数据附加到对象，便于后续使用
        cls.img = img
        file.file.seek(0)  # 重置文件指针
        return file

    class Config:
        arbitrary_types_allowed = True  # 允许 UploadFile 类型