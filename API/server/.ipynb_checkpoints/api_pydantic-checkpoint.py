from pydantic import BaseModel
from typing import List, Dict, Any, Optional

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