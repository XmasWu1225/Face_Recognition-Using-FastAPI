# server/api_routes/face_routes.py
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
from pydantic import BaseModel, field_validator
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import os
import time
import uuid
from server.models import scrfd, face_system, align_face
from server.logger import logger
from utils.helpers import distance2bbox, distance2kps
from settings import settings
from server.api_pydantic import BaseResponse, ErrorResponse, ListResponse, SearchResponse, PaginatedUsersResponse, ImageUpload

router = APIRouter(prefix="/face", tags=["Face Recognition"])

@router.post("/detect", response_model=SearchResponse)
async def detect_and_align_faces(
    file: UploadFile = File(...),
    conf_thres: float = settings.DEFAULT_CONF_THRES,  # 使用默认值
    iou_thres: float = settings.DEFAULT_IOU_THRES     # 使用默认值
):
    logger.info("Starting face detection")
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        logger.error("Invalid image file uploaded")
        raise HTTPException(status_code=400, detail="Invalid image file")

    det, kpss = await scrfd.detect(image, conf_thres, iou_thres)
    save_dir = settings.DETECTED_FACES_DIR  # 使用配置中的路径
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    faces_info = []

    for idx, (bbox, kps) in enumerate(zip(det, kpss)):
        x1, y1, x2, y2, score = bbox
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            logger.debug(f"Skipping invalid bbox at index {idx}")
            continue

        try:
            face_region = image[y1:y2, x1:x2]
            crop_kps = kps - np.array([[x1, y1]])
            aligned_face = align_face(face_region, crop_kps)
            filename = f"aligned_{timestamp}_{unique_id}_{idx}.jpg"
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, aligned_face)
            faces_info.append({
                "bbox": [float(x) for x in bbox],
                "keypoints": kps.tolist(),
                "aligned_face_path": save_path
            })
        except Exception as e:
            logger.error(f"Error processing face {idx}: {str(e)}")
            continue

    if not faces_info:
        raise HTTPException(status_code=404, detail="No faces detected")
    logger.info(f"Detected {len(faces_info)} faces")
    return SearchResponse(results=faces_info)

# 人脸搜索
@router.post("/search", response_model=BaseResponse)
async def search_face(
    image: ImageUpload = Depends()  # 使用 Depends 注入 Pydantic 模型
):
    logger.info("Starting face search")
    img = image.img  # 从 Pydantic 模型中获取验证后的图像
    result = face_system.search_face(img)
    if result["status"] == "empty_database":
        logger.warning("Search failed: Database is empty")
        raise HTTPException(status_code=503, detail="Database is empty")
    if result["status"] == "no_match":
        logger.info("No matching face found")
        raise HTTPException(status_code=404, detail="No matching face found")
    
    logger.info(f"Face found for user_id: {result.get('user_id')}")
    return BaseResponse(success=True, message="Face found", data=result)

# 人脸注册
@router.post("/register", response_model=BaseResponse)
async def register_face(
    user_id: str = Form(...),
    image: ImageUpload = Depends()
):
    logger.info(f"Registering face for user_id: {user_id}")
    img = image.img  # 使用验证后的图像
    result = face_system.register_face(user_id, img)
    if result["status"] == "duplicate_id":
        logger.warning(f"Duplicate user_id: {user_id}")
        raise HTTPException(status_code=409, detail="User ID already exists")
    
    logger.info(f"Face registered successfully for user_id: {user_id}")
    return BaseResponse(success=True, message="Face registered successfully", data={"user_id": user_id})

# 数据库状态
@router.get("/database/status", response_model=BaseResponse)
def database_status():
    logger.info("Retrieving database status")
    with face_system.lock:
        status_data = {
            "total_users": len(face_system.id_map),
            "index_size": face_system.index.ntotal,
            "storage_path": str(face_system.data_dir)
        }
    logger.debug(f"Database status: {status_data}")
    return BaseResponse(
        success=True,
        message="Database status retrieved",
        data=status_data
    )

# 删除用户
@router.delete("/users/{user_id}", response_model=BaseResponse, summary="删除用户")
async def delete_user_api(user_id: str):
    logger.info(f"Deleting user_id: {user_id}")
    result = face_system.delete_user(user_id)
    if result["status"] == "user_not_found":
        logger.warning(f"User not found: {user_id}")
        raise HTTPException(status_code=404, detail="User not found")
    logger.info(f"User {user_id} deleted successfully")
    return BaseResponse(
        success=True,
        message=f"User {user_id} deleted successfully"
    )

# 更新用户
@router.put("/users/{user_id}", response_model=BaseResponse, summary="更新用户特征")
async def update_user_api(
    user_id: str,
    image: ImageUpload = Depends()
):
    logger.info(f"Updating user_id: {user_id}")
    img = image.img  # 使用验证后的图像
    result = face_system.update_user(user_id, img)
    if result["status"] == "user_not_found":
        logger.warning(f"User not found: {user_id}")
        raise HTTPException(status_code=404, detail="User not found")
    if result["status"] == "invalid_image":
        logger.error(f"Invalid image for user_id {user_id}: {result['message']}")
        raise HTTPException(status_code=400, detail=result["message"])
    
    logger.info(f"User {user_id} updated successfully")
    return BaseResponse(success=True, message=f"User {user_id} updated successfully")

# 获取所有用户
@router.get("/users", response_model=PaginatedUsersResponse, summary="获取所有用户及其特征")
async def get_all_users_api(limit: int = 100, offset: int = 0):
    logger.info(f"Fetching users with limit={limit}, offset={offset}")
    result = face_system.get_all_users_with_features()
    users = result["users"]
    total = len(users)
    paginated_users = users[offset:offset + limit]
    logger.info(f"Returning {len(paginated_users)} users out of {total}")
    return PaginatedUsersResponse(results=paginated_users, total=total)