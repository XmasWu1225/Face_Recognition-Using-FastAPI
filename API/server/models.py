from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import cv2
import numpy as np
import faiss
from typing import Dict
import threading
import tensorrt as trt
import utils.tensorrt_common as common
import os
import json
from server.logger import logger
from pathlib import Path
import time
import uuid
from utils.helpers import distance2bbox, distance2kps
from settings import settings

TRT_LOGGER = trt.Logger()

# 模型组件 -------------------------------------------------
# 人脸对齐函数
def align_face(img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    使用关键点进行人脸对齐
    :param img: 原始人脸区域图像
    :param landmarks: 5个关键点坐标 (x,y) 格式
    :return: 对齐后的112x112人脸图像
    """
    # ArcFace的标准对齐坐标（基于InsightFace）
    REFERENCE_FACIAL_POINTS = np.array(
    [[38.2946, 51.6963], # 左眼中心
     [73.5318, 51.5014], # 右眼中心
     [56.0252, 71.7366], # 鼻子尖
     [41.5493, 92.3655], # 左嘴角
     [70.7299, 92.2041]], # 右嘴角 
    dtype=np.float32)
    
    # 确保传入5个关键点
    assert landmarks.shape[0] == 5, "需要5个面部关键点"
    
    # 提取关键点坐标（SCRFD的关键点顺序通常为：左眼、右眼、鼻子、左嘴角、右嘴角）
    src_pts = landmarks.astype(np.float32)
    
    # 计算仿射变换矩阵（使用眼睛和鼻子作为基准点）
    tfm = cv2.estimateAffinePartial2D(src_pts[:2], REFERENCE_FACIAL_POINTS[:2], method=cv2.LMEDS)[0]
    
    # 执行仿射变换
    aligned_face = cv2.warpAffine(img, tfm, (112, 112), flags=cv2.INTER_LINEAR)
    
    
    return aligned_face

class SCRFD:
    def __init__(self, engine_file_path, input_resolution=(640, 640)):
        self.engine_file_path = engine_file_path
        self.input_resolution = input_resolution
        self.engine = self.get_engine()
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)

    def get_engine(self):
        with open(self.engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def preprocess_image(self, image):
        original_height, original_width = image.shape[:2]
        input_height, input_width = self.input_resolution

        im_ratio = float(original_height) / original_width
        model_ratio = input_height / input_width
        if im_ratio > model_ratio:
            new_height = input_height
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_width
            new_height = int(new_width * im_ratio)

        resized_image = cv2.resize(image, (new_width, new_height))
        det_scale = float(new_height) / image.shape[0]
        det_image = np.zeros((input_height, input_width, 3), dtype=np.uint8)
        det_image[:new_height, :new_width, :] = resized_image
        input_size = tuple(det_image.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(det_image, 1.0 / 128.0, input_size, (127.5, 127.5, 127.5), swapRB=True)

        return blob, det_image, det_scale

    def postprocess_outputs(self, outputs, blob, det_scale, conf_thres=0.5, iou_thres=0.4):
        fmc = 3
        feat_stride_fpn = [8, 16, 32]
        num_anchors = 2

        scores_list, bboxes_list, kpss_list = [], [], []
        center_cache = {}
        input_height, input_width = blob.shape[2], blob.shape[3]

        for idx, stride in enumerate(feat_stride_fpn):
            scores = outputs[idx]
            bbox_preds = outputs[idx + fmc].reshape(-1, 4) * stride
            kps_preds = outputs[idx + fmc * 2].reshape(-1, 10) * stride

            height, width = input_height // stride, input_width // stride
            key = (height, width, stride)
            if key in center_cache:
                anchor_centers = center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))
                if len(center_cache) < 100:
                    center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= conf_thres)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            kpss = distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

        scores = np.vstack(scores_list)
        order = scores.ravel().argsort()[::-1]
        bboxes = np.vstack(bboxes_list)[order, :]
        kpss = np.vstack(kpss_list)[order, :, :]

        bboxes /= det_scale
        kpss /= det_scale

        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        keep = self.nms(pre_det, iou_thres=iou_thres)
        det = pre_det[keep, :]
        kpss = kpss[keep, :, :]
        return det, kpss

    def nms(self, dets, iou_thres):
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            indices = np.where(ovr <= iou_thres)[0]
            order = order[indices + 1]
        return keep

    async def detect(self, image: np.ndarray, conf_thres=0.5, iou_thres=0.4):
        blob, det_image, det_scale = self.preprocess_image(image)
        self.inputs[0].host = blob
        trt_outputs = common.do_inference(
            self.context, engine=self.engine, bindings=self.bindings, 
            inputs=self.inputs, outputs=self.outputs, stream=self.stream
        )
        trt_outputs = [output.reshape(-1, 1) for output in trt_outputs]
        det, kpss = self.postprocess_outputs(trt_outputs, blob, det_scale, conf_thres, iou_thres)
        return det, kpss

class ArcFace:
    """TensorRT推理的ArcFace模型"""
    def __init__(self, engine_file_path: str):
        self.engine_file_path = engine_file_path
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)

    def _load_engine(self) -> trt.ICudaEngine:
        """加载序列化的TensorRT引擎"""
        with open(self.engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """执行推理"""
        # 预处理
        blob = cv2.dnn.blobFromImage(
            image,
            1.0 / 127.5,
            (112, 112),
            (127.5, 127.5, 127.5),
            swapRB=True
        )
        
        # 推理
        self.inputs[0].host = blob
        trt_outputs = common.do_inference(
            self.context,
            engine=self.engine,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream,
        )
        return trt_outputs[0].flatten()

# 系统核心 -------------------------------------------------
class FaceRecSystem:
    """带持久化的人脸识别系统"""
    def __init__(self, engine_path: str, data_dir: str = "./face_data"):
        # 初始化配置
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化模型
        self.arcface = ArcFace(engine_path)
        self.dim = 512  # ArcFace特征维度
        
        # 初始化存储
        self.lock = threading.Lock()
        self.gpu_res = faiss.StandardGpuResources()
        self._init_database()
    
    def _init_database(self):
        """初始化或加载已有数据库"""
        # 加载 FAISS 索引
        index_file = self.data_dir / "faiss_index.bin"
        if index_file.exists():
            cpu_index = faiss.read_index(str(index_file))
            self.index = faiss.index_cpu_to_gpu(self.gpu_res, 0, cpu_index)
        else:
            self.index = faiss.index_cpu_to_gpu(
                self.gpu_res, 0, faiss.IndexFlatL2(self.dim)
            )
        
        # 加载 ID 映射 ?
        map_file = self.data_dir / "id_map.json"
        self.id_map = {}
        if map_file.exists():
            try:
                with open(map_file, "r") as f:
                    content = f.read().strip()
                    if content:
                        self.id_map = json.loads(content)
                    else:
                        logger.warning(f"{map_file} is empty, initializing with empty id_map")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse {map_file}: {e}, initializing with empty id_map")
            except Exception as e:
                logger.error(f"Failed to read {map_file}: {e}, initializing with empty id_map")
        
        if self.index.ntotal != len(self.id_map):
            logger.warning(f"Index count ({self.index.ntotal}) doesn't match ID count ({len(self.id_map)}), resetting index")
            self.index = faiss.index_cpu_to_gpu(
                self.gpu_res, 0, faiss.IndexFlatL2(self.dim)
            )
            self.id_map = {}
            self._save_database()  # 保存修复后的状态
    def _save_database(self):
        """保存数据库到磁盘"""
        # 保存FAISS索引
        cpu_index = faiss.index_gpu_to_cpu(self.index)
        faiss.write_index(cpu_index, str(self.data_dir / "faiss_index.bin"))
        
        # 保存ID映射
        with open(self.data_dir / "id_map.json", "w") as f:
            json.dump(self.id_map, f, indent=2)

    def _validate_image(self, img: np.ndarray):
        """验证输入图像格式"""
        if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
            raise ValueError("Image must be uint8 numpy array")
        if img.shape != (112, 112, 3):
            raise ValueError(f"Invalid image shape {img.shape}, expected (112, 112, 3)")

    def search_face(self, image: np.ndarray) -> Dict: #参考项目的实际计算方式
        """人脸搜索"""
        self._validate_image(image)
        
        # 特征提取
        embedding = self.arcface(image).astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(embedding)
        
        with self.lock:
            if self.index.ntotal == 0:
                return {"status": "empty_database"}
            
            distances, indices = self.index.search(embedding, 1)
        
        # 处理结果
        if indices[0][0] == -1 or distances[0][0] > 1.0:
            return {"status": "no_match"}
        
        user_id = next(k for k, v in self.id_map.items() if v == indices[0][0])
        return {
            "status": "success",
            "user_id": user_id,
            "confidence": float(1 - distances[0][0]/2) # 余弦相似度转换为置信度
        }

    def register_face(self, user_id: str, image: np.ndarray) -> Dict:
        """人脸注册"""
        self._validate_image(image)
        
        # 特征提取
        embedding = self.arcface(image).astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(embedding)
        
        with self.lock:
            # 检查重复ID
            if user_id in self.id_map:
                return {"status": "duplicate_id"}
            
            # 添加到数据库
            self.index.add(embedding)
            self.id_map[user_id] = self.index.ntotal - 1
            self._save_database()  # 持久化
        
        return {"status": "success"}
    def delete_user(self, user_id: str) -> Dict:
        """删除指定用户"""
        with self.lock:
            if user_id not in self.id_map:
                return {"status": "user_not_found"}
            
            # 获取被删除的索引位置
            deleted_idx = self.id_map[user_id]
            
            # 创建新的索引并复制除被删除项外的所有向量
            old_index = faiss.index_gpu_to_cpu(self.index)
            new_index = faiss.IndexFlatL2(self.dim)
            
            # 复制所有向量，除了要删除的那个
            for i in range(old_index.ntotal):
                if i != deleted_idx:
                    vector = old_index.reconstruct(i)
                    new_index.add(np.array([vector]))
            
            # 更新GPU索引
            self.index = faiss.index_cpu_to_gpu(self.gpu_res, 0, new_index)
            
            # 更新id_map
            del self.id_map[user_id]
            # 更新后续索引
            for uid in self.id_map:
                if self.id_map[uid] > deleted_idx:
                    self.id_map[uid] -= 1
                    
            self._save_database()
            
            return {"status": "success"}

    def update_user(self, user_id: str, new_image: np.ndarray) -> Dict:
        """更新用户特征"""
        try:
            self._validate_image(new_image)
        except ValueError as e:
            return {"status": "invalid_image", "message": str(e)}
        
        # 提取新特征
        new_embedding = self.arcface(new_image).astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(new_embedding)
        
        with self.lock:
            if user_id not in self.id_map:
                return {"status": "user_not_found"}
            
            # 获取目标索引
            target_idx = self.id_map[user_id]
            
            # 创建新索引并替换旧向量
            old_index = faiss.index_gpu_to_cpu(self.index)
            new_index = faiss.IndexFlatL2(self.dim)
            
            # 复制所有向量，替换目标向量
            for i in range(old_index.ntotal):
                if i == target_idx:
                    new_index.add(new_embedding)
                else:
                    vector = old_index.reconstruct(i)
                    new_index.add(np.array([vector]))
            
            # 更新GPU索引
            self.index = faiss.index_cpu_to_gpu(self.gpu_res, 0, new_index)
            self._save_database()
            
            return {"status": "success"}
        
    def _rebuild_index(self):
        """重建Faiss索引"""
        self.gpu_index.reset()
        if len(self.embeddings) > 0:
            self.gpu_index.add(np.vstack(self.embeddings))

    def get_all_users_with_features(self) -> Dict[str, list]:
        with self.lock:
            try:
                if not self.id_map:
                    return {"users": []}
                
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                users_data = []
                
                for user_id, idx in self.id_map.items():
                    try:
                        feature_vector = cpu_index.reconstruct(idx).tolist()
                        users_data.append({
                            "user_id": user_id,
                            "features": feature_vector
                        })
                    except Exception as e:
                        print(f"Error reconstructing vector for {user_id}: {str(e)}")
                        continue
                
                return {"users": users_data}
            except Exception as e:
                print(f"Error getting users: {str(e)}")
                return {"users": [], "error": str(e)}

    def clear_database(self):
        """清空所有数据"""
        with self.lock:
            # 重置 FAISS 索引为一个新的空索引
            self.index = faiss.index_cpu_to_gpu(
                self.gpu_res, 0, faiss.IndexFlatL2(self.dim)
            )
            # 清空 ID 映射
            self.id_map.clear()
            # 保存数据库更改
            self._save_database()

# 初始化模型引擎 -------------------------------------------------
app = FastAPI(title="Face Recognition API")
scrfd = SCRFD(settings.SCRFD_ENGINE_PATH)

face_system = FaceRecSystem(
    engine_path=settings.ARCFACE_ENGINE_PATH,
    data_dir=settings.DATA_DIR
)