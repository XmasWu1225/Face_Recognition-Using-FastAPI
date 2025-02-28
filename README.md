以下是更新后的 README 文件，我在“项目概述”部分添加了一个介绍，突出项目的特点和独到之处，同时保留了之前的结构和补充了 `settings.py` 的内容：

---

# Face Recognition API 项目

## 项目概述

这是一个基于 FastAPI 构建的高效人脸识别服务，集成了前沿的人脸检测（SCRFD）、人脸对齐、特征提取（ArcFace）和人脸识别功能，专为高性能和实时性需求设计。项目的独特之处在于其深度优化的技术栈与灵活的架构：通过 TensorRT 实现 GPU 加速推理，结合 FAISS 的高效向量搜索，确保毫秒级的识别响应；同时引入统一的配置管理（`settings.py`），支持动态调整模型路径、存储目录和运行参数，极大提升了部署的适应性和可维护性。与传统人脸识别系统相比，本项目在性能、扩展性和易用性上具有显著优势，特别适合需要快速集成和定制的商业或研究场景。

### 项目特点与独到之处
- **极致性能**：利用 TensorRT 加速 SCRFD 和 ArcFace 模型推理，配合 FAISS GPU 索引，实现超低延迟的人脸检测与识别。
- **模块化设计**：人脸检测、对齐、特征提取和识别逻辑分离，便于独立优化或替换模块。
- **灵活配置**：通过 `settings.py` 和 `.env` 文件统一管理配置，支持运行时环境变量覆盖，适应多种部署环境。
- **数据持久化**：FAISS 索引和用户 ID 映射以文件形式存储，保证重启后数据不丢失。
- **用户友好**：提供直观的 RESTful API 和交互式文档（`/docs`），附带美观的欢迎页面，便于快速上手。
- **开源可扩展**：代码结构清晰，易于二次开发，支持添加新功能如批量处理或多模型支持。

## 功能特性

- **人脸检测**: 使用 SCRFD 模型检测图像中的人脸位置和关键点。
- **人脸对齐**: 根据检测到的关键点对齐人脸到标准 112x112 尺寸。
- **特征提取**: 使用 ArcFace 模型提取人脸特征向量。
- **人脸搜索**: 通过 FAISS 索引快速搜索匹配的人脸。
- **用户管理**: 支持人脸注册、更新、删除及数据库状态查看。
- **持久化存储**: 人脸特征和用户 ID 映射存储在本地 FAISS 索引和 JSON 文件中。
- **API 接口**: 提供 RESTful API，支持图像上传和人脸识别操作。
- **日志记录**: 集成日志系统，记录运行状态和错误信息。
- **前端界面**: 提供简洁的欢迎页面，展示服务状态。
- **配置管理**: 通过 `settings.py` 统一管理服务器、模型路径和存储路径等配置。

## 项目结构

```
face_recognition_api/
├── server/
│   ├── api_routes/
│   │   └── face_routes.py     # API 路由定义
│   ├── models.py             # 核心模型（SCRFD, ArcFace, FaceRecSystem）
│   ├── face_api.py           # FastAPI 应用入口
│   └── api_pydantic.py       # Pydantic 数据模型和验证
├── utils/
│   └── helpers.py            # 工具函数（distance2bbox, distance2kps 等）
├── data/
│   ├── detected_faces/       # 检测到的人脸图像存储目录
│   └── face_recognition/     # FAISS 索引和 ID 映射存储目录
├── settings.py               # 项目配置文件（新增）
├── .env                      # 环境变量文件（可选）
└── README.md                 # 项目说明文档
```

## 技术栈

- **后端框架**: FastAPI
- **模型推理**: TensorRT
- **人脸检测**: SCRFD
- **特征提取**: ArcFace
- **向量搜索**: FAISS (GPU 加速)
- **图像处理**: OpenCV
- **数据验证**: Pydantic
- **配置管理**: Pydantic Settings (`pydantic-settings`)
- **日志**: 自定义 logger
- **运行时**: Python 3.8+

## 安装与运行

### 依赖安装
```bash
pip install fastapi uvicorn opencv-python numpy faiss-gpu tensorrt pydantic-settings
```

### 模型准备
1. 将 SCRFD 模型文件（如 `API_detection-l.engine`）和 ArcFace 模型文件（如 `API_arcface.engine`）放置在 `settings.py` 中指定的路径（默认 `/home/aa/API/SCRFD/`）。
2. 确保 GPU 环境支持 TensorRT 和 FAISS。

### 配置管理
项目使用 `settings.py` 管理配置，默认配置如下：
- **服务器地址**: `0.0.0.0:8000`
- **模型路径**: `/home/aa/API/SCRFD/API_detection-l.engine` 和 `/home/aa/API/SCRFD/API_arcface.engine`
- **存储路径**: `./data/face_recognition` 和 `./data/detected_faces`

你可以通过以下方式自定义配置：
1. **修改 `settings.py` 中的默认值**。
2. **创建 `.env` 文件**，例如：
   ```
   SCRFD_ENGINE_PATH=/path/to/scrfd.engine
   ARCFACE_ENGINE_PATH=/path/to/arcface.engine
   DATA_DIR=/custom/data/face_recognition
   DETECTED_FACES_DIR=/custom/data/detected_faces
   API_SERVER__host=127.0.0.1
   API_SERVER__port=8080
   ```
3. **运行时环境变量覆盖**，例如：
   ```bash
   export SCRFD_ENGINE_PATH=/new/path/to/scrfd.engine
   ```

### 运行服务
```bash
cd /home/aa/API
python3 -m server.face_api
```

服务默认运行在 `http://0.0.0.0:8000`（或 `.env` 中指定的地址），访问 `/` 查看欢迎页面，或访问 `/docs` 使用交互式 API 文档。

## API 端点

| 端点                   | 方法   | 描述                | 参数                              |
|------------------------|--------|---------------------|-----------------------------------|
| `/face/detect`         | POST   | 检测并对齐人脸      | `file` (图像文件), `conf_thres`, `iou_thres` |
| `/face/search`         | POST   | 搜索匹配人脸        | `file` (图像文件)                |
| `/face/register`       | POST   | 注册新用户          | `user_id`, `file` (图像文件)     |
| `/face/database/status`| GET    | 获取数据库状态      | 无                               |
| `/face/users/{user_id}`| DELETE | 删除指定用户        | `user_id`                        |
| `/face/users/{user_id}`| PUT    | 更新用户特征        | `user_id`, `file` (图像文件)     |
| `/face/users`          | GET    | 获取所有用户信息    | `limit`, `offset`                |

## 数据存储

- **检测结果**: 对齐后的人脸图像存储在 `settings.DETECTED_FACES_DIR`（默认 `./data/detected_faces/`）。
- **特征数据库**: FAISS 索引存储在 `settings.DATA_DIR/faiss_index.bin`（默认 `./data/face_recognition/faiss_index.bin`），ID 映射存储在 `settings.DATA_DIR/id_map.json`。

## 注意事项

- 输入图像需为 112x112 RGB 格式，否则会被拒绝。
- 服务依赖 GPU 加速，确保运行环境支持 CUDA。
- 日志记录需配置 `server.logger` 模块（未提供完整实现）。
- Python 版本需为 3.8 或更高，若使用 `|` 类型注解需升级至 3.10+。

## 未来改进

- 添加异步文件读取优化上传性能。
- 支持多模型切换和动态配置。
- 集成用户认证和权限控制。
- 提供批量注册和搜索功能。
- 在 `settings.py` 中支持更多高级配置（如日志级别、FAISS 参数）。

## 贡献

欢迎提交 Issue 或 Pull Request 来改进项目！

