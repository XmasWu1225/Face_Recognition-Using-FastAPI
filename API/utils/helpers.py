'''
一个用于人脸检测和对齐的工具模块，
主要用于图像预处理，可能与人脸识别（如ArcFace）相关。
它包含了几个独立的功能函数，
涉及人脸关键点对齐、图像裁剪、边界框计算、关键点调整和特征向量标准化等操作。
'''
import cv2
import numpy as np
from skimage.transform import SimilarityTransform # 用于计算仿射变换矩阵

# 这些是人脸识别模型的参考点，用于对齐人脸
# 5个关键点的坐标
reference_alignment = np.array(
    [[
        [38.2946, 51.6963], # 左眼
        [73.5318, 51.5014], # 右眼
        [56.0252, 71.7366], # 鼻尖
        [41.5493, 92.3655], # 左 嘴角nostril
        [70.7299, 92.2041] # 右 嘴角nostril
    ]],
    dtype=np.float32
)

# 估计关键点对齐矩阵和关键点对齐索引
def estimate_norm(landmark, image_size=112):
    assert landmark.shape == (5, 2)
    min_matrix = []
    min_index = []
    min_error = float('inf')

    landmark_transform = np.insert(landmark, 2, values=np.ones(5), axis=1)
    transform = SimilarityTransform()

    if image_size == 112:
        alignment = reference_alignment
    else:
        alignment = float(image_size) / 112 * reference_alignment

    # 循环是为了找到最小误差的对齐矩阵
    for i in np.arange(alignment.shape[0]):
        transform.estimate(landmark, alignment[i])
        matrix = transform.params[0:2, :]
        results = np.dot(matrix, landmark_transform.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - alignment[i]) ** 2, axis=1)))
        if error < min_error:
            min_error = error
            min_matrix = matrix
            min_index = i
    return min_matrix, min_index


# 根据关键点对齐矩阵裁剪图像
def norm_crop_image(image, landmark, image_size=112, mode='arcface'):
    M, pose_index = estimate_norm(landmark, image_size)
    # warpAffine()函数 它的第一个参数是输入图像，第二个参数是变换矩阵，第三个参数是输出图像的大小。
    warped = cv2.warpAffine(image, M, (image_size, image_size), borderValue=0.0)
    return warped

# 计算边界框
def distance2bbox(points, distance, max_shape=None):

    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

# 计算关键点
def distance2kps(points, distance, max_shape=None):

    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

# 对特征向量进行L2归一化
def normalize(X):
    norms = np.linalg.norm(X, axis=1)
    norms = np.where(norms == 0, 1, norms)
    X /= norms[:, np.newaxis]
    return X
