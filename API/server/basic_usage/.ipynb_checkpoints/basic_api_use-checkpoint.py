import asyncio
from typing import List, Dict, Generator
from server.utils import run_async, iter_over_async, run_in_thread_pool, run_in_process_pool
from server.api_pydantic import *


# 示例数据
def sample_data() -> List[Dict]:
    return [{"id": i, "name": f"Task-{i}"} for i in range(5)]

# 示例异步生成器
async def async_generator_example():
    for i in range(5):
        await asyncio.sleep(0.1)
        yield {"id": i, "name": f"Async-Task-{i}"}

# 示例任务函数
def sample_task(task_id: int) -> Dict:
    return {"task_id": task_id, "status": "completed"}

# 示例异步任务函数
async def async_sample_task(task_id: int) -> Dict:
    await asyncio.sleep(0.1)
    return {"task_id": task_id, "status": "completed"}

# 示例 API 函数
def get_sync_task(task_id: int) -> Dict:
    """同步任务"""
    return sample_task(task_id)

async def get_async_task(task_id: int) -> Dict:
    """异步任务"""
    return await async_sample_task(task_id)

def get_tasks_in_thread_pool() -> List[Dict]:
    """在线程池中运行任务"""
    tasks = list(run_in_thread_pool(sample_task, [{"task_id": i} for i in range(5)]))
    return ListResponse(items=tasks)
    
def get_tasks_in_process_pool() -> Generator:
    """在进程池中运行任务"""
    tasks = list(run_in_process_pool(sample_task, [{"task_id": i} for i in range(5)]))
    return ListResponse(items=tasks)

def run_async_sample_task(task_id: int) -> Dict:
    """在同步环境 使用 run_async 运行异步任务"""
    return run_async(async_sample_task(task_id))

async def get_async_generator() -> List[Dict]:
    """将异步生成器封装为同步生成器"""
    return [item async for item in async_generator_example()]

def get_sync_generator() -> List[Dict]:
    """将异步生成器封装为同步生成器"""
    tasks = list(iter_over_async(async_generator_example()))
    return ListResponse(items=tasks)

    
def post_sync_task(task: Dict) -> Dict:
    """POST 请求处理同步任务"""
    return sample_task(**task)

async def post_async_task(task: Dict) -> Dict:
    """POST 请求处理异步任务"""
    return await async_sample_task(**task)

def search_docs(query: str) -> List[Dict]:
    """GET 请求搜索文档"""
    return [{"doc_id": i, "content": f"Result for {query}-{i}"} for i in range(3)]

async def async_search_docs(query: str) -> List[Dict]:
    """异步 GET 请求搜索文档"""
    await asyncio.sleep(0.1)
    return [{"doc_id": i, "content": f"Result for {query}-{i}"} for i in range(3)]