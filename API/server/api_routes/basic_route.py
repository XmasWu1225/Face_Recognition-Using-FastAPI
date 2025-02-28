from fastapi import APIRouter
from server.api_pydantic import *

from server.basic_usage.basic_api_use import (
    get_sync_task,
    get_async_task,
    get_tasks_in_thread_pool,
    get_tasks_in_process_pool,
    run_async_sample_task,
    get_sync_generator,
    post_sync_task,
    post_async_task,
    search_docs,
    async_search_docs,
)

basic_router = APIRouter(prefix="/basic_example", tags=["Basic Example"])

# 同步任务
basic_router.get("/sync-task/{task_id}", response_model=BaseResponse, summary="同步任务")(get_sync_task)

# 异步任务
basic_router.get("/async-task/{task_id}", response_model=BaseResponse, summary="异步任务")(get_async_task)

# 线程池任务
basic_router.get("/thread-pool-tasks", response_model=ListResponse, summary="线程池任务")(get_tasks_in_thread_pool)

# 进程池任务
basic_router.get("/process-pool-tasks", response_model=ListResponse, summary="进程池任务")(get_tasks_in_process_pool)

# 同步环境的异步任务
basic_router.get("/run_async_sample_task/{task_id}", response_model=BaseResponse, summary="同步环境的异步任务")(run_async_sample_task)

# 同步生成器
basic_router.get("/sync-generator", response_model=ListResponse, summary="同步生成器")(get_sync_generator)

# POST 同步任务
basic_router.post("/post-sync-task", response_model=BaseResponse, summary="POST 同步任务")(post_sync_task)

# POST 异步任务
basic_router.post("/post-async-task", response_model=BaseResponse, summary="POST 异步任务")(post_async_task)

# GET 搜索文档
basic_router.get("/search-docs", response_model=SearchResponse, summary="搜索文档")(search_docs)

# 异步 GET 搜索文档
basic_router.get("/async-search-docs", response_model=SearchResponse, summary="异步搜索文档")(async_search_docs)