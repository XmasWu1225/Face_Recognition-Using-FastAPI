import argparse
import asyncio
import logging
import multiprocessing as mp
import os
import sys
from typing import Optional
from fastapi import FastAPI
from uvicorn import run as uvicorn_run
from settings import basic_settings
from server.utils import set_httpx_config
from server.api_routes.basic_route import basic_router
from server.utils import MakeFastAPIOffline

# 配置日志工具
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
def create_app(run_mode: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="API Server", version=basic_settings.VERSION)
    if basic_settings.OPEN_CROSS_DOMAIN:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # 其他路由可以在这里添加
    app.include_router(basic_router)

    return app

# 运行 API 服务
def run_api_server(started_event: Optional[mp.Event] = None, run_mode: Optional[str] = None):
    set_httpx_config()
    app = create_app(run_mode=run_mode)
    MakeFastAPIOffline(app)
    
    # 设置启动事件
    if started_event is not None:
        @app.on_event("startup")
        async def on_startup():
            started_event.set()

    host = basic_settings.API_SERVER["host"]
    port = basic_settings.API_SERVER["port"]

    ssl_keyfile = basic_settings.SSL_KEYFILE
    ssl_certfile = basic_settings.SSL_CERTFILE

    if ssl_keyfile and ssl_certfile:
        uvicorn_run(
            app,
            host=host,
            port=port,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
        )
    else:
        uvicorn_run(app, host=host, port=port)

# 主函数：启动多个 API 进程
async def start_main_server(args):
    import signal

    def handler(signalname):
        def f(signal_received, frame):
            raise KeyboardInterrupt(f"{signalname} received")
        return f

    signal.signal(signal.SIGINT, handler("SIGINT"))
    signal.signal(signal.SIGTERM, handler("SIGTERM"))

    mp.set_start_method("spawn")
    manager = mp.Manager()

    processes = {}
    api_started_1 = manager.Event()
    api_started_2 = manager.Event()

    # 启动第一个 API 进程
    process_1 = mp.Process(
        target=run_api_server,
        name="API Server 1",
        kwargs=dict(started_event=api_started_1),
        daemon=False,
    )
    processes["api_1"] = process_1

    # 启动第二个 API 进程
    process_2 = mp.Process(
        target=run_api_server,
        name="API Server 2",
        kwargs=dict(started_event=api_started_2),
        daemon=False,
    )
    processes["api_2"] = process_2

    try:
        # 启动所有进程
        for name, process in processes.items():
            process.start()
            process.name = f"{process.name} ({process.pid})"
            logger.info(f"Started {name}")

        # 等待所有进程启动完成
        api_started_1.wait()
        api_started_2.wait()

        logger.info("All API servers have started successfully.")

        # 等待所有进程退出
        while processes:
            for name, process in list(processes.items()):
                process.join(2)
                if not process.is_alive():
                    processes.pop(name)
                    logger.info(f"{name} has stopped.")

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        logger.warning("Caught exception! Stopping all processes...")

    finally:
        # 强制终止所有进程
        for process in processes.values():
            logger.warning(f"Sending SIGKILL to {process.name}")
            process.kill()
        logger.info("All processes have been terminated.")

# 主入口
def main():
    cwd = os.getcwd()
    sys.path.append(cwd)
    print(f"Current working directory: {cwd}")

    if sys.version_info < (3, 10):
        loop = asyncio.get_event_loop()
    else:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(start_main_server(None))

if __name__ == "__main__":
    main()