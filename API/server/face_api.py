# cd /home/aa/API
# python3 -m server.face_api
from settings import settings
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from server.api_routes.face_routes import router as face_router
from server.models import scrfd, face_system  # 从 models 导入实例
from server.logger import logger

app = FastAPI(title="Face Recognition API")
app.include_router(face_router)

# 记录程序启动
logger.info("Starting Face Recognition API...")

# 根界面
@app.get("/", response_class=HTMLResponse)
async def read_root():
    # 获取状态信息
    with face_system.lock:
        total_users = len(face_system.id_map)
        index_size = face_system.index.ntotal
    
    # HTML 内容
    html_content = f"""
    <html>
        <head>
            <title>Face Recognition API - 欢迎</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    background: linear-gradient(135deg, #ffafbd, #ffc3a0);
                    margin: 0;
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    color: #333;
                }}
                .container {{
                    background: rgba(255, 255, 255, 0.9);
                    padding: 40px;
                    border-radius: 20px;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
                    text-align: center;
                    max-width: 600px;
                    animation: fadeIn 1s ease-in-out;
                }}
                h1 {{
                    color: #ff6f61;
                    font-size: 48px;
                    margin-bottom: 20px;
                    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
                }}
                p {{
                    font-size: 20px;
                    line-height: 1.6;
                    margin: 10px 0;
                }}
                .status {{
                    background: #ffebcd;
                    padding: 15px;
                    border-radius: 10px;
                    margin: 20px 0;
                    font-weight: bold;
                    color: #e67e22;
                    animation: pulse 2s infinite;
                }}
                a {{
                    color: #3498db;
                    text-decoration: none;
                    font-weight: bold;
                    transition: color 0.3s;
                }}
                a:hover {{
                    color: #e74c3c;
                }}
                .highlight {{
                    color: #9b59b6;
                    font-style: italic;
                }}
                @keyframes fadeIn {{
                    from {{ opacity: 0; transform: translateY(-20px); }}
                    to {{ opacity: 1; transform: translateY(0); }}
                }}
                @keyframes pulse {{
                    0% {{ transform: scale(1); }}
                    50% {{ transform: scale(1.05); }}
                    100% {{ transform: scale(1); }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>欢迎体验人脸识别 API ✨</h1>
                <p>这是一个基于 <span class="highlight">FastAPI</span> 构建的高效人脸识别服务！</p>
                <div class="status">
                    当前状态：<br>
                    注册用户数: {total_users} &nbsp;|&nbsp; 索引大小: {index_size}
                </div>
                <p>想要交互式测试？请访问：<br>
                    <a href="/docs" target="_blank">http://localhost:8000/docs</a>
                </p>
                <p>让我们一起探索人脸识别的奇妙世界吧！🎉</p>
            </div>
        </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    import uvicorn
    ssl_kwargs = {}
    if settings.SSL_KEYFILE and settings.SSL_CERTFILE:
        ssl_kwargs = {
            "ssl_keyfile": settings.SSL_KEYFILE,
            "ssl_certfile": settings.SSL_CERTFILE,
        }
    uvicorn.run(
        app,
        host=settings.API_SERVER["host"],
        port=settings.API_SERVER["port"],
        **ssl_kwargs
    )