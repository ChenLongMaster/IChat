@echo off

:: Start Qdrant if not running
docker inspect -f "Qdrant running" chatbot-qdrant 2>nul || (
    echo Starting Qdrant...
    docker run -d --name chatbot-qdrant -p 6333:6333 qdrant/qdrant
)

:: Activate virtual environment
call venv\Scripts\activate

:: Run the app with correct port
uvicorn app.main:app --host 0.0.0.0 --port 5001
