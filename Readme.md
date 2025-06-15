1. Create .env file (ask Long for detail content)
2. Open Terminal, run: ".\venv\Scripts\activate" to create virtual env
3. Install packages: "pip install -r requirements.txt"
4. Run Qdrant: "docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant" (Docker install is required)
5. Run app: "uvicorn app.main:app --reload"

--Docker Deployment
1. .\venv\Scripts\activate 
2. pip freeze > requirements.txt
1. docker build -t ichat-api .