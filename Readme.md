1. Create .env file (ask Long for detail content)
2. Open Terminal, run: ".\venv\Scripts\activate" to create virtual env
3. Install packages: "pip install -r requirements.txt"
4. Run Qdrant: "docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant" (Docker install is required)
5. Run app: "uvicorn main:app --reload"