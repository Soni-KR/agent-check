# Agent Check

A local AI agent with calculator and notes tools, plus automated reliability checks. Built with React, FastAPI, and SQLite. Uses free Ollama models; no API key needed.

## Run

Install Python 3.11+, Node.js 22+, and Ollama. Open Ollama, then download the model once:

```powershell
ollama pull qwen2.5:7b
```

From the project folder, start the backend:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
cd backend
..\.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

In a second terminal, from the project folder:

```powershell
cd frontend
npm.cmd ci
npm.cmd run dev
```

Open **http://127.0.0.1:5173**. Use **Autonomous agent** for tasks or **Evaluation lab** for tests.
