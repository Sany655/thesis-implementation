# Dengue Clinical Risk Dashboard — Local Development Guide

This directory contains the full-stack clinical decision support system for dengue severity risk assessment and SHAP-based explainability across pediatric and adult cohorts.

---

## 🏗️ Architecture Overview

- **Backend (`src/backend`)**: FastAPI REST API providing ML inference (`XGBoost`, `Random Forest`), localized SHapley Additive exPlanations (`SHAP`), and an asynchronous SQLite database for patient assessment records.
- **Frontend (`src/dashboard`)**: Modern React 19 + Vite web application featuring interactive biomarker inputs, dynamic SHAP waterfall charts, longitudinal patient timeline tracking, and PDF report export.

```
       ┌───────────────────────────────┐
       │   React 19 + Vite Frontend    │
       │   (http://localhost:5173)     │
       └──────────────┬────────────────┘
                      │
                      │ HTTP / REST API
                      ▼
       ┌───────────────────────────────┐
       │     FastAPI ML Backend        │
       │   (http://localhost:8000)     │
       └───────┬──────────────┬────────┘
               │              │
        ┌──────▼──────┐ ┌─────▼──────────┐
        │ SQLite DB   │ │ ML Models/SHAP │
        └─────────────┘ └────────────────┘
```

---

## 📋 Prerequisites

Before running the application, make sure you have installed:
- **Python**: `3.10` or higher (`python --version`)
- **Node.js**: `v18.0.0` or higher (`node -v`)
- **npm**: `v9.0.0` or higher (`npm -v`)

---

## 🚀 Quick Start Guide

### 1️⃣ Backend Setup & Launch (Terminal 1)

Navigate to the `backend` directory:

```bash
cd src/backend
```

#### Step A: Create and Activate Virtual Environment

* **Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
  *(If script execution is restricted in PowerShell, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

* **Windows (Command Prompt / CMD)**:
  ```cmd
  python -m venv venv
  venv\Scripts\activate.bat
  ```

* **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

#### Step B: Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### Step C: Environment Configuration (Optional)

You can create a `.env` file from the provided template:

```bash
cp .env.example .env
```

Default settings in `.env`:
```env
DATABASE_URL=sqlite+aiosqlite:///./dengue_dashboard.db
MODEL_DIR=../models
```

#### Step D: Run the FastAPI Server

```bash
python -m uvicorn api:app --reload --port 8000

or

uvicorn api:app --reload --host 127.0.0.1
```

* **Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **API Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

### 2️⃣ Frontend Setup & Launch (Terminal 2)

Open a new terminal window and navigate to the `dashboard` directory:

```bash
cd src/dashboard
```

#### Step A: Install Node Dependencies

```bash
npm install
```

#### Step B: Environment Configuration (Optional)

Copy `.env.example` to `.env` if not already present:

```bash
cp .env.example .env
```

Ensure `VITE_API_URL` points to your local FastAPI backend:
```env
VITE_API_URL=http://localhost:8000
```

#### Step C: Run the Vite Development Server

```bash
npm run dev
```

* **Dashboard URL**: [http://localhost:5173](http://localhost:5173)

| Component | Task | Command |
| :--- | :--- | :--- |
| **Backend** | Activate venv (PowerShell) | `.\src\backend\venv\Scripts\Activate.ps1` |
| **Backend** | Install dependencies | `pip install -r src/backend/requirements.txt` |
| **Backend** | Start dev server | `python -m uvicorn api:app --reload --port 8000` *(inside `src/backend`)* |
| **Frontend** | Install dependencies | `npm install` *(inside `src/dashboard`)* |
| **Frontend** | Start dev server | `npm run dev` *(inside `src/dashboard`)* |
| **Frontend** | Build for production | `npm run build` *(inside `src/dashboard`)* |

---

## 🔍 Troubleshooting

- **Backend says models not found**:
  Ensure the `models/` directory contains `pediatric_best.pkl` and `adult_best.pkl`. If launching from a different directory, verify `MODEL_DIR=../models` in `.env`.
- **CORS Error on Dashboard**:
  Ensure the backend is running on port `8000` and `VITE_API_URL=http://localhost:8000` in `src/dashboard/.env`.
- **Port 8000 or 5173 already in use**:
  - For backend: change port using `--port 8001` and update `VITE_API_URL` accordingly.
  - For frontend: Vite will automatically suggest the next open port (e.g. `5174`).
