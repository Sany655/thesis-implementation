# Dengue Risk Assessment Dashboard

This folder contains the source code for the Early-stage Explainable Dengue Risk Assessment Dashboard. The project is split into two components: a Python FastAPI backend and a React/Vite frontend.

## Prerequisites
- **Python 3.8+** (for the backend)
- **Node.js 18+** (for the frontend)

---

## 1. Starting the Backend (API & Database)

The backend is built with FastAPI, SQLAlchemy, and relies on XGBoost/SHAP for machine learning inference. It uses a local SQLite database (`dengue_dashboard.db`) to store patient assessments.

1. Open a terminal and navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the FastAPI server:
   ```bash
   uvicorn api:app --reload
   ```
   *The backend API will be available at `http://localhost:8000`.*
   *You can view the interactive API documentation at `http://localhost:8000/docs`.*

---

## 2. Starting the Frontend (Dashboard UI)

The frontend is a modern React application powered by Vite.

1. Open a **new** terminal and navigate to the `dashboard` directory:
   ```bash
   cd dashboard
   ```
2. Install the necessary Node.js dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
   *The dashboard will typically be available at `http://localhost:5173`.*

---

## Usage
Once both the backend and frontend are running, open your web browser to the frontend URL (e.g., `http://localhost:5173`). 
You can enter patient demographics and clinical laboratory parameters to generate an XGBoost risk prediction, view SHAP feature explanations, and load longitudinal assessment histories using the `Patient ID`.
