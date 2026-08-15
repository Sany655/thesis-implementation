# Dengue Clinical Risk Dashboard — Deployment Guide

---

## ⚠️ Why Vercel Fails for Full Python ML Backends (500 MB Limit)

When deploying Python serverless functions on Vercel, the platform packages all Python libraries into an AWS Lambda container with a **strict hard limit of 500 MB uncompressed**:
- `xgboost` + `shap` + `scikit-learn` + `scipy` + `pandas` + `llvmlite` = **1640.37 MB** (exceeds Vercel's 500 MB limit).

### The Solution: Decoupled Production Architecture (Industry Standard)
- **Frontend (Vite + React)**: Hosted on **Vercel** (Blazing fast global edge network, 0 MB ML overhead, free).
- **Backend (FastAPI + XGBoost + SHAP)**: Hosted on **Render**, **Railway**, **Fly.io**, or **Hugging Face Spaces** (Full container support, no bundle size limits, persistent database).

```
          ┌─────────────────────────┐
          │   React + Vite Client   │
          │   (Hosted on Vercel)    │
          └───────────┬─────────────┘
                      │
                      │ HTTPS API Calls (VITE_API_URL)
                      ▼
          ┌─────────────────────────┐
          │   FastAPI ML Backend    │
          │  (Render / Railway / HF)│
          └───────────┬─────────────┘
                      │
          ┌───────────▼─────────────┐
          │   SQLite / PostgreSQL   │
          │  (Assessment History)   │
          └─────────────────────────┘
```

---

## 🚀 Step 1: Deploy Backend (2 Minutes on Render — Free)

1. Push your repository to **GitHub**.
2. Go to [Render Dashboard](https://dashboard.render.com/) and click **New +** → **Web Service**.
3. Select your repository and configure the settings:
   - **Name**: `dengue-risk-api`
   - **Region**: Oregon (US West) or Frankfurt (EU)
   - **Root Directory**: `src/backend` (or `backend`)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: `Free`
4. Expand **Environment Variables** and add:
   - `DATABASE_URL` = `sqlite+aiosqlite:///./dengue_dashboard.db`
   - `MODEL_DIR` = `../models`
5. Click **Create Web Service**.
6. Once deployed, copy your live backend URL (e.g., `https://dengue-risk-api.onrender.com`).

> **Tip**: Verify your backend is running by opening `https://dengue-risk-api.onrender.com/health` in your browser. It should return:
> `{"status":"ok","models_loaded":["Pediatric","Adult"],"db":"connected"}`

---

## ⚡ Step 2: Deploy Frontend on Vercel

1. Go to [vercel.com/new](https://vercel.com/new) and import your GitHub repository.
2. Configure the project settings:
   - **Framework Preset**: `Vite`
   - **Root Directory**: Click `Edit` and select `src/dashboard` (or `dashboard`)
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
3. Expand **Environment Variables** and add:
   - **Key**: `VITE_API_URL`
   - **Value**: `https://dengue-risk-api.onrender.com` *(your live Render backend URL from Step 1)*
4. Click **Deploy**.

---

## 🧪 Step 3: Verification Checklist

1. **Check Backend Health**:
   ```bash
   curl https://<your-render-api>.onrender.com/health
   # Returns: {"status":"ok","models_loaded":["Pediatric","Adult"],"db":"connected"}
   ```

2. **Check Dashboard in Browser**:
   - Open your deployed Vercel URL (`https://your-dashboard.vercel.app`).
   - Enter patient parameters (e.g. Age: 25, WBC: 3.0, PLT: 40) and click **Run & Store Assessment Snapshot**.
   - Confirm risk score, SHAP waterfall chart, and timeline update in real-time.
