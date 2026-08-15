# Dengue Clinical Risk Dashboard — Deployment Guide

This guide provides end-to-end instructions for deploying the **React + Vite Dashboard** and the **FastAPI Machine Learning Backend** together on a **single Vercel domain** (or via a decoupled architecture).

---

## 🏗️ Architecture: Unified Single-Domain Deployment on Vercel

```
                                  https://your-app.vercel.app
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
          Static Client Assets                            Serverless API Function
       (dashboard/dist/index.html)                            (api/index.py)
                      │                                               │
             React 19 + Recharts                             FastAPI Backend
          • Risk Assessment Form                          • /predict & /api/predict
          • Waterfall & Radar Charts                      • /assessments & /api/assessments
          • Patient Timeline & Deltas                     • /health & /api/health
                                                                      │
                                                          XGBoost + SHAP Tree Models
                                                         (pediatric_best / adult_best)
                                                                      │
                                                             SQLite Database
                                                       (/tmp/dengue_dashboard.db)
```

---

## 🚀 How to Deploy on Vercel in 1 Domain

### Option 1: Via Vercel Web Dashboard (GitHub Import)
1. Push your repository to GitHub.
2. Go to [vercel.com/new](https://vercel.com/new) and import your repository.
3. In the project settings:
   - **Framework Preset**: `Vite`
   - **Root Directory**: `src` (if deploying from `src/` folder) or `./` (if root of repo is `src`)
   - **Build Command**: `cd dashboard && npm install && npm run build` *(auto-configured via `vercel.json`)*
   - **Output Directory**: `dashboard/dist` *(auto-configured via `vercel.json`)*
4. **Environment Variables**:
   - `MODEL_DIR`: `models` *(optional, models are automatically bundled in `src/models/`)*
   - Leave `VITE_API_URL` empty so the frontend automatically calls relative `/predict`, `/assessments`, and `/health` routes on the same domain.
5. Click **Deploy**.

---

### Option 2: Via Vercel CLI
```bash
# Install Vercel CLI if needed
npm install -g vercel

# From the src directory:
vercel

# For production deployment:
vercel --prod
```

---

## 🧪 Verification & Health Check

Once deployed, you can verify your single-domain deployment:

1. **API Status & Health**:
   ```bash
   curl https://<your-vercel-domain>.vercel.app/health
   # Returns: {"status":"ok","models_loaded":["Pediatric","Adult"],"db":"connected"}
   ```

2. **Run a Test Prediction**:
   ```bash
   curl -X POST https://<your-vercel-domain>.vercel.app/predict \
     -H "Content-Type: application/json" \
     -d '{"patient_id":"P001","age":30,"gender":0,"wbc":3.2,"hct":44.0,"rbc":4.5,"lymph":18.0,"neut":72.0,"alt":65.0,"ast":80.0,"plt":40.0}'
   ```

3. **Dashboard Web Interface**:
   - Open `https://<your-vercel-domain>.vercel.app` in your browser.
   - Enter patient parameters and click **Run & Store Assessment Snapshot**.
   - Verify risk score, SHAP waterfall explanations, and patient longitudinal history render cleanly without CORS or routing issues.
