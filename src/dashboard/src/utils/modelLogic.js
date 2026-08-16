// ─── API base URL ─────────────────────────────────────────────────────────────
// For deployment: set VITE_API_URL env var. Falls back to localhost for dev.
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Parses a backend error response into a readable message string.
 */
async function parseErrorMessage(response) {
  try {
    const body = await response.json();
    return body?.detail || `Server error (${response.status})`;
  } catch {
    return `Server error (${response.status}: ${response.statusText})`;
  }
}

/**
 * POST /predict
 * Returns the full prediction result, or throws with a descriptive message.
 * Includes plt_context and skipped fields for UI handling.
 */
export const calculateRisk = async (data) => {
  let response;
  try {
    response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        patient_id: data.patient_id || 'P001',
        age: parseFloat(data.age) || 0,
        gender: parseInt(data.gender) || 0,
        wbc: parseFloat(data.wbc) || 0,
        hct: parseFloat(data.hct) || 0,
        rbc: parseFloat(data.rbc) || 0,
        lymph: parseFloat(data.lymph) || 0,
        neut: parseFloat(data.neut) || 0,
        alt: parseFloat(data.alt) || 0,
        ast: parseFloat(data.ast) || 0,
        plt: parseFloat(data.plt) || 0,
      }),
    });
  } catch (networkErr) {
    // Network-level failure (backend unreachable, CORS, etc.)
    throw new Error('Cannot reach the backend server. Is uvicorn running?');
  }

  if (!response.ok) {
    const msg = await parseErrorMessage(response);
    throw new Error(msg);
  }

  const result = await response.json();
  return result;
};

/**
 * GET /assessments/:patientId
 * Returns array of past assessments, or throws with a descriptive message.
 */
export const fetchAssessments = async (patientId) => {
  let response;
  try {
    response = await fetch(`${API_BASE}/assessments/${encodeURIComponent(patientId)}`);
  } catch (networkErr) {
    throw new Error('Cannot reach the backend server. Is uvicorn running?');
  }

  if (!response.ok) {
    const msg = await parseErrorMessage(response);
    throw new Error(msg);
  }

  const result = await response.json();
  return result;
};

/**
 * GET /patients
 * Returns array of patients in the system.
 */
export const fetchPatients = async () => {
  let response;
  try {
    response = await fetch(`${API_BASE}/patients`);
  } catch (networkErr) {
    throw new Error('Cannot reach the backend server. Is uvicorn running?');
  }

  if (!response.ok) {
    const msg = await parseErrorMessage(response);
    throw new Error(msg);
  }

  const result = await response.json();
  return result;
};
