import { User } from 'lucide-react';

// ─── Field Definitions ────────────────────────────────────────────────────────
// Units and placeholders are grounded in the actual training dataset
// (Comprehensive Dengue Hematology Dataset, Bangladesh, n=972 NS1+ confirmed cases)
// Dataset medians: WBC=5.18, HCT=37.85, RBC=4.55, Lymph%=25.3,
//                  Neut%=64.9, AST=48, ALT=35, PLT=67 (all ×10³/µL for PLT)
//
// Feature engineering thresholds used in training:
//   wbc_leukopenia  : WBC < 4.0 (×10³/µL)
//   lymph_low       : Lymph % < 20%
//   neut_high       : Neut % > 70%
//   liver_involvement: AST > 40 or ALT > 40 (U/L)
//
// PLT is NOT a model input — used only for clinical pre-check triage.
// PLT < 50 ×10³/µL = Severe proxy target.

const FIELDS = [
  {
    name: 'wbc',
    label: 'WBC',
    unit: '×10³/µL',
    placeholder: '5.2',
    hint: 'Normal: 4–10 · Leukopenia flag: < 4.0',
    step: '0.01',
  },
  {
    name: 'lymph',
    label: 'Lymphocyte',
    unit: '%',
    placeholder: '25.0',
    hint: 'Lymphopenia flag: < 20%',
    step: '0.1',
  },
  {
    name: 'neut',
    label: 'Neutrophil',
    unit: '%',
    placeholder: '65.0',
    hint: 'Neutrophilia flag: > 70%',
    step: '0.1',
  },
  {
    name: 'hct',
    label: 'Hematocrit (HCT)',
    unit: '%',
    placeholder: '38.0',
    hint: 'Normal: 36–50%',
    step: '0.1',
  },
  {
    name: 'rbc',
    label: 'RBC',
    unit: '×10⁶/µL',
    placeholder: '4.5',
    hint: 'Normal: 3.8–6.1',
    step: '0.01',
  },
  {
    name: 'ast',
    label: 'AST',
    unit: 'U/L',
    placeholder: '38',
    hint: 'Liver flag: > 40 U/L · Dataset median: 48',
    step: '1',
  },
  {
    name: 'alt',
    label: 'ALT',
    unit: 'U/L',
    placeholder: '32',
    hint: 'Liver flag: > 40 U/L · Dataset median: 35',
    step: '1',
  },
  {
    name: 'plt',
    label: 'Platelets (PLT)',
    unit: '×10³/µL',
    placeholder: '70',
    hint: 'Severe: < 50 · Moderate: 50–100',
    step: '1',
    isTriage: true,
  },
];

const PatientInputForm = ({ data, onChange }) => {
  return (
    <div className="card">
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
        <User color="var(--primary)" />
        <h2 style={{ margin: 0 }}>Patient Demographics &amp; Labs</h2>
      </div>

      {/* ── Demographics ── */}
      <div className="grid-2" style={{ marginBottom: '1rem' }}>
        <div className="form-group" style={{ gridColumn: '1 / -1' }}>
          <label>Patient ID</label>
          <input
            type="text"
            name="patient_id"
            className="form-control"
            value={data.patient_id}
            onChange={onChange}
            placeholder="e.g. P001"
          />
        </div>

        <div className="form-group">
          <label>Gender</label>
          <select
            name="gender"
            className="form-control"
            value={data.gender}
            onChange={onChange}
          >
            <option value="">Select...</option>
            <option value="0">Male (0)</option>
            <option value="1">Female (1)</option>
          </select>
        </div>

        <div className="form-group">
          <label>
            Age <span style={{ color: 'var(--text-light)', fontWeight: 400, fontSize: '0.8em' }}>(years)</span>
          </label>
          <input
            type="number"
            name="age"
            className="form-control"
            value={data.age}
            onChange={onChange}
            min="1"
            max="120"
            placeholder="e.g. 26"
            title="Age in years. Pediatric: ≤18, Adult: >18"
          />
          <small style={{ color: 'var(--text-light)', fontSize: '0.75rem' }}>
            Pediatric cohort: ≤ 18 · Adult cohort: &gt; 18
          </small>
        </div>
      </div>

      {/* ── Lab Values ── */}
      <div className="grid-2">
        {FIELDS.map(({ name, label, unit, placeholder, hint, step, isTriage }) => (
          <div className="form-group" key={name}>
            <label>
              {label}{' '}
              <span style={{ color: 'var(--text-light)', fontWeight: 400, fontSize: '0.8em' }}>
                ({unit})
              </span>
              {isTriage && (
                <span
                  title="Used for clinical triage only. PLT is excluded from model features to prevent data leakage."
                  style={{
                    marginLeft: '0.4rem',
                    background: 'var(--warning, #d97706)',
                    color: '#fff',
                    fontSize: '0.65rem',
                    padding: '1px 5px',
                    borderRadius: '4px',
                    verticalAlign: 'middle',
                    cursor: 'help',
                  }}
                >
                  TRIAGE ONLY
                </span>
              )}
            </label>
            <input
              type="number"
              name={name}
              className="form-control"
              value={data[name]}
              onChange={onChange}
              step={step}
              min="0"
              placeholder={`e.g. ${placeholder}`}
              title={hint}
            />
            <small style={{ color: 'var(--text-light)', fontSize: '0.75rem' }}>{hint}</small>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PatientInputForm;
