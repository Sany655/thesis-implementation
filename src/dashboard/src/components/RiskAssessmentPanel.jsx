import { Download, AlertTriangle, Activity } from 'lucide-react';

const RiskAssessmentPanel = ({ assessment, onDownload }) => {
  const { activeModel, pediatric, adult, cohort } = assessment;
  const currentModelData = activeModel === 'Pediatric' ? pediatric : adult;

  const isHighRisk = currentModelData.probability > 50;

  return (
    <div className="card" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
        <h2 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Activity color="var(--primary)" />
          Risk Assessment
        </h2>
        <span className="badge" style={{ backgroundColor: 'var(--primary)', color: 'white' }}>
          {cohort} Model Active
        </span>
      </div>

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
        <div className="probability-circle" style={{ '--percentage': currentModelData.probability, '--danger': isHighRisk ? '#ef4444' : '#10b981' }}>
          <div className="probability-circle-content">
            <h3>{Math.round(currentModelData.probability)}%</h3>
          </div>
        </div>

        <h3 style={{ marginTop: '1.5rem', fontSize: '1.5rem', color: isHighRisk ? 'var(--danger)' : 'var(--success)' }}>
          {currentModelData.prediction}
        </h3>
      </div>

      <div style={{ marginTop: '2rem' }}>
        <button className="btn btn-primary" style={{ width: '100%' }} onClick={onDownload}>
          <Download size={18} /> Generate PDF Report
        </button>
      </div>
    </div>
  );
};

export default RiskAssessmentPanel;
