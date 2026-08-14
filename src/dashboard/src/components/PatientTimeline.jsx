import { Clock } from 'lucide-react';

const PatientTimeline = ({ assessments }) => {
  if (!assessments || assessments.length === 0) return null;

  return (
    <div className="card" style={{ marginTop: '2rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
        <Clock color="var(--primary)" />
        <h2 style={{ margin: 0 }}>Longitudinal Patient Timeline</h2>
      </div>

      <p style={{ color: 'var(--text-light)', marginBottom: '1.5rem', fontStyle: 'italic' }}>
        The dashboard provides longitudinal visualization of repeated laboratory measurements and independently generated model predictions. Temporal prediction is outside the scope of the current model.
      </p>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '2rem 0', position: 'relative' }}>
        {/* Timeline Line */}
        <div style={{ position: 'absolute', top: '50%', left: '5%', right: '5%', height: '4px', background: '#e5e7eb', zIndex: 0 }}></div>

        {assessments.map((assessment, index) => {
          const { activeModel, pediatric, adult, snapshotName } = assessment;
          const currentModelData = activeModel === 'Pediatric' ? pediatric : adult;
          const isHighRisk = currentModelData.probability > 50;

          return (
            <div key={index} style={{ position: 'relative', zIndex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', background: 'var(--card-bg)', padding: '0.5rem' }}>
              <div style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>{snapshotName || `Snapshot ${index + 1}`}</div>
              
              <div style={{
                width: '60px', height: '60px', borderRadius: '50%', 
                background: 'white', border: `4px solid ${isHighRisk ? 'var(--danger)' : 'var(--success)'}`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontWeight: 'bold', fontSize: '1.2rem', color: isHighRisk ? 'var(--danger)' : 'var(--success)'
              }}>
                {Math.round(currentModelData.probability)}%
              </div>
              
              <div style={{ marginTop: '0.5rem', color: isHighRisk ? 'var(--danger)' : 'var(--success)', fontWeight: '500' }}>
                {currentModelData.prediction}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default PatientTimeline;
