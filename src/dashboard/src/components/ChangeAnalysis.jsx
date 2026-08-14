import { TrendingUp } from 'lucide-react';

const ChangeAnalysis = ({ assessments }) => {
  if (!assessments || assessments.length < 2) return null;

  const current = assessments[assessments.length - 1];
  const previous = assessments[assessments.length - 2];

  const currentModelData = current.activeModel === 'Pediatric' ? current.pediatric : current.adult;
  const previousModelData = previous.activeModel === 'Pediatric' ? previous.pediatric : previous.adult;

  const fields = ['age', 'wbc', 'hct', 'rbc', 'lymph', 'neut', 'alt', 'ast'];
  const labels = {
    age: 'Age', wbc: 'WBC', hct: 'HCT', rbc: 'RBC', lymph: 'Lymphocyte %', neut: 'Neutrophil %', alt: 'ALT', ast: 'AST'
  };

  const calculateChange = (prev, curr) => {
    const p = parseFloat(prev);
    const c = parseFloat(curr);
    if (isNaN(p) || isNaN(c) || p === 0) return 'N/A';
    const percentChange = ((c - p) / p) * 100;
    return `${percentChange > 0 ? '+' : ''}${percentChange.toFixed(1)}%`;
  };

  const calculateRawChange = (prev, curr) => {
    const p = parseFloat(prev);
    const c = parseFloat(curr);
    if (isNaN(p) || isNaN(c)) return 0;
    return Math.abs(c - p);
  };

  // Find largest input changes
  const changes = fields.map(field => ({
    name: labels[field],
    rawDiff: calculateRawChange(previous.input[field], current.input[field])
  })).sort((a, b) => b.rawDiff - a.rawDiff);

  const topChanges = changes.filter(c => c.rawDiff > 0).slice(0, 3).map(c => c.name);

  return (
    <div className="card" style={{ marginTop: '2rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
        <TrendingUp color="var(--primary)" />
        <h2 style={{ margin: 0 }}>Change Analysis (Current vs Previous)</h2>
      </div>

      <div style={{ padding: '1rem', background: 'var(--secondary)', borderRadius: '8px', marginBottom: '1.5rem' }}>
        <h3 style={{ margin: 0 }}>
          Model prediction changed from {Math.round(previousModelData.probability)}% to {Math.round(currentModelData.probability)}%.
        </h3>
        {topChanges.length > 0 && (
          <p style={{ marginTop: '0.5rem', color: 'var(--text-dark)' }}>
            <strong>Largest input changes:</strong> {topChanges.join(', ')}
          </p>
        )}
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ backgroundColor: 'var(--secondary)' }}>
              <th style={{ padding: '0.75rem', borderBottom: '2px solid #e5e7eb' }}>Feature</th>
              <th style={{ padding: '0.75rem', borderBottom: '2px solid #e5e7eb' }}>Previous ({previous.snapshotName || 'Snapshot 1'})</th>
              <th style={{ padding: '0.75rem', borderBottom: '2px solid #e5e7eb' }}>Current ({current.snapshotName || 'Snapshot 2'})</th>
              <th style={{ padding: '0.75rem', borderBottom: '2px solid #e5e7eb' }}>Change</th>
            </tr>
          </thead>
          <tbody>
            {fields.map(field => (
              <tr key={field}>
                <td style={{ padding: '0.75rem', borderBottom: '1px solid #e5e7eb', fontWeight: '500' }}>{labels[field]}</td>
                <td style={{ padding: '0.75rem', borderBottom: '1px solid #e5e7eb' }}>{previous.input[field] || 'N/A'}</td>
                <td style={{ padding: '0.75rem', borderBottom: '1px solid #e5e7eb' }}>{current.input[field] || 'N/A'}</td>
                <td style={{ padding: '0.75rem', borderBottom: '1px solid #e5e7eb' }}>
                  {calculateChange(previous.input[field], current.input[field])}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ChangeAnalysis;
