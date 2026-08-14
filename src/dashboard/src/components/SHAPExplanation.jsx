import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';
import { HelpCircle } from 'lucide-react';

const SHAPExplanation = ({ assessment }) => {
  const { activeModel, pediatric, adult } = assessment;
  const currentModelData = activeModel === 'Pediatric' ? pediatric : adult;
  
  // Sort features by absolute contribution to show most impactful at the top
  const sortedFeatures = [...currentModelData.features].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div style={{ background: '#fff', padding: '10px', border: '1px solid #ccc', borderRadius: '4px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
          <p><strong>{data.name}</strong></p>
          <p>Value: {data.val}</p>
          <p>Split: {data.threshold}</p>
          <p>Impact: {data.contribution > 0 ? 'Increased Risk' : 'Reduced Risk'}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="card">
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
        <HelpCircle color="var(--primary)" />
        <h2 style={{ margin: 0 }}>Why this prediction? (SHAP Explanation)</h2>
      </div>
      
      <p style={{ color: 'var(--text-light)', marginBottom: '1.5rem' }}>
        This panel displays model-derived SHAP splits as contextual research findings. It explains how each patient feature contributed to the model's prediction. Features pushing the risk higher are shown in red (to the right), and features reducing risk are shown in green (to the left). Note: probability comes strictly from the trained model and is never calculated from the SHAP split.
      </p>

      <div style={{ height: 300, width: '100%' }}>
        <ResponsiveContainer>
          <BarChart
            data={sortedFeatures}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <XAxis type="number" domain={[-0.5, 0.5]} />
            <YAxis dataKey="name" type="category" width={100} />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine x={0} stroke="#000" />
            <Bar dataKey="contribution" barSize={20}>
              {sortedFeatures.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.contribution > 0 ? '#ef4444' : '#10b981'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid-2" style={{ marginTop: '2rem' }}>
        <div>
          <h4 style={{ color: 'var(--danger)', marginBottom: '0.5rem' }}>Increasing Risk</h4>
          <ul className="feature-list">
            {sortedFeatures.filter(f => f.contribution > 0).map((f, i) => (
              <li key={i} className="feature-item">
                <span>{f.name} ({f.val})</span>
                <span className="badge badge-positive">+{f.contribution}</span>
              </li>
            ))}
            {sortedFeatures.filter(f => f.contribution > 0).length === 0 && (
              <li className="feature-item" style={{ color: 'var(--text-light)' }}>None</li>
            )}
          </ul>
        </div>
        <div>
          <h4 style={{ color: 'var(--success)', marginBottom: '0.5rem' }}>Reducing Risk</h4>
          <ul className="feature-list">
            {sortedFeatures.filter(f => f.contribution <= 0).map((f, i) => (
              <li key={i} className="feature-item">
                <span>{f.name} ({f.val})</span>
                <span className="badge badge-negative">{f.contribution}</span>
              </li>
            ))}
            {sortedFeatures.filter(f => f.contribution <= 0).length === 0 && (
              <li className="feature-item" style={{ color: 'var(--text-light)' }}>None</li>
            )}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default SHAPExplanation;
