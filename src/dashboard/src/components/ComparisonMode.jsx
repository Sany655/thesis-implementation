import { GitCompare } from 'lucide-react';

const ComparisonMode = ({ assessment }) => {
  const { pediatric, adult } = assessment;

  const topPedFeature = [...pediatric.features].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))[0];
  const topAdultFeature = [...adult.features].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))[0];

  return (
    <div className="card">
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
        <GitCompare color="var(--primary)" />
        <h2 style={{ margin: 0 }}>Pediatric vs Adult Comparison Mode</h2>
      </div>

      <p style={{ color: 'var(--text-light)', marginBottom: '1.5rem' }}>
        This section demonstrates why age-specific models are necessary. The same patient characteristics receive different model contributions and risk predictions depending on the age-specific model used.
      </p>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ backgroundColor: 'var(--secondary)' }}>
              <th style={{ padding: '1rem', borderBottom: '2px solid #e5e7eb' }}>Metric</th>
              <th style={{ padding: '1rem', borderBottom: '2px solid #e5e7eb' }}>Pediatric Model</th>
              <th style={{ padding: '1rem', borderBottom: '2px solid #e5e7eb' }}>Adult Model</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={{ padding: '1rem', borderBottom: '1px solid #e5e7eb', fontWeight: '500' }}>Algorithm</td>
              <td style={{ padding: '1rem', borderBottom: '1px solid #e5e7eb' }}>XGBoost</td>
              <td style={{ padding: '1rem', borderBottom: '1px solid #e5e7eb' }}>XGBoost</td>
            </tr>
            <tr>
              <td style={{ padding: '1rem', borderBottom: '1px solid #e5e7eb', fontWeight: '500' }}>Predicted Risk</td>
              <td style={{ padding: '1rem', borderBottom: '1px solid #e5e7eb', color: pediatric.probability > 50 ? 'var(--danger)' : 'var(--success)', fontWeight: 'bold' }}>
                {Math.round(pediatric.probability)}%
              </td>
              <td style={{ padding: '1rem', borderBottom: '1px solid #e5e7eb', color: adult.probability > 50 ? 'var(--danger)' : 'var(--success)', fontWeight: 'bold' }}>
                {Math.round(adult.probability)}%
              </td>
            </tr>
            <tr>
              <td style={{ padding: '1rem', borderBottom: '1px solid #e5e7eb', fontWeight: '500' }}>Top Feature</td>
              <td style={{ padding: '1rem', borderBottom: '1px solid #e5e7eb' }}>{topPedFeature?.name || 'N/A'}</td>
              <td style={{ padding: '1rem', borderBottom: '1px solid #e5e7eb' }}>{topAdultFeature?.name || 'N/A'}</td>
            </tr>
            <tr>
              <td style={{ padding: '1rem', borderBottom: '1px solid #e5e7eb', fontWeight: '500' }}>SHAP Contribution</td>
              <td style={{ padding: '1rem', borderBottom: '1px solid #e5e7eb' }}>
                <span className={topPedFeature?.contribution > 0 ? "badge badge-positive" : "badge badge-negative"}>
                  {topPedFeature?.contribution > 0 ? '+' : ''}{topPedFeature?.contribution}
                </span>
              </td>
              <td style={{ padding: '1rem', borderBottom: '1px solid #e5e7eb' }}>
                <span className={topAdultFeature?.contribution > 0 ? "badge badge-positive" : "badge badge-negative"}>
                  {topAdultFeature?.contribution > 0 ? '+' : ''}{topAdultFeature?.contribution}
                </span>
              </td>
            </tr>
            <tr>
              <td style={{ padding: '1rem', borderBottom: '1px solid #e5e7eb', fontWeight: '500' }}>Relevant Split</td>
              <td style={{ padding: '1rem', borderBottom: '1px solid #e5e7eb' }}>{topPedFeature?.threshold || 'N/A'}</td>
              <td style={{ padding: '1rem', borderBottom: '1px solid #e5e7eb' }}>{topAdultFeature?.threshold || 'N/A'}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ComparisonMode;
