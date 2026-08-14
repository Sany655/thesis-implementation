import { useState } from 'react';
import { Activity, Download, ShieldAlert, Plus, RefreshCw } from 'lucide-react';
import toast, { Toaster } from 'react-hot-toast';
import { calculateRisk, fetchAssessments } from './utils/modelLogic';
import { generatePDFReport } from './utils/pdfGenerator';
import PatientInputForm from './components/PatientInputForm';
import RiskAssessmentPanel from './components/RiskAssessmentPanel';
import SHAPExplanation from './components/SHAPExplanation';
import ComparisonMode from './components/ComparisonMode';
import PatientTimeline from './components/PatientTimeline';
import ChangeAnalysis from './components/ChangeAnalysis';

function App() {
  const [patientData, setPatientData] = useState({
    patient_id: '',
    age: '',
    gender: '',
    wbc: '',
    hct: '',
    rbc: '',
    lymph: '',
    neut: '',
    alt: '',
    ast: '',
    plt: ''
  });

  const [assessments, setAssessments] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setPatientData(prev => ({ ...prev, [name]: value }));
  };

  const loadHistory = async () => {
    if (patientData.patient_id === '') {
      toast.error("Please enter a Patient ID to load history.");
      return;
    }
    setIsProcessing(true);
    try {
      const historyPromise = fetchAssessments(patientData.patient_id);
      
      toast.promise(historyPromise, {
        loading: 'Loading patient history...',
        success: 'History loaded successfully!',
        error: (err) => `Failed to load history: ${err.message}`
      });

      const history = await historyPromise;
      if (history && history.length > 0) {
        setAssessments(history);
      } else {
        toast('No history found for this Patient ID.', { icon: 'ℹ️' });
        setAssessments([]);
      }
    } catch (err) {
      setAssessments([]);
    } finally {
      setIsProcessing(false);
    }
  };

  const runAssessment = async () => {
    if (patientData.patient_id === '' || patientData.age === '' || patientData.lymph === '' || patientData.gender === '') {
      toast.error("Please enter at least Patient ID, Age, Gender, and Lymphocyte %");
      return;
    }
    
    setIsProcessing(true);
    try {
      const assessmentPromise = calculateRisk(patientData);

      toast.promise(assessmentPromise, {
        loading: 'Running assessment...',
        success: (res) => {
          if (res?.skipped) return '\u26A0\uFE0F Assessment skipped — see platelet warning.';
          if (res?.plt_context?.plt_warning) return '\u26A0\uFE0F Assessment complete — platelet note attached.';
          return 'Assessment completed successfully!';
        },
        error: (err) => `Assessment failed: ${err.message}`
      });

      const result = await assessmentPromise;

      // Show PLT clinical context as a separate warning toast
      if (result?.plt_context?.plt_warning) {
        toast(result.plt_context.plt_message, {
          icon: result.plt_context.plt_level === 'normal' ? '\uD83D\uDEA8' : '\u26A0\uFE0F',
          duration: 8000,
          style: {
            background: result.plt_context.plt_level === 'normal' ? '#7c3aed' : '#d97706',
            color: '#fff',
            maxWidth: '420px',
          },
        });
      }

      // Only fetch & update state if the model actually ran
      if (result && !result.skipped) {
        const history = await fetchAssessments(patientData.patient_id);
        setAssessments(history);
      }
    } catch (err) {
      // Error is surfaced by toast.promise above
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownloadPDF = () => {
    generatePDFReport('report-container');
  };
  
  const currentAssessment = assessments.length > 0 ? assessments[assessments.length - 1] : null;

  return (
    <div className="app-container">
      <Toaster position="top-right" />
      <header className="header no-print">
        <Activity size={48} color="var(--primary)" style={{ margin: '0 auto 1rem' }} />
        <h1>Early-stage Explainable Dengue Risk Assessment Dashboard</h1>
        <p>Research prototype for structured longitudinal data collection</p>
      </header>

      <div className="alert alert-warning no-print">
        <ShieldAlert size={24} style={{ flexShrink: 0 }} />
        <div>
          <strong>Disclaimer:</strong> The dashboard is an early-stage research interface intended for model demonstration and structured data collection. It is not intended for clinical diagnosis, treatment selection, or patient management. Its predictions are based on a retrospective dataset and require independent and prospective validation.
        </div>
      </div>

      <div id="report-container">
        <div className="grid-2">
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <PatientInputForm data={patientData} onChange={handleInputChange} />
            <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
              <button 
                className="btn btn-secondary" 
                style={{ flex: 1 }}
                onClick={loadHistory}
                disabled={isProcessing}
              >
                <RefreshCw size={18} /> Load History
              </button>
              <button 
                className="btn btn-primary" 
                style={{ flex: 2 }}
                onClick={runAssessment}
                disabled={isProcessing}
              >
                <Plus size={18} /> Run & Store Assessment Snapshot
              </button>
            </div>
          </div>
          
          <div>
            {currentAssessment ? (
              <RiskAssessmentPanel 
                assessment={currentAssessment} 
                onDownload={handleDownloadPDF} 
              />
            ) : (
              <div className="card" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center', color: 'var(--text-light)' }}>
                Please enter patient data and click "Run Assessment" to evaluate the current snapshot, or "Load History" to view past assessments.
              </div>
            )}
          </div>
        </div>

        {assessments.length > 1 && (
          <>
            <PatientTimeline assessments={assessments} />
            <ChangeAnalysis assessments={assessments} />
          </>
        )}

        {currentAssessment && (
          <>
            <SHAPExplanation assessment={currentAssessment} />
            <ComparisonMode assessment={currentAssessment} />
          </>
        )}
      </div>
    </div>
  );
}

export default App;
