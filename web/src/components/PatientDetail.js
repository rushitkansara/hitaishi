import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { getStatusText } from '../data/mockData';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

// Define vital names and emergency classes for UI
const VITAL_NAMES = [
  'Heart Rate', 'Systolic BP', 'Diastolic BP', 'SpO2',
  'Temperature', 'Respiratory Rate', 'Blood Glucose'
];

const EMERGENCY_CLASSES = {
  2: 'Heart_Attack',
  3: 'Arrhythmia',
  4: 'Heart_Failure',
  5: 'Hypoglycemia',
  6: 'Hyperglycemia_DKA',
  7: 'Respiratory_Distress',
  8: 'Sepsis',
  9: 'Stroke',
  10: 'Shock',
  11: 'Hypertensive_Crisis',
  12: 'Fall_Unconscious'
};

const PatientDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [patient, setPatient] = useState(null);
  const [vitalData, setVitalData] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [alertStatus, setAlertStatus] = useState(null);
  const [simulationSequence, setSimulationSequence] = useState(null);
  const sequenceIndexRef = useRef(0);
  const [activityGraphData, setActivityGraphData] = useState([]);

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

  // Load patient data from localStorage
  useEffect(() => {
    const storedPatients = JSON.parse(localStorage.getItem('patients')) || [];
    const foundPatient = storedPatients.find(p => p.id === parseInt(id));
    if (foundPatient) {
      setPatient(foundPatient);
      // Initialize vital data with empty arrays for 60 seconds
      const initialData = {};
      VITAL_NAMES.forEach((name, index) => {
        initialData[index] = Array(60).fill(
          foundPatient.baselines ? foundPatient.baselines[index] : 0
        );      });
      setVitalData(initialData);
    } else {
      navigate('/'); // Redirect if patient not found
    }
  }, [id, navigate]);

  // Main simulation loop
  useEffect(() => {
    if (patient) {
      fetchAndStartSequence(); // Initial fetch
    }

    const tick = () => {
      setSimulationSequence(currentSequence => {
        if (currentSequence) {
          const currentIndex = sequenceIndexRef.current;
          const nextDataPoint = currentSequence[currentIndex];

          if (nextDataPoint) {
            setVitalData(prevGraphData => {
              const updatedGraphData = {};
              VITAL_NAMES.forEach((name, index) => {
                const newSeries = [...(prevGraphData[index] || []), nextDataPoint[index]];
                if (newSeries.length > 60) newSeries.shift();
                updatedGraphData[index] = newSeries;
              });
              return updatedGraphData;
            });
          }

          // Advance index or loop/fetch new data
          if (currentIndex >= currentSequence.length - 1) {
            fetchAndStartSequence(); // Fetch next block of stable data
          } else {
            sequenceIndexRef.current += 1;
          }
        }
        return currentSequence; // Return sequence unchanged for next tick
      });
    };

    const intervalId = setInterval(tick, 1000);

    return () => clearInterval(intervalId);
  }, [patient]);

  useEffect(() => {
    if (patient && patient.total_steps !== undefined) {
      const data = Array.from({ length: 9 }, (_, i) => {
        if (i === 0) return 0;
        if (i === 8) return patient.total_steps;
        const remainingSteps = patient.total_steps - (patient.total_steps / 8 * i);
        const randomFactor = Math.random() * 0.5 + 0.75;
        return Math.min(patient.total_steps, Math.round(patient.total_steps / 8 * i * randomFactor));
      });
      setActivityGraphData(data);
    }
  }, [patient]);



  const fetchAndStartSequence = async () => {
    if (!patient) return;
    try {
      const response = await fetch(`${BACKEND_URL}/generate_stable_data`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: patient.name,
          age: patient.age,
          gender: patient.gender,
          activity_level: patient.activity_level,
          primary_condition: patient.primary_condition
        }),
      });
      const data = await response.json();
      setSimulationSequence(data.sequence);
      sequenceIndexRef.current = 0; // Reset index for new sequence
      setPrediction(null);
      setAlertStatus(null);
    } catch (error) {
      console.error("Error fetching stable data:", error);
    }
  };

  const startStableDataLoop = () => {
    // This function is now simplified, the main loop is in useEffect
    fetchAndStartSequence();
  };

  const handleSimulateEmergency = async (emergencyType) => {
    // No need to clear interval, the main loop will handle the new sequence
    setPrediction(null);
    setAlertStatus(null);

    try {
      const response = await fetch(`${BACKEND_URL}/simulate_emergency`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_profile: {
            name: patient.name,
            age: patient.age,
            gender: patient.gender,
            activity_level: patient.activity_level,
            primary_condition: patient.primary_condition
          },
          emergency_type: emergencyType,
          emergency_contact: patient.emergency_contact
        }),
      });
      const result = await response.json();

      // Set the emergency sequence and let the tick loop handle rendering
      setSimulationSequence(result.sequence);
      sequenceIndexRef.current = 0;

      // Display prediction and alert *after* the simulation has finished displaying
      setTimeout(() => {
        setPrediction(result.prediction);
        setAlertStatus(result.alert_status);
        // After the alert, go back to fetching stable data
        fetchAndStartSequence();
      }, result.sequence.length * 1000); // Wait for the full sequence to play out

    } catch (error) {
      console.error("Error simulating emergency:", error);
      alert('Failed to simulate emergency.');
      fetchAndStartSequence(); // Revert to stable loop on error
    }
  };

  const handleDeletePatient = () => {
    if (window.confirm('Are you sure you want to delete this patient?')) {
      const storedPatients = JSON.parse(localStorage.getItem('patients')) || [];
      const updatedPatients = storedPatients.filter(p => p.id !== parseInt(id));
      localStorage.setItem('patients', JSON.stringify(updatedPatients));
      navigate('/');
    }
  };

  if (!patient) {
    return (
      <div className="container mt-5">
        <div className="alert alert-danger">
          Patient not found
        </div>
        <button 
          className="btn btn-primary"
          onClick={() => navigate('/')}
        >
          ← Back to Dashboard
        </button>
      </div>
    );
  }

  const getStatusClass = (riskName) => {
    if (riskName === 'Stable') return 'status-stable';
    if (riskName === 'Monitor') return 'status-warning';
    return 'status-critical';
  };

  const VITAL_CONFIG = {
    0: { name: 'Heart Rate', unit: 'bpm', precision: 0 },
    1: { name: 'Systolic BP', unit: 'mmHg', precision: 0 },
    2: { name: 'Diastolic BP', unit: 'mmHg', precision: 0 },
    3: { name: 'SpO2', unit: '%', precision: 0 },
    4: { name: 'Temperature', unit: '°C', precision: 1 },
    5: { name: 'Respiratory Rate', unit: 'breaths/min', precision: 0 },
    6: { name: 'Blood Glucose', unit: 'mg/dL', precision: 0 },
  };

  const generateChartOptions = (vitalIndex) => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {
      legend: { display: false },
      tooltip: { enabled: false },
    },
    scales: {
      x: { display: false },
      y: { 
        display: true,
        ticks: {
          precision: VITAL_CONFIG[vitalIndex].precision
        },
        title: {
          display: true,
          text: VITAL_CONFIG[vitalIndex].unit
        }
      },
    },
  });

  return (
    <div className="container-fluid">
      {/* Header */}
      <div className="dashboard-header">
        <div className="container">
          <div className="d-flex justify-content-between align-items-center">
            <div>
              <h1 className="dashboard-title">Patient Details</h1>
              <p className="dashboard-subtitle">Real-time Health Monitoring</p>
            </div>
            <a 
              href="/" 
              className="back-button"
              onClick={(e) => {
                e.preventDefault();
                navigate('/');
              }}
            >
              ← Back to Dashboard
            </a>
          </div>
        </div>
      </div>

      <div className="container">
        {/* Patient Profile */}
        <div className="patient-detail-header">
          <div className="patient-profile">
            <div className="patient-avatar">
              {patient.name.split(' ').map(n => n[0]).join('')}
            </div>
            <div className="patient-info">
              <h2>{patient.name}</h2>
              <p><strong>Age:</strong> {patient.age} years</p>
              <p><strong>Gender:</strong> {patient.gender}</p>
              <p><strong>Condition:</strong> {patient.primary_condition}</p>
              <p><strong>Emergency Contact:</strong> {patient.emergency_contact}</p>
              <p><strong>Patient ID:</strong> #{patient.id.toString().padStart(4, '0')}</p>
              {prediction && (
                <span className={`status-indicator ${getStatusClass(prediction.risk_name)}`}>
                  {getStatusText(prediction.risk_name)}
                </span>
              )}
              {alertStatus && <p className="mt-2">SMS Alert Status: <strong>{alertStatus}</strong></p>}
            </div>
            <div className="patient-actions">
                <button className="btn btn-danger" onClick={handleDeletePatient}>Delete Patient</button>
            </div>
          </div>
        </div>

        {/* Emergency Simulation Buttons */}
        <div className="my-4 text-center">
          <h3>Simulate Emergency:</h3>
          <div className="d-flex flex-wrap justify-content-center gap-2">
            {Object.entries(EMERGENCY_CLASSES).map(([label, name]) => (
              <button 
                key={label} 
                className="btn btn-danger btn-sm"
                onClick={() => handleSimulateEmergency(parseInt(label))}
              >
                {name}
              </button>
            ))}
          </div>
        </div>

        {/* Vital Charts */}
        <div className="row">
          {VITAL_NAMES.map((vitalName, index) => (
            <div key={vitalName} className="col-lg-6 col-md-12 mb-4">
              <div className="chart-container">
                <h3 className="chart-title">{vitalName} Trend</h3>
                <div className="chart-wrapper">
                  <Line 
                    data={{
                      labels: Array.from({ length: 60 }, (_, i) => i + 1),
                      datasets: [
                        {
                          label: vitalName,
                          data: vitalData[index],
                          borderColor: '#007bff',
                          backgroundColor: 'rgba(0, 123, 255, 0.1)',
                          borderWidth: 2,
                          pointRadius: 0,
                          tension: 0.4,
                        },
                      ],
                    }}
                    options={generateChartOptions(index)}
                  />
                </div>
              </div>
            </div>
          ))}

          {/* Activity Graph Card - Placed after Blood Glucose (index 6) */}
          {patient.total_steps !== undefined && (
            <div key="activity-graph" className="col-lg-6 col-md-12 mb-4">
              <div className="chart-container">
                <h3 className="chart-title">Activity (Last 24h)</h3>
                <div className="chart-wrapper">
                  <Line 
                    data={{
                      labels: ['0h', '3h', '6h', '9h', '12h', '15h', '18h', '21h', '24h'],
                      datasets: [
                        {
                          label: 'Steps',
                          data: activityGraphData, // Use the state variable here
                          borderColor: '#28a745',
                          backgroundColor: 'rgba(40, 167, 69, 0.1)',
                          borderWidth: 2,
                          pointRadius: 3,
                          tension: 0.4,
                        },
                      ],
                    }}

                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: { display: false },
                        tooltip: { mode: 'index', intersect: false },
                      },
                      scales: {
                        x: { 
                          title: {
                            display: true,
                            text: 'Time'
                          }
                        },
                        y: { 
                          min: 0,
                          max: 15000, // Max possible steps
                          title: {
                            display: true,
                            text: 'Steps'
                          }
                        },
                      },
                    }}
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Prediction Result */}
        {prediction && (
          <div className="alert alert-info mt-4 text-center">
            <h4>Model Prediction:</h4>
            <p>Condition: <strong>{prediction.risk_name}</strong></p>
            <p>Confidence: {prediction.confidence.toFixed(2)}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default PatientDetail;
