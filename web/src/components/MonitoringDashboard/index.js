import React, { useState, useEffect, useCallback } from 'react';
import { Container, Row, Col, Card, Button, Spinner, Alert } from 'react-bootstrap';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  TimeScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import 'chartjs-adapter-date-fns'; // Import date adapter for Chart.js
import styles from './MonitoringDashboard.module.css';
import useVitalSignStream from '../../hooks/useVitalSignStream';
import { triggerEmergency } from '../../services/api';

// Register Chart.js components
ChartJS.register(
  TimeScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const VitalSignGraphs = ({ vitalSignData }) => {
  const chartOptions = {
    responsive: true,
    animation: {
        duration: 0 // Disable animation for smoother real-time updates
    },
    scales: {
      x: {
        type: 'time',
        time: {
          unit: 'second',
          displayFormats: {
            second: 'HH:mm:ss'
          }
        },
        title: {
          display: true,
          text: 'Time'
        }
      },
      y: {
        beginAtZero: false,
        title: {
          display: true,
          text: 'Value'
        }
      }
    },
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: false, // Title will be in Card.Header
      },
    },
    maintainAspectRatio: false, // Allow charts to fit their containers
  };

  const vitalSignsToDisplay = Object.keys(vitalSignData);

  return (
    <>
      {vitalSignsToDisplay.length > 0 ? (
        vitalSignsToDisplay.map((signName, index) => {
          const data = {
            labels: vitalSignData[signName].map(d => new Date(d.timestamp)),
            datasets: [
              {
                label: signName.replace('_', ' '),
                data: vitalSignData[signName].map(d => d.value),
                borderColor: `hsl(${index * 30}, 70%, 50%)`, // Adjusted hue for more distinct colors
                backgroundColor: `hsla(${index * 30}, 70%, 50%, 0.2)`,
                tension: 0.1,
                pointRadius: 0,
              },
            ],
          };

          return (
            <Card key={signName} className="mb-3">
              <Card.Header className="text-capitalize">{signName.replace('_', ' ')}</Card.Header>
              <Card.Body style={{ height: '200px' }}>
                <Line
                  id={`chart-dash-${signName}`}
                  data={data}
                  options={{
                    ...chartOptions,
                  }}
                />
              </Card.Body>
            </Card>
          );
        })
      ) : (
        <Card className="mb-3">
          <Card.Header>Vital Signs</Card.Header>
          <Card.Body>
            <p className="text-center">Waiting for vital sign data...</p>
          </Card.Body>
        </Card>
      )}
    </>
  );
};

const EmergencyButtons = ({ onTriggerEmergency }) => {
  const emergencies = [
    "Heart Attack", "Stroke", "Anaphylaxis", "Asthma Attack", "Diabetic Ketoacidosis",
    "Hypoglycemia", "Sepsis", "Pulmonary Embolism", "Hypertensive Crisis", "Status Epilepticus",
    "Heat Stroke", "Hypothermia"
  ];
  return (
    <div className="d-flex flex-wrap gap-2">
      {emergencies.map((e, index) => (
        <Button
          key={index}
          variant="outline-danger"
          onClick={() => onTriggerEmergency(index + 1)} // Assuming state_code 1-12
        >
          {e}
        </Button>
      ))}
    </div>
  );
};

const StatusIndicator = ({ status, emergencyName }) => {
  let statusClass = '';
  let displayText = '';

  switch (status) {
    case 'stable':
      statusClass = styles.statusStable;
      displayText = 'Stable';
      break;
    case 'monitor':
      statusClass = `${styles.statusMonitor} ${styles.blink}`;
      displayText = 'Monitor';
      break;
    case 'emergency':
      statusClass = styles.statusEmergency;
      displayText = `EMERGENCY: ${emergencyName || 'Unknown'}`;
      break;
    default:
      statusClass = styles.statusStable; // Default to stable if unknown
      displayText = 'Unknown Status';
  }

  return (
    <Card className={`text-center ${statusClass} mb-3`}>
      <Card.Body>
        <h4 className="mb-0">{displayText}</h4>
      </Card.Body>
    </Card>
  );
};

const EmergencyReport = ({ reportData }) => {
  if (!reportData) return null;
  return (
    <Card className="border-danger">
      <Card.Header className="bg-danger text-white">Emergency Report</Card.Header>
      <Card.Body>
        <p><strong>Detected Emergency:</strong> {reportData.risk_name}</p>
        <p><strong>Confidence:</strong> {(reportData.confidence * 100).toFixed(2)}%</p>
        <p><strong>Rationale:</strong> {reportData.rationale || 'No rationale provided.'}</p>
        {/* Add more details from reportData if available and relevant */}
      </Card.Body>
    </Card>
  );
};


const MonitoringDashboard = () => {
  const { vitalSignData, predictionData, loading, error, initializeSimulation } = useVitalSignStream();
  const [patientInitialized, setPatientInitialized] = useState(false);

  // Default patient profile for initialization
  const defaultPatientProfile = {
    name: "Simulated Patient",
    age: 45,
    gender: "male",
    activity_level: "moderate",
    primary_condition: "Healthy"
  };

  useEffect(() => {
    // Initialize simulation when component mounts, only once
    if (!patientInitialized) {
      initializeSimulation(defaultPatientProfile)
        .then(() => setPatientInitialized(true))
        .catch(err => console.error("Failed to initialize patient simulation:", err));
    }
  }, [initializeSimulation, patientInitialized]);


  const onTriggerEmergency = useCallback(async (emergencyId) => {
    console.log(`Triggering emergency: ${emergencyId}`);
    try {
      await triggerEmergency(emergencyId);
      console.log(`Emergency ${emergencyId} triggered successfully.`);
      // Optionally reset prediction data or show a temporary message
    } catch (err) {
      console.error(`Failed to trigger emergency ${emergencyId}:`, err);
      alert(`Failed to trigger emergency: ${err.message}`);
    }
  }, []);

  // Determine current status based on predictionData
  const currentStatus = predictionData ? predictionData.status : 'stable';
  let emergencyName = null;
  let emergencyReport = null;

  if (predictionData && predictionData.status === 'emergency' && predictionData.confidence > 0.85) {
    emergencyName = predictionData.risk_name;
    emergencyReport = predictionData; // Pass the whole prediction object as report data
  } else if (predictionData && predictionData.status === 'monitor') {
    // Do nothing for emergencyName/Report, just show monitor status
  } else {
    // Stable or other low confidence state
  }

  return (
    <Container fluid className="py-4">
      <h1 className="mb-4 text-center">Real-time Patient Monitoring</h1>
      <Row>
        {/* Left Panel: Vital Sign Graphs */}
        <Col md={8}>
          {loading && !patientInitialized && (
            <div className="text-center">
              <Spinner animation="border" role="status">
                <span className="visually-hidden">Loading simulation...</span>
              </Spinner>
              <p className="mt-3">Initializing simulation and fetching vital signs...</p>
            </div>
          )}
          {error && (
            <Alert variant="danger">
              Error: {error}
              <p>Please ensure the backend server is running and accessible at {process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000'}.</p>
            </Alert>
          )}
          {!loading && !error && <VitalSignGraphs vitalSignData={vitalSignData} />}
          <h3 className="mt-4">Trigger Emergency Simulations:</h3>
          <EmergencyButtons onTriggerEmergency={onTriggerEmergency} />
        </Col>

        {/* Right Panel: Status Indicator and Emergency Report */}
        <Col md={4}>
          <StatusIndicator status={currentStatus} emergencyName={emergencyName} />
          {emergencyReport && <EmergencyReport reportData={emergencyReport} />}
        </Col>
      </Row>
    </Container>
  );
};

export default MonitoringDashboard;

