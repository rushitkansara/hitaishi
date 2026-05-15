import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Container, Row, Col, Card, Button, Spinner, Alert, Badge } from 'react-bootstrap';
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
import 'chartjs-adapter-date-fns';
import styles from './PatientMonitorPage.module.css';
import useVitalSignStream from '../hooks/useVitalSignStream';
import { triggerEmergency, getPatient } from '../services/api';
import { useParams, useNavigate } from 'react-router-dom';
import { useNotifications } from '../contexts/NotificationContext';

ChartJS.register(TimeScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const CLUSTER_CONFIG_EMERGENCY_MAP = {
    2: 'Heart Attack',
    3: 'Arrhythmia',
    4: 'Heart Failure',
    5: 'Hypoglycemia',
    6: 'Hyperglycemia',
    7: 'Respiratory Distress',
    8: 'Sepsis',
    9: 'Stroke',
    10: 'Shock',
    11: 'Hypertensive Crisis',
    12: 'Fall Unconscious'
};

const CLUSTER_CONFIG = {
  Cardio: {
    vitals: ['heart_rate', 'systolic_bp', 'diastolic_bp'],
    colors: ['#e91e63', '#2196f3', '#03a9f4'],
    emergencies: ['Heart_Attack', 'Arrhythmia', 'Heart_Failure', 'Shock', 'Hypertensive_Crisis']
  },
  Metabolic: {
    vitals: ['blood_glucose', 'temperature'],
    colors: ['#ff9800', '#9c27b0'],
    emergencies: ['Hypoglycemia', 'Hyperglycemia_DKA', 'Sepsis']
  },
  Respiratory: {
    vitals: ['resp_rate', 'spo2'],
    colors: ['#4caf50', '#009688'],
    emergencies: ['Respiratory_Distress', 'Sepsis']
  }
};

const MedicalStatusBanner = ({ prediction, alertStage }) => {
  const getStatusClass = (stage) => {
    switch(stage) {
      case 'critical': return styles['status-critical'];
      case 'emergency': return styles['status-emergency'];
      case 'monitor': return styles['status-monitor'];
      default: return styles['status-stable'];
    }
  };

  const getStatusText = (stage) => {
    switch(stage) {
      case 'critical': return 'CRITICAL';
      case 'emergency': return 'EMERGENCY';
      case 'monitor': return 'MONITOR';
      default: return 'STABLE';
    }
  };

  return (
    <div className={`${styles.clinicalPanel} mb-4`}>
      <div className={`${styles.statusZone} ${getStatusClass(alertStage)}`}>
        <div className="text-uppercase">{getStatusText(alertStage)}</div>
      </div>
      <div className={styles.reasonZone}>
        <div className={styles.rationaleText}>{prediction.rationale}</div>
        <div className={styles.confidenceIndicator}>CONFIDENCE LEVEL: {Math.round(prediction.confidence * 100)}%</div>
      </div>
    </div>
  );
};

const DualScaleChart = ({ title, data, duration, isLive, unit = 's', hasInitialData }) => {
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
      x: {
        type: 'linear',
        grid: { display: false },
        ticks: {
          callback: function(value) {
            const offset = Math.round(value);
            return offset === 0 ? 'Now' : `${offset}s`;
          },
          autoSkip: true,
          maxRotation: 0
        },
        min: -duration,
        max: 0
      },
      y: { 
        beginAtZero: false,
        grid: { color: '#f0f0f0' },
        ticks: { font: { size: 10 } }
      }
    },
    plugins: {
      legend: { display: isLive, position: 'top', labels: { boxWidth: 10, font: { size: 10 } } },
      tooltip: { enabled: true }
    }
  };

  const chartData = useMemo(() => ({
    datasets: data.map((d) => ({
      label: d.label,
      data: d.points.slice(-(duration + 1)).map((p, idx, arr) => ({
        x: idx - (arr.length - 1),
        y: p.y
      })),
      borderColor: d.color,
      backgroundColor: d.color,
      borderWidth: isLive ? 3 : 1,
      pointRadius: 0,
      tension: 0.3,
      fill: false,
      spanGaps: true
    }))
  }), [data, duration, isLive]);

  return (
    <div className={styles.chartContainer}>
      <div className={styles.chartLabel}>{title}</div>
      <div className={styles.chartWrapper}>
        <Line data={chartData} options={options} />
      </div>
    </div>
  );
};

const SystemCluster = ({ name, config, streamData, emphasized, isDimmed }) => {
  const clusterData = useMemo(() => {
    return config.vitals.map((v, i) => ({
      label: v.replace('_', ' ').toUpperCase(),
      color: config.colors[i],
      points: streamData[v] || []
    }));
  }, [config.vitals, config.colors, streamData]);

  return (
    <div className={`${styles.systemCluster} ${emphasized ? styles.emphasized : ''} ${isDimmed ? styles.dimmed : ''}`}>
      <div className={styles.clusterHeader}>
        <h3>{name} System</h3>
        {emphasized && <Badge bg="danger" className="ms-2">Primary Concern</Badge>}
      </div>
      <div className={styles.clusterCharts}>
        <DualScaleChart title="Live Focus (30 seconds)" data={clusterData} duration={30} isLive={true} unit="s" />
        <DualScaleChart title="Context Trend (600 seconds)" data={clusterData} duration={600} isLive={false} unit="s" />
      </div>
    </div>
  );
};

const PatientMonitorPage = () => {
  const { showInfo, showError } = useNotifications();
  const { patientId } = useParams();
  const navigate = useNavigate();
  const { streamData, loading: simLoading, error: simError, initializeSimulation, triggerEmergencyForPatient, abortSimulation } = useVitalSignStream(patientId);
  const [patient, setPatient] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isTriggering, setIsTriggering] = useState(false);
  const initializedRef = useRef(null);

  if (!streamData) {
    return <div className="p-5 text-center"><Spinner animation="border" /></div>;
  }

  useEffect(() => {
    const fetchPatientData = async () => {
      try {
        setLoading(true);
        const data = await getPatient(patientId);
        setPatient(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    if (patientId) fetchPatientData();
  }, [patientId]);

  useEffect(() => {
    let ignore = false;
    if (patient && initializedRef.current !== patientId) {
      initializedRef.current = patientId;
      initializeSimulation(patient).catch(err => {
        if (!ignore) {
          console.error("Failed to init simulation:", err);
          initializedRef.current = null;
        }
      });
    }
    return () => { ignore = true; };
  }, [patient, initializeSimulation, patientId]);

  const onTriggerEmergency = useCallback(async (emergencyId) => {
    if (isTriggering) return;
    setIsTriggering(true);
    try {
      await triggerEmergencyForPatient(emergencyId);
      showInfo("Emergency triggered. Observing physiological response...");
    } catch (err) {
      showError("Failed to trigger emergency: " + err.message);
    } finally {
      setIsTriggering(false);
    }
  }, [triggerEmergencyForPatient, showInfo, showError, isTriggering]);

  const onAbortEmergency = useCallback(async () => {
    try {
        await abortSimulation(patientId);
        showInfo("Simulation aborted. Returning to stable monitoring.");
    } catch (err) {
        showError("Failed to abort simulation: " + err.message);
    }
  }, [patientId, abortSimulation, showInfo, showError]);

  // ... inside PatientMonitorPage render
  console.log("DEBUG: Current streamData object:", streamData);
  const currentLabel = streamData.prediction?.label || 'Stable';
  const emphasizedSystem = useMemo(() => {
    for (const [sys, config] of Object.entries(CLUSTER_CONFIG)) {
      if (config.emergencies.includes(currentLabel)) return sys;
    }
    return null;
  }, [currentLabel]);

  const isSimActive = (streamData.mode === 'transitioning' || streamData.mode === 'emergency' || !!streamData.simulation_active);
  const activeScenarioId = streamData.emergency_scenario;

  return (
    <div className={styles.patientMonitorPage}>
      <Container fluid>
      <Row className="mb-4 mt-4">
        <Col lg={12} className="d-flex justify-content-between align-items-center mb-3">
          <Button variant="outline-secondary" onClick={() => navigate('/')}>← Dashboard</Button>
          <h2 className="mb-0">Monitoring: {patient?.name}</h2>
        </Col>

        <Col lg={12}>
          <MedicalStatusBanner 
            prediction={streamData.prediction} 
            alertStage={streamData.alertStage} 
            key={`${streamData.alertStage}-${streamData.prediction?.label}`} 
          />
        </Col>
      </Row>

      <Row>
        <Col lg={9}>
          <div className="d-flex flex-column gap-3">
            {Object.entries(CLUSTER_CONFIG).map(([name, config]) => (
              <SystemCluster 
                key={name} 
                name={name} 
                config={config} 
                streamData={streamData} 
                emphasized={emphasizedSystem === name}
                isDimmed={emphasizedSystem && emphasizedSystem !== name}
              />
            ))}
          </div>
        </Col>
        <Col lg={3}>
          <Card className="shadow-sm border-0 mb-4">
           <Card.Header className="bg-primary text-white font-weight-bold d-flex justify-content-between align-items-center">
             Simulation Controls
             {isSimActive && <Badge bg="danger">ACTIVE</Badge>}
           </Card.Header>
           <Card.Body>
             {isSimActive ? (
               <div className="mb-4">
                 <Alert variant="info" className="py-2 px-3 mb-2 small fw-bold border-2">
                   <i className="bi bi-play-circle-fill me-2"></i>
                   {CLUSTER_CONFIG_EMERGENCY_MAP[activeScenarioId]} in progress
                 </Alert>
                 <Button variant="danger" className="w-100 fw-bold mb-3 shadow-sm" onClick={onAbortEmergency}>
                   <i className="bi bi-stop-circle me-2"></i>STOP SIMULATION
                 </Button>
                 <hr/>
               </div>
             ) : (
               <p className="small text-muted mb-3">Trigger an emergency to observe the cross-system physiological response.</p>
             )}
             
             <div className="d-grid gap-2">
                 {[
                   { id: 2, label: 'Heart Attack' },
                   { id: 3, label: 'Arrhythmia' },
                   { id: 4, label: 'Heart Failure' },
                   { id: 5, label: 'Hypoglycemia' },
                   { id: 6, label: 'Hyperglycemia' },
                   { id: 7, label: 'Resp. Distress' },
                   { id: 8, label: 'Sepsis' },
                   { id: 9, label: 'Stroke' },
                   { id: 10, label: 'Shock' },
                   { id: 11, label: 'Hypertensive Crisis' },
                   { id: 12, label: 'Fall Unconscious' }
                 ].map(e => {
                   const isActive = activeScenarioId === e.id;
                   return (
                     <Button
                       key={e.id}
                       variant={isActive ? "danger" : "outline-danger"}
                       size="sm"
                       disabled={isSimActive && !isActive}
                       className={isActive ? "fw-bold shadow border-2" : ""}
                       onClick={() => onTriggerEmergency(e.id)}
                     >
                       {isActive && <Spinner as="span" animation="grow" size="sm" role="status" aria-hidden="true" className="me-2"/>}
                       Simulate {e.label}
                     </Button>
                   );
                 })}
             </div>
           </Card.Body>
         </Card>

          <Card className="shadow-sm border-0">
            <Card.Header className="bg-dark text-white font-weight-bold">Emergency Contacts</Card.Header>
            <Card.Body className="p-0">
              <ul className="list-group list-group-flush">
                {patient?.contacts?.map((c, i) => (
                  <li key={i} className="list-group-item d-flex justify-content-between align-items-center">
                    <div>
                      <div className="font-weight-bold">{c.name}</div>
                      <small className="text-muted">{c.phone}</small>
                    </div>
                    {c.verified ? <Badge bg="success">Verified</Badge> : <Badge bg="secondary">Unverified</Badge>}
                  </li>
                ))}
              </ul>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  </div>
  );
};

export default PatientMonitorPage;
