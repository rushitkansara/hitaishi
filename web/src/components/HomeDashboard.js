import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { mockPatients } from '../data/mockData';
import { simulationData } from '../data/simulationData';
import PatientCard from './PatientCard';

const HomeDashboard = () => {
  const navigate = useNavigate();
  const [simulationResults, setSimulationResults] = useState([]);
  const [isSimulating, setIsSimulating] = useState(false);
  const [recipientNumber, setRecipientNumber] = useState('');

  const handlePatientClick = (patientId) => {
    navigate(`/patient/${patientId}`);
  };

  const handleSaveRecipientNumber = async () => {
    try {
      await fetch(`${process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000'}/set_recipient`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ number: recipientNumber }),
      });
      alert('Recipient number saved successfully!');
    } catch (error) {
      console.error("Error saving recipient number:", error);
      alert('Failed to save recipient number.');
    }
  };

  const runSimulation = async () => {
    setIsSimulating(true);
    setSimulationResults([]);

    for (const data of simulationData) {
      try {
        const response = await fetch(`${process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000'}/predict`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(data),
        });
        const result = await response.json();
        setSimulationResults(prevResults => [...prevResults, { ...result, patient_name: data.patient_name }]);
        await new Promise(resolve => setTimeout(resolve, 2000)); // 2-second delay
      } catch (error) {
        console.error("Error during simulation:", error);
      }
    }
    setIsSimulating(false);
  };

  return (
    <div className="container-fluid">
      {/* Header */}
      <div className="dashboard-header">
        <div className="container">
          <div className="text-center">
            <h1 className="dashboard-title">MediSense</h1>
            <p className="dashboard-subtitle">AI-Powered Home Isolation Assistant</p>
          </div>
        </div>
      </div>

      {/* Simulation Control */}
      <div className="container text-center my-4">
        <button className="btn btn-primary btn-lg" onClick={runSimulation} disabled={isSimulating}>
          {isSimulating ? 'Running Simulation...' : 'Run Emergency Simulation'}
        </button>
      </div>

      {/* Recipient Number Control */}
      <div className="container text-center my-4">
        <div className="row justify-content-center">
          <div className="col-md-6">
            <div className="input-group">
              <input
                type="text"
                className="form-control"
                placeholder="Enter recipient phone number"
                value={recipientNumber}
                onChange={(e) => setRecipientNumber(e.target.value)}
              />
              <button className="btn btn-outline-secondary" type="button" onClick={handleSaveRecipientNumber}>
                Save
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Simulation Results */}
      {simulationResults.length > 0 && (
        <div className="container">
          <div className="row">
            <div className="col-12">
              <div className="dashboard-section-header">
                <h2 className="section-title">
                  <span className="section-icon">🔬</span>
                  Simulation Results
                </h2>
              </div>
              <div className="list-group">
                {simulationResults.map((result, index) => (
                  <div key={index} className="list-group-item list-group-item-action flex-column align-items-start">
                    <div className="d-flex w-100 justify-content-between">
                      <h5 className="mb-1">{result.patient_name}</h5>
                      <small>Prediction {index + 1}</small>
                    </div>
                    <p className="mb-1">Predicted Condition: <strong>{result.risk_name}</strong></p>
                    <small>Confidence: {result.confidence.toFixed(2)}</small>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Patient Cards Grid */}
      <div className="container mt-5">
        <div className="row">
          <div className="col-12">
            <div className="dashboard-section-header">
              <h2 className="section-title">
                <span className="section-icon">👥</span>
                Patient Overview
              </h2>
              <p className="section-subtitle">Monitor all patients in real-time</p>
            </div>
          </div>
        </div>
        
        <div className="row g-4">
          {mockPatients.map((patient) => (
            <div key={patient.id} className="col-lg-4 col-md-6 col-sm-12">
              <div 
                className="patient-card"
                onClick={() => handlePatientClick(patient.id)}
              >
                <PatientCard patient={patient} />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default HomeDashboard;
