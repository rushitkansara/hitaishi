import React from 'react';
import { getStatusText } from '../data/mockData';

const VITAL_CONFIG = {
  0: { name: 'Heart Rate', unit: 'bpm', precision: 0 },
  1: { name: 'Systolic BP', unit: 'mmHg', precision: 0 },
  2: { name: 'Diastolic BP', unit: 'mmHg', precision: 0 },
  3: { name: 'SpO2', unit: '%', precision: 0 },
  4: { name: 'Temperature', unit: '°C', precision: 1 },
  5: { name: 'Respiratory Rate', unit: 'breaths/min', precision: 0 },
  6: { name: 'Blood Glucose', unit: 'mg/dL', precision: 0 },
};

const PatientCard = ({ patient }) => {
  const getStatusClass = (status) => {
    // For now, all dynamically added patients are 'stable' until a simulation changes their status
    return 'status-stable'; 
  };

  return (
    <>
      {/* Card Header with Photo */}
      <div className="patient-card-header">
        <div className="d-flex align-items-start">
          <div className="patient-photo-container">
            <div className="patient-photo-placeholder">
              <span>{patient.name.split(' ').map(n => n[0]).join('')}</span>
            </div>
          </div>
          <div className="patient-info-section">
            <div className="d-flex justify-content-between align-items-start">
              <div>
                <h5 className="patient-name">{patient.name}</h5>
                <p className="patient-condition">{patient.primary_condition}</p>
                <p className="patient-age">Age: {patient.age}</p>
              </div>
              <span className={`status-indicator ${getStatusClass('stable')}`}>
                {getStatusText('stable')}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Vitals */}
      <div className="patient-vitals">
        {patient.baselines && Object.entries(patient.baselines).map(([key, value], index) => {
          const config = VITAL_CONFIG[index];
          if (!config) return null;
          return (
            <div className="vital-item" key={key}>
              <span className="vital-label">{config.name}</span>
              <span className="vital-value">{Number(value).toFixed(config.precision)} {config.unit}</span>
            </div>
          )
        })}
        <div className="vital-item">
            <span className="vital-label">Activity (24h)</span>
            <span className="vital-value">{patient.total_steps} steps</span>
        </div>
      </div>
    </>
  );
};

export default PatientCard;
