import React from 'react';
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

const PatientCard = ({ patient }) => {
  // Prepare data for mini chart (heart rate trend)
  const chartData = {
    labels: ['6h ago', '5h ago', '4h ago', '3h ago', '2h ago', '1h ago', 'Now'],
    datasets: [
      {
        label: 'Heart Rate',
        data: patient.trends.heartRate,
        borderColor: '#1976d2',
        backgroundColor: 'rgba(25, 118, 210, 0.1)',
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 5,
        tension: 0.4,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        enabled: false,
      },
    },
    scales: {
      x: {
        display: false,
      },
      y: {
        display: false,
      },
    },
    elements: {
      point: {
        radius: 0,
      },
    },
  };

  const getStatusClass = (status) => {
    switch (status) {
      case 'stable':
        return 'status-stable';
      case 'warning':
        return 'status-warning';
      case 'critical':
        return 'status-critical';
      default:
        return 'status-stable';
    }
  };

  return (
    <>
      {/* Card Header with Photo */}
      <div className="patient-card-header">
        <div className="d-flex align-items-start">
          <div className="patient-photo-container">
            <img 
              src={patient.photo} 
              alt={patient.name}
              className="patient-photo"
              onError={(e) => {
                e.target.style.display = 'none';
                e.target.nextSibling.style.display = 'flex';
              }}
            />
            <div className="patient-photo-placeholder">
              <span>{patient.name.split(' ').map(n => n[0]).join('')}</span>
            </div>
          </div>
          <div className="patient-info-section">
            <div className="d-flex justify-content-between align-items-start">
              <div>
                <h5 className="patient-name">{patient.name}</h5>
                <p className="patient-condition">{patient.condition}</p>
                <p className="patient-age">Age: {patient.age}</p>
              </div>
              <span className={`status-indicator ${getStatusClass(patient.status)}`}>
                {getStatusText(patient.status)}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Vitals */}
      <div className="patient-vitals">
        <div className="vital-item">
          <span className="vital-label">Heart Rate</span>
          <span className="vital-value">{patient.vitals.heartRate} bpm</span>
        </div>
        <div className="vital-item">
          <span className="vital-label">Oxygen Level</span>
          <span className="vital-value">{patient.vitals.oxygenLevel}%</span>
        </div>
        <div className="vital-item">
          <span className="vital-label">Temperature</span>
          <span className="vital-value">{patient.vitals.temperature}°F</span>
        </div>
        <div className="vital-item">
          <span className="vital-label">Blood Pressure</span>
          <span className="vital-value">{patient.vitals.bloodPressure}</span>
        </div>
      </div>

      {/* Mini Chart */}
      <div className="mini-chart-container">
        <Line data={chartData} options={chartOptions} />
      </div>
    </>
  );
};

export default PatientCard;
