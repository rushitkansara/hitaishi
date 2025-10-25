import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { mockPatients, getStatusText } from '../data/mockData';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const PatientDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const patient = mockPatients.find(p => p.id === parseInt(id));

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
          Back to Dashboard
        </button>
      </div>
    );
  }

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

  // Chart data for heart rate trend
  const heartRateData = {
    labels: ['6h ago', '5h ago', '4h ago', '3h ago', '2h ago', '1h ago', 'Now'],
    datasets: [
      {
        label: 'Heart Rate (bpm)',
        data: patient.trends.heartRate,
        borderColor: '#dc3545',
        backgroundColor: 'rgba(220, 53, 69, 0.1)',
        borderWidth: 3,
        pointRadius: 4,
        pointHoverRadius: 6,
        tension: 0.4,
      },
    ],
  };

  // Chart data for oxygen level trend
  const oxygenData = {
    labels: ['6h ago', '5h ago', '4h ago', '3h ago', '2h ago', '1h ago', 'Now'],
    datasets: [
      {
        label: 'Oxygen Level (%)',
        data: patient.trends.oxygenLevel,
        borderColor: '#28a745',
        backgroundColor: 'rgba(40, 167, 69, 0.1)',
        borderWidth: 3,
        pointRadius: 4,
        pointHoverRadius: 6,
        tension: 0.4,
      },
    ],
  };

  // Chart data for temperature trend
  const temperatureData = {
    labels: ['6h ago', '5h ago', '4h ago', '3h ago', '2h ago', '1h ago', 'Now'],
    datasets: [
      {
        label: 'Temperature (°F)',
        data: patient.trends.temperature,
        borderColor: '#ffc107',
        backgroundColor: 'rgba(255, 193, 7, 0.1)',
        borderWidth: 3,
        pointRadius: 4,
        pointHoverRadius: 6,
        tension: 0.4,
      },
    ],
  };

  // Chart data for current vitals comparison
  const vitalsComparisonData = {
    labels: ['Heart Rate', 'Oxygen Level', 'Temperature'],
    datasets: [
      {
        label: 'Current Values',
        data: [
          patient.vitals.heartRate,
          patient.vitals.oxygenLevel,
          patient.vitals.temperature
        ],
        backgroundColor: [
          'rgba(220, 53, 69, 0.8)',
          'rgba(40, 167, 69, 0.8)',
          'rgba(255, 193, 7, 0.8)'
        ],
        borderColor: [
          '#dc3545',
          '#28a745',
          '#ffc107'
        ],
        borderWidth: 2,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top',
      },
      tooltip: {
        mode: 'index',
        intersect: false,
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Time',
        },
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Value',
        },
      },
    },
  };

  const barChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Vital Signs',
        },
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Value',
        },
      },
    },
  };

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
              <p><strong>Condition:</strong> {patient.condition}</p>
              <p><strong>Patient ID:</strong> #{patient.id.toString().padStart(4, '0')}</p>
              <p><strong>Last Updated:</strong> {new Date(patient.lastUpdated).toLocaleString()}</p>
              <span className={`status-indicator ${getStatusClass(patient.status)}`}>
                {getStatusText(patient.status)}
              </span>
            </div>
          </div>
        </div>

        {/* Current Vitals */}
        <div className="current-vitals">
          <div className="vital-card">
            <div className="vital-card-value">{patient.vitals.heartRate}</div>
            <div className="vital-card-label">Heart Rate (bpm)</div>
          </div>
          <div className="vital-card">
            <div className="vital-card-value">{patient.vitals.oxygenLevel}%</div>
            <div className="vital-card-label">Oxygen Level</div>
          </div>
          <div className="vital-card">
            <div className="vital-card-value">{patient.vitals.temperature}°F</div>
            <div className="vital-card-label">Temperature</div>
          </div>
          <div className="vital-card">
            <div className="vital-card-value">{patient.vitals.bloodPressure}</div>
            <div className="vital-card-label">Blood Pressure</div>
          </div>
        </div>

        {/* Charts */}
        <div className="row">
          <div className="col-lg-6 col-md-12 mb-4">
            <div className="chart-container">
              <h3 className="chart-title">Heart Rate Trend</h3>
              <div className="chart-wrapper">
                <Line data={heartRateData} options={chartOptions} />
              </div>
            </div>
          </div>
          
          <div className="col-lg-6 col-md-12 mb-4">
            <div className="chart-container">
              <h3 className="chart-title">Oxygen Level Trend</h3>
              <div className="chart-wrapper">
                <Line data={oxygenData} options={chartOptions} />
              </div>
            </div>
          </div>
          
          <div className="col-lg-6 col-md-12 mb-4">
            <div className="chart-container">
              <h3 className="chart-title">Temperature Trend</h3>
              <div className="chart-wrapper">
                <Line data={temperatureData} options={chartOptions} />
              </div>
            </div>
          </div>
          
          <div className="col-lg-6 col-md-12 mb-4">
            <div className="chart-container">
              <h3 className="chart-title">Current Vitals Overview</h3>
              <div className="chart-wrapper">
                <Bar data={vitalsComparisonData} options={barChartOptions} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PatientDetail;
