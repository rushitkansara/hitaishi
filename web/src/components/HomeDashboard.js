import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import PatientCard from './PatientCard';

// Define PROFILE_FACTORS for frontend dropdowns
const PROFILE_FACTORS = {
  genders: ['male', 'female'],
  age_bins: ['18-30', '31-45', '46-60', '61-75', '76-90'],
  activity_levels: ['sedentary', 'moderate', 'active'],
  primary_conditions: [
      'Healthy',
      'Hypertension',
      'Type 2 Diabetes',
      'Heart Failure',
      'COPD',
      'Obesity',
      'Chronic Kidney Disease',
      'Atrial Fibrillation'
  ]
};

const HomeDashboard = () => {
  const navigate = useNavigate();
  const [patients, setPatients] = useState([]);
  const [newPatient, setNewPatient] = useState({
    name: '',
    age: '',
    gender: PROFILE_FACTORS.genders[0],
    activity_level: PROFILE_FACTORS.activity_levels[0],
    primary_condition: PROFILE_FACTORS.primary_conditions[0],
    emergency_contact: ''
  });

  useEffect(() => {
    // Load patients from localStorage on component mount
    const storedPatients = JSON.parse(localStorage.getItem('patients')) || [];
    setPatients(storedPatients);
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setNewPatient({ ...newPatient, [name]: value });
  };

  const handleAddPatient = async (e) => {
    e.preventDefault();
    if (!newPatient.name || !newPatient.age || !newPatient.emergency_contact) {
      alert('Please fill in all required patient fields.');
      return;
    }

    const patientId = Date.now(); // Simple unique ID generation

    // Fetch initial baseline vitals from backend
    try {
      const response = await fetch(`${process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000'}/generate_stable_data`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newPatient.name,
          age: parseInt(newPatient.age),
          gender: newPatient.gender,
          activity_level: newPatient.activity_level,
          primary_condition: newPatient.primary_condition
        }),
      });
      const data = await response.json();
      const initialBaselines = data.sequence[0]; // Take the first frame as static baselines

      // Assign static total steps based on activity level
      let total_steps;
      switch (newPatient.activity_level) {
        case 'sedentary':
          total_steps = Math.floor(Math.random() * 2001);
          break;
        case 'moderate':
          total_steps = 2001 + Math.floor(Math.random() * 5999);
          break;
        case 'active':
          total_steps = 8001 + Math.floor(Math.random() * 7000);
          break;
        default:
          total_steps = 5000; // Default value
      }

      const patientToAdd = { 
        ...newPatient, 
        id: patientId, 
        age: parseInt(newPatient.age), 
        baselines: initialBaselines, 
        total_steps: total_steps
      };
      const updatedPatients = [...patients, patientToAdd];
      setPatients(updatedPatients);
      localStorage.setItem('patients', JSON.stringify(updatedPatients));

      // Clear form
      setNewPatient({
        name: '',
        age: '',
        gender: PROFILE_FACTORS.genders[0],
        activity_level: PROFILE_FACTORS.activity_levels[0],
        primary_condition: PROFILE_FACTORS.primary_conditions[0],
        emergency_contact: ''
      });
    } catch (error) {
      console.error("Error adding patient or fetching baselines:", error);
      alert('Failed to add patient or fetch baselines.');
    }
  };

  const handlePatientClick = (patientId) => {
    navigate(`/patient/${patientId}`);
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

      {/* Add Patient Form */}
      <div className="container my-4">
        <div className="card">
          <div className="card-header">Add New Patient</div>
          <div className="card-body">
            <form onSubmit={handleAddPatient}>
              <div className="row">
                <div className="col-md-6 mb-3">
                  <label htmlFor="name" className="form-label">Patient Name</label>
                  <input type="text" className="form-control" id="name" name="name" value={newPatient.name} onChange={handleInputChange} required />
                </div>
                <div className="col-md-6 mb-3">
                  <label htmlFor="age" className="form-label">Age</label>
                  <input type="number" className="form-control" id="age" name="age" value={newPatient.age} onChange={handleInputChange} required />
                </div>
                <div className="col-md-6 mb-3">
                  <label htmlFor="gender" className="form-label">Gender</label>
                  <select className="form-select" id="gender" name="gender" value={newPatient.gender} onChange={handleInputChange}>
                    {PROFILE_FACTORS.genders.map(g => <option key={g} value={g}>{g}</option>)}
                  </select>
                </div>
                <div className="col-md-6 mb-3">
                  <label htmlFor="activity_level" className="form-label">Activity Level</label>
                  <select className="form-select" id="activity_level" name="activity_level" value={newPatient.activity_level} onChange={handleInputChange}>
                    {PROFILE_FACTORS.activity_levels.map(al => <option key={al} value={al}>{al}</option>)}
                  </select>
                </div>
                <div className="col-md-6 mb-3">
                  <label htmlFor="primary_condition" className="form-label">Primary Condition</label>
                  <select className="form-select" id="primary_condition" name="primary_condition" value={newPatient.primary_condition} onChange={handleInputChange}>
                    {PROFILE_FACTORS.primary_conditions.map(pc => <option key={pc} value={pc}>{pc}</option>)}
                  </select>
                </div>
                <div className="col-md-6 mb-3">
                  <label htmlFor="emergency_contact" className="form-label">Emergency Contact</label>
                  <input type="text" className="form-control" id="emergency_contact" name="emergency_contact" value={newPatient.emergency_contact} onChange={handleInputChange} required />
                </div>
              </div>
              <button type="submit" className="btn btn-primary">Add Patient</button>
            </form>
          </div>
        </div>
      </div>

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
          {patients.length > 0 ? (
            patients.map((patient) => (
              <div key={patient.id} className="col-lg-4 col-md-6 col-sm-12">
                <div 
                  className="patient-card"
                  onClick={() => handlePatientClick(patient.id)}
                >
                  <PatientCard patient={patient} />
                </div>
              </div>
            ))
          ) : (
            <p className="text-center text-muted">No patients added yet. Use the form above to add a new patient.</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default HomeDashboard;
