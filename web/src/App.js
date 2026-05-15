import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import HomeDashboard from './components/HomeDashboard';
import PatientDetail from './components/PatientDetail';
import PatientMonitorPage from './components/PatientMonitorPage';
import { NotificationProvider } from './contexts/NotificationContext';
import NotificationContainer from './components/notifications/NotificationContainer';

function App() {
  return (
    <NotificationProvider>
      <Router>
        <div className="App">
          <NotificationContainer />
          <Routes>
            <Route path="/" element={<HomeDashboard />} />
            <Route path="/patient/:id" element={<PatientDetail />} />
            <Route path="/monitor/:patientId" element={<PatientMonitorPage />} />
          </Routes>
        </div>
      </Router>
    </NotificationProvider>
  );
}

export default App;
