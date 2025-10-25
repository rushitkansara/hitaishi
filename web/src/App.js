import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import HomeDashboard from './components/HomeDashboard';
import PatientDetail from './components/PatientDetail';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<HomeDashboard />} />
          <Route path="/patient/:id" element={<PatientDetail />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
