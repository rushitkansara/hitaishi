// hitaishi/web/src/services/api.js

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
const API_PREFIX = `${BACKEND_URL}/api/v1_2`;

export const getPatients = async () => {
  try {
    const response = await fetch(`${API_PREFIX}/patients`);
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to fetch patients');
    }
    return response.json();
  } catch (error) {
    console.error("Error fetching patients:", error);
    throw error;
  }
};

export const getPatient = async (patientId) => {
  try {
    const response = await fetch(`${API_PREFIX}/patients/${patientId}`);
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to fetch patient details');
    }
    return response.json();
  } catch (error) {
    console.error(`Error fetching patient ${patientId}:`, error);
    throw error;
  }
};

export const deletePatient = async (patientId) => {
  try {
    const response = await fetch(`${API_PREFIX}/patients/${patientId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to delete patient');
    }
    return response.json();
  } catch (error) {
    console.error(`Error deleting patient ${patientId}:`, error);
    throw error;
  }
};

export const initSimulation = async (patientProfile) => {
  try {
    const response = await fetch(`${API_PREFIX}/patients`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patientProfile),
    });
    if (!response.ok) {
      const errorData = await response.json();
      let errorMsg = 'Failed to initialize simulation';
      if (errorData.detail) {
        if (typeof errorData.detail === 'string') {
          errorMsg = errorData.detail;
        } else if (Array.isArray(errorData.detail)) {
          errorMsg = errorData.detail.map(err => `${err.loc.join('.')}: ${err.msg}`).join(', ');
        }
      }
      throw new Error(errorMsg);
    }
    return response.json();
  } catch (error) {
    console.error("Error initializing simulation:", error);
    throw error;
  }
};

export const triggerEmergency = async (patientId, stateCode) => {
  try {
    const response = await fetch(`${API_PREFIX}/sim/trigger/${stateCode}?patient_id=${patientId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to trigger emergency');
    }
    return response.json();
  } catch (error) {
    console.error(`Error triggering emergency ${stateCode}:`, error);
    throw error;
  }
};

export const getSimulationTick = async (patientId, requestHistory = false) => {
  try {
    const url = `${API_PREFIX}/sim/tick/${patientId}${requestHistory ? '?request_history=true' : ''}`;
    const response = await fetch(url);
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to fetch simulation tick');
    }
    return response.json();
  } catch (error) {
    console.error("Error fetching simulation tick:", error);
    throw error;
  }
};

export const sendVerificationSms = async (contactId, patientName, phone) => {
  try {
    const response = await fetch(`${API_PREFIX}/contacts/${contactId}/verify/send?patient_name=${encodeURIComponent(patientName)}&phone=${encodeURIComponent(phone)}`, { method: 'POST' });
    const data = await response.json();
    if (data.status === 'failed') {
      throw new Error(data.error || 'Failed to send SMS');
    }
    return data;
  } catch (error) {
    console.error("Error sending verification SMS:", error);
    throw error;
  }
};

export const verifyContactCode = async (contactId, code) => {
  try {
    const response = await fetch(`${API_PREFIX}/contacts/${contactId}/verify/confirm`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code })
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Verification request failed');
    }
    const data = await response.json();
    console.log("Verification response:", data);
    return data;
  } catch (error) {
    console.error("Error verifying code:", error);
    throw error;
  }
};
