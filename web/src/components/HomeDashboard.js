import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Button, Form, Collapse, Badge } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import { initSimulation, getPatients, deletePatient } from '../services/api';
import ContactVerificationModal from './patient/ContactVerificationModal';
import { useNotifications } from '../contexts/NotificationContext';

const CONDITIONS = [
  'Healthy', 'Hypertension', 'Type 2 Diabetes', 'Heart Failure', 
  'COPD', 'Obesity', 'Chronic Kidney Disease', 'Atrial Fibrillation'
];

const ACTIVITY_LEVELS = ['active', 'moderate', 'sedentary'];

const PatientCard = ({ patient, onDelete }) => {
  const navigate = useNavigate();
  return (
    <Card className="mb-3 patient-card shadow-sm border-0">
      <Card.Body>
        <div className="d-flex justify-content-between align-items-center mb-2">
          <h5 className="mb-0 text-primary">{patient.name}</h5>
          <div className="d-flex gap-2 align-items-center">
            <Badge bg="success">Stable</Badge>
            <Button 
              variant="link" 
              className="text-danger p-0 line-height-1" 
              onClick={(e) => {
                e.stopPropagation();
                onDelete(patient.id);
              }}
              style={{ fontSize: '1.5rem', textDecoration: 'none' }}
              title="Delete Patient"
            >
              &times;
            </Button>
          </div>
        </div>
        <div className="small text-muted mb-3">
            Age: {patient.age} | {patient.gender} | {patient.primary_condition}
        </div>
        <Button variant="outline-primary" size="sm" onClick={() => navigate(`/monitor/${patient.id}`)}>
          Open Monitor
        </Button>
      </Card.Body>
    </Card>
  );
};

const HomeDashboard = () => {
  const { showInfo, showSuccess, showWarning, showError } = useNotifications();
  const [patients, setPatients] = useState([]);
  const [open, setOpen] = useState(false);
  const [newPatient, setNewPatient] = useState({ 
    name: '', age: '', gender: 'male', primary_condition: 'Healthy', activity_level: 'moderate', contacts: [{ name: '', phone: '', verified: 0 }] 
  });
  const [verifyingContact, setVerifyingContact] = useState(null);

  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const data = await getPatients();
        setPatients(data);
      } catch (error) {
        console.error("Failed to fetch patients:", error);
        showError("Failed to load patient data. Please try refreshing the page.");
      }
    };
    fetchPatients();
  }, [showError]);

  useEffect(() => {
    if (!sessionStorage.getItem('hasSeenHomeGuide')) {
      showInfo("Click Add Patient to add patient info.");
      sessionStorage.setItem('hasSeenHomeGuide', 'true');
    }
  }, [showInfo]);

  const handleContactChange = (index, field, value) => {
    const updatedContacts = [...newPatient.contacts];
    updatedContacts[index][field] = value;
    setNewPatient({ ...newPatient, contacts: updatedContacts });
  };

  const handleAddContact = () => {
    setNewPatient({
      ...newPatient,
      contacts: [...newPatient.contacts, { name: '', phone: '', verified: false }]
    });
  };

  const handleVerifySuccess = (index) => {
    const updatedContacts = [...newPatient.contacts];
    if (updatedContacts[index]) {
      updatedContacts[index].verified = true;
      setNewPatient({ ...newPatient, contacts: updatedContacts });
    }
  };

  const handleDelete = async (id) => {
    if (window.confirm("Are you sure you want to delete this patient? This will stop their simulation and remove all records.")) {
      try {
        await deletePatient(id);
        setPatients(prev => prev.filter(p => p.id !== id));
        showSuccess("Patient deleted successfully.");
      } catch (error) {
        showError("Failed to delete patient: " + error.message);
      }
    }
  };

  const finalizeRegistration = () => {
    showSuccess("Contact verified!");
  };

  const handleAddPatient = async () => {
    try {
      if (!newPatient.name || !newPatient.age) {
        showWarning("Please fill in patient name and age.");
        return;
      }
      
      const result = await initSimulation(newPatient);
      if (result.status === 'success' || result.status === 'exists') {
        if (result.status === 'success') {
          showSuccess(result.message);
          const addedPatient = { ...newPatient, id: result.patient_id };
          setPatients(prev => [...prev, addedPatient]);
          
          // If there are contacts, notify that alerts are disabled for now
          if (newPatient.contacts.length > 0 && newPatient.contacts[0].name) {
             showWarning("SMS alerts are currently disabled and will be turned on later.");
          }
        } else {
          showWarning(result.message);
        }
        setOpen(false);
        setNewPatient({ 
          name: '', age: '', gender: 'male', primary_condition: 'Healthy', activity_level: 'moderate', contacts: [{ name: '', phone: '', verified: 0 }] 
        });
      }
    } catch (error) {
      showError("Failed to register patient: " + error.message);
    }
  };

  return (
    <Container className="py-4">
      <div className="d-flex justify-content-between align-items-center mb-4">
        <h1>Patient Fleet</h1>
        <Button variant="primary" onClick={() => setOpen(!open)}>
          {open ? 'Hide Form' : 'Add Patient'}
        </Button>
      </div>

      <Collapse in={open}>
        <div id="add-patient-form" className="mb-4 p-3 border rounded shadow-sm">
          <Form>
            <Row>
              <Col md={12}><Form.Control placeholder="Patient Name" value={newPatient.name} onChange={e => setNewPatient({...newPatient, name: e.target.value})} className="mb-2" /></Col>
              <Col md={6}><Form.Control type="number" placeholder="Age" value={newPatient.age} onChange={e => setNewPatient({...newPatient, age: e.target.value})} className="mb-2" /></Col>
              <Col md={6}>
                  <Form.Select value={newPatient.gender} onChange={e => setNewPatient({...newPatient, gender: e.target.value})} className="mb-2">
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                  </Form.Select>
              </Col>
              <Col md={6}>
                  <Form.Select value={newPatient.activity_level} onChange={e => setNewPatient({...newPatient, activity_level: e.target.value})} className="mb-2">
                    {ACTIVITY_LEVELS.map(a => <option key={a} value={a}>{a.charAt(0).toUpperCase() + a.slice(1)}</option>)}
                  </Form.Select>
              </Col>
              <Col md={6}>
                  <Form.Select value={newPatient.primary_condition} onChange={e => setNewPatient({...newPatient, primary_condition: e.target.value})} className="mb-2">
                    {CONDITIONS.map(c => <option key={c} value={c}>{c}</option>)}
                  </Form.Select>
              </Col>
            </Row>
            
            <h6>Emergency Contacts</h6>
            {newPatient.contacts.map((contact, index) => (
                <Row key={index} className="mb-2 align-items-center">
                    <Col><Form.Control placeholder="Name" value={contact.name} onChange={e => handleContactChange(index, 'name', e.target.value)} /></Col>
                    <Col><Form.Control placeholder="Phone" value={contact.phone} onChange={e => handleContactChange(index, 'phone', e.target.value)} /></Col>
                </Row>
            ))}
            {newPatient.contacts.length < 5 && <Button variant="link" onClick={handleAddContact}>+ Add Contact</Button>}
            
            <br />
            <Button variant="success" className="mt-2 px-4" onClick={handleAddPatient}>Register Patient</Button>
          </Form>
        </div>
      </Collapse>

      {verifyingContact && (
        <ContactVerificationModal 
          contact={verifyingContact} 
          onClose={() => setVerifyingContact(null)}
          onVerified={(index) => {
            handleVerifySuccess(index);
            finalizeRegistration();
            setVerifyingContact(null);
          }}
        />
      )}

      <Row>
        {patients.map(p => (
          <Col key={p.id} md={4}>
            <PatientCard patient={p} onDelete={handleDelete} />
          </Col>
        ))}
      </Row>
    </Container>
  );
};

export default HomeDashboard;
