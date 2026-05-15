import React, { useState, useEffect } from 'react';
import { Row, Col, Button, Form, Modal, Spinner } from 'react-bootstrap';
import { sendVerificationSms, verifyContactCode } from '../../services/api';

const ContactVerificationModal = ({ contact, onClose, onVerified }) => {
  const [stage, setStage] = useState('sending');
  const [error, setError] = useState('');
  const [secretWord, setSecretWord] = useState('');

  useEffect(() => {
    const autoSend = async () => {
      try {
        const res = await sendVerificationSms(contact.index, contact.name, contact.phone);
        // We expect the backend to return the code it sent so we can show it to the user
        setSecretWord(res.code || 'IRONMAN'); 
        setStage('awaiting');
      } catch (e) {
        setError('Failed to send SMS: ' + e.message);
        setStage('failed');
      }
    };
    autoSend();
  }, [contact]);

  const confirm = async (success) => {
    if (!success) {
      setError('Contact did not receive the message.');
      setStage('failed');
      return;
    }

    setStage('verifying');
    try {
      // Since the user confirmed they saw the word, we "force" verify on the backend
      const res = await verifyContactCode(contact.index, 'FORCE_VERIFY');
      if (res.valid) {
        setStage('success');
        setTimeout(() => onVerified(contact.index), 1000);
      } else {
        setError(res.reason || 'Verification failed');
        setStage('failed');
      }
    } catch (e) {
      setError('Error verifying: ' + e.message);
      setStage('failed');
    }
  };

  return (
    <Modal show onHide={onClose} backdrop="static" keyboard={false}>
      <Modal.Header closeButton>
        <Modal.Title>Verifying Contact</Modal.Title>
      </Modal.Header>
      <Modal.Body className="text-center p-4">
        {stage === 'sending' && (
          <>
            <Spinner animation="border" className="mb-3" />
            <p>Sending verification SMS to {contact.phone}...</p>
          </>
        )}

        {stage === 'awaiting' && (
            <>
                <h5 className="mb-3">SMS sent to your number.</h5>
                <p>Ask {contact.name}: "Did you receive a text with the word <strong>{secretWord.toUpperCase()}</strong>?"</p>
                <div className="d-flex justify-content-center gap-3 mb-2">
                  <Button variant="success" className="px-4" onClick={() => confirm(true)}>Yes, received</Button>
                  <Button variant="danger" className="px-4" onClick={() => confirm(false)}>No</Button>
                </div>
            </>
        )}

        {stage === 'verifying' && (
          <>
            <Spinner animation="border" className="mb-3" />
            <p>Finalizing...</p>
          </>
        )}

        {stage === 'success' && (
          <div className="text-success">
            <i className="bi bi-check-circle-fill" style={{ fontSize: '2rem' }}></i>
            <p className="mt-2">Verified! Completing registration...</p>
          </div>
        )}

        {stage === 'failed' && (
          <>
            <p className="text-danger">{error}</p>
            <Button variant="secondary" onClick={onClose}>Cancel</Button>
            <Button variant="primary" className="ms-2" onClick={() => setStage('sending')}>Retry</Button>
          </>
        )}
      </Modal.Body>
    </Modal>
  );
};

export default ContactVerificationModal;
