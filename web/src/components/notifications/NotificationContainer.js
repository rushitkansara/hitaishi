import React from 'react';
import { Toast, ToastContainer } from 'react-bootstrap';
import { useNotifications } from '../../contexts/NotificationContext';

const NotificationContainer = () => {
  const { notifications, removeNotification } = useNotifications();

  const getIcon = (type) => {
    switch (type) {
      case 'success': return '✅';
      case 'warning': return '⚠️';
      case 'info': return 'ℹ️';
      default: return '';
    }
  };

  const getBg = (type) => {
    switch (type) {
      case 'success': return 'success';
      case 'warning': return 'warning';
      case 'info': return 'info';
      default: return 'light';
    }
  };

  return (
    <ToastContainer position="top-end" className="p-3" style={{ zIndex: 9999, position: 'fixed' }}>
      {notifications.map((n) => (
        <Toast 
          key={n.id} 
          onClose={() => removeNotification(n.id)} 
          bg={getBg(n.type)}
          className={n.type === 'warning' ? '' : 'text-white'}
        >
          <Toast.Header closeButton>
            <strong className="me-auto">{getIcon(n.type)} Notification</strong>
          </Toast.Header>
          <Toast.Body style={{ fontSize: '0.95rem', fontWeight: 500 }}>
            {n.message}
          </Toast.Body>
        </Toast>
      ))}
    </ToastContainer>
  );
};

export default NotificationContainer;
