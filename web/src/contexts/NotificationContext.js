import React, { createContext, useContext, useState, useCallback } from 'react';

const NotificationContext = createContext(null);

let notificationCounter = 0; // Simple counter for unique IDs

export const useNotifications = () => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotifications must be used within a NotificationProvider');
  }
  return context;
};

export const NotificationProvider = ({ children }) => {
  const [notifications, setNotifications] = useState([]);

  const addNotification = useCallback((message, type = 'info', duration = 5000) => {
    notificationCounter++; // Increment counter for a new unique ID
    const id = `notif-${notificationCounter}`;
    setNotifications((prev) => [...prev, { id, message, type, duration }]);
    
    if (duration !== Infinity) {
      setTimeout(() => {
        removeNotification(id);
      }, duration);
    }
  }, []);

  const removeNotification = useCallback((id) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  }, []);

  const showInfo = useCallback((message, duration = 7000) => addNotification(message, 'info', duration), [addNotification]);
  const showSuccess = useCallback((message, duration = 7000) => addNotification(message, 'success', duration), [addNotification]);
  const showWarning = useCallback((message, duration) => addNotification(message, 'warning', duration || 10000), [addNotification]);
  const showError = useCallback((message, duration) => addNotification(message, 'danger', duration || Infinity), [addNotification]);

  return (
    <NotificationContext.Provider value={{ notifications, showInfo, showSuccess, showWarning, showError, removeNotification }}>
      {children}
    </NotificationContext.Provider>
  );
};
