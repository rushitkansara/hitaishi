// Helper function to get status color
export const getStatusColor = (status) => {
  switch (status) {
    case 'stable':
      return '#28a745';
    case 'warning':
      return '#ffc107';
    case 'critical':
      return '#dc3545';
    default:
      return '#6c757d';
  }
};

// Helper function to get status text
export const getStatusText = (status) => {
  switch (status) {
    case 'stable':
      return 'Stable';
    case 'warning':
      return 'Warning';
    case 'critical':
      return 'Critical';
    default:
      return 'Unknown';
  }
};