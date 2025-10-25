// Mock data for patient health metrics
export const mockPatients = [
  {
    id: 1,
    name: "Sarah Johnson",
    age: 34,
    condition: "COVID-19 Recovery",
    photo: "https://images.unsplash.com/photo-1494790108755-2616b612b786?w=150&h=150&fit=crop&crop=face",
    lastUpdated: "2024-01-15T10:30:00Z",
    status: "stable",
    vitals: {
      heartRate: 72,
      oxygenLevel: 98,
      temperature: 98.6,
      bloodPressure: "120/80"
    },
    trends: {
      heartRate: [75, 73, 72, 71, 72, 70, 72],
      oxygenLevel: [97, 98, 98, 99, 98, 98, 98],
      temperature: [98.4, 98.6, 98.5, 98.7, 98.6, 98.5, 98.6]
    }
  },
  {
    id: 2,
    name: "Michael Chen",
    age: 28,
    condition: "Post-Surgery Monitoring",
    photo: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&h=150&fit=crop&crop=face",
    lastUpdated: "2024-01-15T10:25:00Z",
    status: "stable",
    vitals: {
      heartRate: 68,
      oxygenLevel: 99,
      temperature: 98.2,
      bloodPressure: "115/75"
    },
    trends: {
      heartRate: [70, 69, 68, 67, 68, 69, 68],
      oxygenLevel: [98, 99, 99, 100, 99, 99, 99],
      temperature: [98.1, 98.2, 98.0, 98.3, 98.2, 98.1, 98.2]
    }
  },
  {
    id: 3,
    name: "Emily Rodriguez",
    age: 45,
    condition: "Chronic Condition",
    photo: "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=150&h=150&fit=crop&crop=face",
    lastUpdated: "2024-01-15T10:20:00Z",
    status: "critical",
    vitals: {
      heartRate: 95,
      oxygenLevel: 92,
      temperature: 99.8,
      bloodPressure: "140/95"
    },
    trends: {
      heartRate: [90, 92, 94, 95, 96, 95, 95],
      oxygenLevel: [94, 93, 92, 91, 92, 92, 92],
      temperature: [99.2, 99.5, 99.7, 99.8, 99.9, 99.8, 99.8]
    }
  },
  {
    id: 4,
    name: "David Thompson",
    age: 52,
    condition: "Diabetes Management",
    photo: "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150&h=150&fit=crop&crop=face",
    lastUpdated: "2024-01-15T10:15:00Z",
    status: "stable",
    vitals: {
      heartRate: 78,
      oxygenLevel: 97,
      temperature: 98.4,
      bloodPressure: "130/85"
    },
    trends: {
      heartRate: [80, 79, 78, 77, 78, 79, 78],
      oxygenLevel: [96, 97, 97, 98, 97, 97, 97],
      temperature: [98.3, 98.4, 98.2, 98.5, 98.4, 98.3, 98.4]
    }
  },
  {
    id: 5,
    name: "Lisa Wang",
    age: 29,
    condition: "Respiratory Monitoring",
    photo: "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=150&h=150&fit=crop&crop=face",
    lastUpdated: "2024-01-15T10:10:00Z",
    status: "stable",
    vitals: {
      heartRate: 65,
      oxygenLevel: 96,
      temperature: 98.1,
      bloodPressure: "110/70"
    },
    trends: {
      heartRate: [67, 66, 65, 64, 65, 66, 65],
      oxygenLevel: [95, 96, 96, 97, 96, 96, 96],
      temperature: [98.0, 98.1, 97.9, 98.2, 98.1, 98.0, 98.1]
    }
  },
  {
    id: 6,
    name: "Robert Kim",
    age: 38,
    condition: "Cardiac Monitoring",
    photo: "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=150&h=150&fit=crop&crop=face",
    lastUpdated: "2024-01-15T10:05:00Z",
    status: "warning",
    vitals: {
      heartRate: 88,
      oxygenLevel: 95,
      temperature: 99.1,
      bloodPressure: "135/88"
    },
    trends: {
      heartRate: [85, 86, 87, 88, 89, 88, 88],
      oxygenLevel: [96, 95, 95, 94, 95, 95, 95],
      temperature: [98.8, 99.0, 99.1, 99.2, 99.1, 99.0, 99.1]
    }
  }
];

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
