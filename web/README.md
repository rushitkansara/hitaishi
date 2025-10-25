# MediSense Dashboard

AI-Powered Home Isolation Assistant - A responsive React.js dashboard for visualizing patient health data in real-time.

## Features

- **Multi-Patient Overview**: Clean grid layout displaying 6 patient cards with key health metrics
- **Individual Patient Details**: Comprehensive charts and trends for each patient
- **Real-time Visualization**: Chart.js integration for health trend monitoring
- **Responsive Design**: Bootstrap 5 powered responsive layout
- **Medical Theme**: Professional medical color scheme with soft blues and whites

## Tech Stack

- React.js 18 (Functional Components + Hooks)
- React Router DOM for navigation
- Chart.js with react-chartjs-2 for data visualization
- Bootstrap 5 for responsive layout
- CSS3 for custom styling

## Installation

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm start
```

3. Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

## Project Structure

```
src/
├── components/
│   ├── HomeDashboard.js    # Main dashboard with patient cards
│   ├── PatientCard.js      # Individual patient card component
│   └── PatientDetail.js    # Detailed patient view with charts
├── data/
│   └── mockData.js         # Mock patient data and helper functions
├── App.js                  # Main app component with routing
├── App.css                 # Custom styles and medical theme
└── index.js                # React app entry point
```

## Features Overview

### Home Dashboard
- Displays 6 patient cards in a responsive grid
- Each card shows: name, heart rate, oxygen level, temperature, blood pressure
- Status indicators (Stable/Warning/Critical) with color coding
- Mini Chart.js graphs showing short-term health trends
- Click any card to navigate to detailed patient view

### Patient Detail Page
- Patient profile with avatar, condition, and last updated timestamp
- Current vital signs overview cards
- Multiple Chart.js visualizations:
  - Heart rate trend line chart
  - Oxygen level trend line chart
  - Temperature trend line chart
  - Current vitals comparison bar chart
- Back button to return to dashboard

## Mock Data

The application uses mock JSON data simulating real-time patient monitoring:
- 6 patients with different conditions and statuses
- Historical trend data for the last 6 hours
- Realistic vital sign ranges and variations

## Design Philosophy

- **Medical Professional**: Clean, clinical design suitable for healthcare professionals
- **Accessibility**: High contrast colors and clear typography
- **Responsive**: Works seamlessly on desktop, tablet, and mobile devices
- **Performance**: Optimized Chart.js configurations for smooth rendering

## Future Enhancements

This dashboard is designed to be easily connected to a backend:
- Replace mock data with API calls
- Add WebSocket integration for real-time updates
- Implement user authentication
- Add patient management features
- Include alert/notification system

## Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge

## License

This project is created for demonstration purposes.
