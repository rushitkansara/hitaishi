# Hitaishi: AI-Powered Health Monitoring System

## Project Overview

Hitaishi is an innovative project demonstrating an end-to-end machine learning pipeline for **real-time health risk prediction** in home-isolated patients. It leverages a lightweight LSTM model to detect 13 critical medical conditions from continuous 60-second vital sign sequences. The system is designed for resource-constrained environments, providing accessible, AI-powered diagnostic support through a web interface and real-time SMS alerts.

## Features

*   **Synthetic Data Generation:** Realistic time-series vital sign data for 13 medical conditions.
*   **Lightweight LSTM Model:** Optimized for low complexity and efficient CPU inference.
*   **High Accuracy:** Achieves ~99% overall accuracy in predicting medical conditions.
*   **Real-time Alerts:** Integrated with Twilio for immediate SMS notifications of emergency conditions.
*   **Web-based Dashboard:** Intuitive React frontend for patient monitoring and simulation.
*   **Resource-Efficient:** Designed for deployment on minimal hardware, including single-board devices.
*   **Scalable:** Financially viable and easily replicable for nationwide implementation.

## Demo



## Project Structure

```
hitaishi/
├── backend/                  # FastAPI server for ML inference and Twilio alerts
├── data-gen/                 # Scripts for synthetic data generation, validation, and visualization
├── models/                   # Stores trained ML models, evaluation reports, and metadata
├── logs/                     # TensorBoard logs for model training
├── src/                      # Source code for ML pipeline (training, evaluation, inference)
│   └── ml/                   # Machine Learning modules
├── web/                      # React frontend for the dashboard
└── venv/                     # Python virtual environment
```

## Setup and Installation

### Prerequisites

*   Python 3.8+
*   Node.js and npm
*   Git
*   A Twilio account with Account SID, Auth Token, and a purchased Twilio Phone Number.

### 1. Clone the Repository

```bash
git clone <repository_url>
cd hitaishi
```

### 2. Set up Python Environment

Create and activate a new Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the core Python dependencies:

```bash
pip install -r requirements.txt
```

Install the backend-specific Python dependencies:

```bash
pip install -r backend/requirements.txt
```

### 3. Set up Frontend Dependencies

Navigate to the `web` directory and install Node.js dependencies:

```bash
cd web
npm install
cd ..
```

### 4. Generate Simulation Data

Run the script to create mock data for the frontend simulation:

```bash
source venv/bin/activate && python3 create_simulation_data.py
```

## Configuration (Twilio)

To enable SMS alerts, you need to configure your Twilio credentials in `backend/twilio_service.py`.

Open `backend/twilio_service.py` and replace the placeholder values with your actual Twilio credentials:

```python
ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "YOUR_TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "YOUR_TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "YOUR_TWILIO_PHONE_NUMBER") # Must be a Twilio number you own
RECIPIENT_PHONE_NUMBER = os.environ.get("RECIPIENT_PHONE_NUMBER", "RECIPIENT_PHONE_NUMBER") # Number to receive alerts
```

**Note:** For production environments, it is highly recommended to use environment variables (`TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, etc.) instead of hardcoding credentials directly in the file.

## Running the Application

### 1. Start the Backend Server

In your main `hitaishi` directory, with your virtual environment activated, run:

```bash
source venv/bin/activate && uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 2. Start the Frontend Application

Open a **new terminal window**, navigate to the `hitaishi/web` directory, and run:

```bash
npm start
```

This will launch the web application in your browser (usually at `http://localhost:3000`).

### 3. Run the Emergency Simulation

Once the frontend loads, click the **"Run Emergency Simulation"** button on the dashboard. The application will send simulated vital sign sequences to the backend, display real-time predictions, and trigger SMS alerts for detected emergencies.

## ML Model Details

The project utilizes a custom LSTM model for multi-class classification of 13 medical conditions. The model achieves an overall accuracy of **~99%** and a critical emergency detection accuracy of **~98.75%**. While full integer quantization proved challenging in this environment, the unquantized Keras model is used for backend inference, providing robust performance.

## License

This project is licensed under the MIT License - see the LICENSE file for details. (Placeholder - create LICENSE file if needed)

## Contributing

