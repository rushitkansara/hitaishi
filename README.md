# Hitaishi: Intelligent Health Monitoring & Emergency Detection System

## Executive Overview
Hitaishi is a clinical-grade health monitoring platform engineered to perform real-time diagnostic analysis on home-isolated patients. The system utilizes a deep learning architecture to continuously monitor 7 vital signs, identifying 13 critical medical conditions with high recall. By integrating physiological data simulation, edge-optimized machine learning, and low-latency alerting, Hitaishi demonstrates a robust, full-stack approach to medical technology.

## Engineering Scope & Contributions
As the lead engineer, I designed and implemented the end-to-end architecture, encompassing:
*   **Data Science Pipeline:** Custom data generation engine to simulate multi-dimensional physiological trajectories.
*   **Deep Learning Architecture:** Optimization of a multi-layer LSTM model for sequence classification, achieving a footprint of ~35 KB via INT8 quantization for edge-readiness.
*   **Backend Engineering:** Scalable FastAPI-based inference server, ensuring high-concurrency stream processing for real-time risk stratification.
*   **Full-Stack Integration:** Development of a high-density clinical dashboard for real-time visualization and emergency response management.

## Technical Specifications
| Component | Technology Stack |
| :--- | :--- |
| **Backend** | Python, FastAPI, SQLAlchemy (PostgreSQL), NumPy, Pandas |
| **Frontend** | React (Hooks/Context), Recharts (for high-frequency telemetry) |
| **Intelligence** | TensorFlow/Keras (LSTM), Scikit-Learn |
| **Deployment** | Docker-ready architecture with production-level environment configuration |

## Core Engineering Achievements
*   **High-Fidelity Simulation Engine:** Developed an engine that generates synthetic cohorts with temporal physiological noise, ensuring the model remains robust against sensor data variance.
*   **Inference Optimization:** Successfully reduced model complexity for low-latency inference on standard CPUs, maintaining high diagnostic accuracy without the need for specialized hardware.
*   **Clinical-Grade UI/UX:** Implemented a 'Status-Reason-Action' diagnostic protocol, designed to translate raw model inference into actionable clinical decisions, minimizing time-to-intervention.
*   **System Reliability:** Engineered input-buffer management and state-synchronization handlers to ensure seamless streaming and zero-crash initialization, even during burst-load scenarios.

## Repository Overview
```
hitaishi/
├── backend/                  # API server, DB logic, and service integrations
├── data_gen/                 # Physiological data simulation and validation pipeline
├── src/ml/                   # ML core: Training, quantization, and inference engine
├── web/                      # React-based clinical dashboard
└── experiments/              # Training history, versioned models, and metadata
```

## Setup & Deployment Instructions
*Instructions provided for development and deployment environments.*

### 1. Environment Configuration
*   **Python:** Install dependencies via `requirements.txt` and `backend/requirements.txt`.
*   **Database:** Configured for PostgreSQL integration.
*   **Alerting:** Twilio integration for automated emergency SMS notification.

### 2. Development Execution
1.  **Backend:** Initialize the API via `uvicorn backend.main:app`.
2.  **Frontend:** Build and serve the dashboard using `npm start`.

---
*For a detailed examination of the underlying physiological modeling and data validation logic, please refer to [DATA_VALIDATION_REWRITTEN.md](./DATA_VALIDATION_REWRITTEN.md).*
