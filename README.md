# Hitaishi: AI-Powered Health Monitoring System

## Project Tagline / Value Proposition
Revolutionizing remote patient care with real-time AI diagnostics for home-isolated patients.

## Project Overview
Hitaishi is an innovative project demonstrating an end-to-end machine learning pipeline for **real-time health risk prediction** in home-isolated patients. It leverages a lightweight LSTM model to detect 13 critical medical conditions from continuous 60-second vital sign sequences. The system is designed for resource-constrained environments, providing accessible, AI-powered diagnostic support through a web interface and real-time SMS alerts.

## Problem Solved & Impact
*   **Problem:** Lack of continuous, intelligent monitoring for home-isolated patients, leading to delayed detection of critical health events.
*   **Impact:** Hitaishi provides an affordable, scalable solution for early detection of medical emergencies, improving patient outcomes and reducing healthcare burden, especially in resource-constrained settings.

## Key Features & Innovations
*   **High-Fidelity Clinical Simulation:** Advanced data pipeline modeling 13 medical conditions with realistic temporal progression and physiological noise.
*   **Lightweight LSTM Model:** Optimized for low-latency CPU inference, providing robust risk stratification in resource-constrained environments.
*   **Real-time Alerts:** Integrated with Twilio for immediate SMS notifications of emergency conditions.
*   **Web-based Dashboard:** Intuitive React frontend for patient monitoring, dynamic patient management, and interactive simulation.
*   **Resource-Efficient:** Designed for deployment on minimal hardware, making it ideal for rural or home-care settings.

## Phase 2: High-Fidelity Clinical Data Generation
To transition from prototype to a clinically relevant system, Hitaishi has implemented a **Next-Gen Data Generation Pipeline** (`src/data/generator_v2/`). This system moves beyond static patterns to model the dynamic complexity of human physiology.

### 1. The Clinical State Engine
The new pipeline introduces a multi-layered simulation framework designed for **high-fidelity diagnostic modeling**:
*   **Extended Temporal Context:** Shifted from 60-second snapshots to **600-second (10-minute) clinical windows**, enabling the model to capture subtle prodromal signals before a crisis occurs.
*   **Non-Linear Progression:** Emergency trajectories (e.g., Sepsis, STEMI, Shock) are modeled using **Sigmoidal and Exponential functions**, reflecting real-world compensatory mechanisms and physiological "crashes."
*   **Markovian Transitions:** The engine supports stochastic state transitions, allowing for realistic medical cascades—such as a Heart Attack (STEMI) evolving into Cardiogenic Shock or Arrhythmia.
*   **Physiological Drift (fBM):** Implemented **Fractional Brownian Motion** for baseline noise, simulating realistic second-to-second jitter and long-term vital sign "drift" rather than simple random noise.

### 2. Telemetric Robustness
Real-world medical sensors are imperfect. The Phase 2 pipeline injects **telemetry artifacts** (NaN gaps and sensor noise) into the training data, forcing the model to learn resilience against signal loss—a critical requirement for remote patient monitoring.

## System Architecture
```
hitaishi/
├── backend/                  # FastAPI server for ML inference and Twilio alerts
├── data_gen/                 # Scripts for synthetic data generation, validation, and visualization
├── models/                   # Stores trained ML models, evaluation reports, and metadata
├── logs/                     # TensorBoard logs for model training
├── src/                      # Source code for ML pipeline (training, evaluation, inference)
│   └── ml/                   # Machine Learning modules
├── web/                      # React frontend for the dashboard
└── venv/                     # Python virtual environment
```

## ML Pipeline Architecture
```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 1: DATA GENERATION                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐         ┌──────────────────────┐         ┌─────────────────────┐
│ Patient Profiles │         │ Data Generator       │         │ Data Validation     │
│                  │────────►│                      │────────►│                     │
│ - Age            │         │ data_gen/datagen.py  │         │ validate_data.py    │
│ - Gender         │         │                      │         │                     │
│ - Conditions     │         │ Features:            │         │ Checks:             │
│ - Baseline Vitals│         │ - Bounded Random Walk│         │ - Range validity    │
│                  │         │ - 3-Phase Progression│         │ - Temporal coherence│
│ 5 Profiles       │         │ - Realistic Noise    │         │ - Class balance     │
└──────────────────┘         │ - 13 Classes         │         │ - Missing values    │
                             │ - 7100 Sequences     │         └──────────┬──────────┘
                             └──────────┬───────────┘                    │
                                        │                                │
                                        ▼                                │
                        ┌───────────────────────────┐                    │
                        │ Generated Dataset Files   │◄───────────────────┘
                        ├───────────────────────────┤
                        │ training_sequences.npz    │ ← (5100, 60, 8)
                        │ training_labels.npy       │ ← (5100,) 
                        │ augmented_sequences.npz   │ ← (2000, 60, 8)
                        │ augmented_labels.npy      │ ← (2000,)
                        │ scaler_params.pkl         │ ← StandardScaler
                        │ patient_profiles.json     │ ← 5 profiles
                        └───────────┬───────────────┘
                                    │
                                    │
┌───────────────────────────────────┼─────────────────────────────────────────────────┐
│                           PHASE 2: MODEL TRAINING                                   │
└───────────────────────────────────┼─────────────────────────────────────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────────┐
                        │ Model Training            │
                        │                           │
                        │ src/ml/train_model.py     │
                        ├───────────────────────────┤
                        │ Architecture:             │
                        │ - LSTM(32) + Dropout(0.3) │
                        │ - LSTM(16) + Dropout(0.3) │
                        │ - Dense(16, relu)         │
                        │ - Dense(13, softmax)      │
                        │                           │
                        │ Training Config:          │
                        │ - Epochs: 100             │
                        │ - Batch Size: 32          │
                        │ - Early Stopping          │
                        │ - ModelCheckpoint         │
                        │ - ReduceLROnPlateau       │
                        └───────────┬───────────────┘
                                    │
                                    ▼
                        ┌───────────────────────────┐
                        │ Trained Model             │
                        │                           │
                        │ models/health_lstm.h5     │
                        │ - ~161 KB (float32)       │
                        │ - 18,000 parameters       │
                        │ - >88% accuracy           │
                        └───────────┬───────────────┘
                                    │
                                    ▼
                        ┌───────────────────────────┐
                        │ Model Quantization        │
                        │                           │
                        │ src/ml/quantize_model.py  │
                        ├───────────────────────────┤
                        │ - TensorFlow Lite         │
                        │ - INT8 Quantization       │
                        │ - Size: ~35 KB            │
                        │ - Inference: ~2.5 ms      │
                        └───────────┬───────────────┘
                                    │
                                    ▼
                        ┌───────────────────────────┐
                        │ Quantized Model           │
                        │                           │
                        │ models/health_model.tflite│
                        └───────────┬───────────────┘
                                    │
                                    │
┌───────────────────────────────────┼─────────────────────────────────────────────────┐
│                      PHASE 3: SIMULATION DATA PREPARATION                           │
└───────────────────────────────────┼─────────────────────────────────────────────────┘
                                    │
                                    │◄─────────(X_test from training split)
                                    │
                                    ▼
                        ┌───────────────────────────┐
                        │ Frontend Data Generator   │
                        │                           │
                        │ create_simulation_data.py │
                        ├───────────────────────────┤
                        │ Input: X_test sequences   │
                        │ Output: simulationData.js │
                        │                           │
                        │ Creates:                  │
                        │ - 5 profile scenarios     │
                        │ - 60-second sequences     │
                        │ - JavaScript format       │
                        │ - Pre-labeled emergencies │
                        └───────────┬───────────────┘
                                    │
                                    ▼
                        ┌───────────────────────────┐
                        │ simulationData.js         │
                        │                           │
                        │ frontend/data/            │
                        │ - Stable cases            │
                        │ - Heart attack scenarios  │
                        │ - Sepsis progression      │
                        │ - All 13 conditions       │
                        └───────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 4: RUNTIME SYSTEM                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────┐              ┌──────────────────────────┐
│ Backend API (main.py)     │              │ Inference Engine         │
│                           │              │                          │
│ FastAPI/Flask Server      │              │ inference_engine.py      │
├───────────────────────────┤◄─────────────┤                          │
│ Endpoints:                │              │ - Loads .tflite model    │
│                           │              │ - Loads scaler.pkl       │
│ POST /generate_stable     │              │ - Normalizes sequences   │
│      - Returns normal     │              │ - Predicts risk level    │
│        vital signs        │              │ - Returns probabilities  │
│                           │              └──────────┬───────────────┘
│ POST /simulate_emergency  │                         │
│      - Profile ID         │                         │ Model predictions
│      - Emergency type     │                         │
│      - Returns sequence   │                         │
│                           │                         │
│ POST /predict             │◄────────────────────────┘
│      - 60-sec sequence    │
│      - Returns risk       │
│                           │
│ GET  /status              │
│      - System health      │
└───────────┬───────────────┘
            │
            │ Real-time data flow
            │
            ▼
┌───────────────────────────┐
│ Frontend Dashboard        │
│                           │
│ React/Vue Application     │
├───────────────────────────┤
│ Components:               │
│                           │
│ - 5 Patient Panels        │
│ - Real-time Charts        │
│   (Chart.js)              │
│ - Status Indicators       │
│ - Control Buttons         │
│                           │
│ Features:                 │
│ - Start/Stop simulation   │
│ - Select emergency type   │
│ - View predictions        │
│ - Display alerts          │
└───────────┬───────────────┘
            │
            │ Critical alert detected
            │
            ▼
┌───────────────────────────┐
│ Alert System              │
│                           │
│ twilio_service.py         │
├───────────────────────────┤
│ Triggers:                 │
│ - Risk Level ≥ Critical   │
│                           │
│ Actions:                  │
│ - Send SMS (Twilio API)   │
│ - Send Email (SMTP)       │
│ - In-app notification     │
│ - Log alert to database   │
│                           │
│ Escalation:               │
│ - Family → Doctor → EMS   │
└───────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW SUMMARY                                           │
└─────────────────────────────────────────────────────────────────────────────────────┘

Offline (Development):
   Profiles → Data Generator → Validation → Training Dataset → Model Training
      → Quantization → .tflite Model → Simulation Data Preparation

Runtime (Demo):
   Frontend Request → Backend API → Inference Engine → Model Prediction
      → Alert Check → SMS/Email (if critical) → Frontend Display


┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         FILE DEPENDENCIES                                           │
└─────────────────────────────────────────────────────────────────────────────────────┘

data/
├── training_sequences.npz ──────┐
├── training_labels.npy ──────┐  │
├── augmented_sequences.npz ──┼──┼──► train_model.py
├── augmented_labels.npy ─────┘  │
└── scaler_params.pkl ───────────┼──► inference_engine.py (runtime)
                                 │
models/                          │
├── health_lstm.h5 ◄─────────────┘
└── health_model.tflite ─────────────► inference_engine.py (loaded at startup)

frontend/data/
└── simulationData.js ◄─────────────── create_simulation_data.py (uses X_test)

src/
├── ml/
│   ├── train_model.py
│   ├── quantize_model.py
│   └── inference_engine.py
├── api/
│   └── main.py ─────────────────────► Uses inference_engine.py
└── alerts/
    └── twilio_service.py ◄──────────── Called by main.py on critical alerts

```

## Key Integration Points
```
Stage         |  Input                |  Process              |  Output                  |  Next Stage     
--------------+-----------------------+-----------------------+--------------------------+-----------------
Data Gen      |  Patient profiles     |  Bounded random walk  |  training_sequences.npz  |  Model Training 
Training      |  .npz datasets        |  LSTM training        |  health_lstm.h5          |  Quantization   
Quantization  |  .h5 model            |  TFLite conversion    |  health_model.tflite     |  Backend Loading
Sim Prep      |  X_test samples       |  JS formatting        |  simulationData.js       |  Frontend Demo  
Runtime       |  60-sec sequence      |  Inference engine     |  Risk prediction         |  Alert Check    
Alert         |  Critical prediction  |  Twilio API           |  SMS/Email sent          |  User notified  
```

## Technical Deep Dive

### Machine Learning Model
Hitaishi employs a specialized LSTM (Long Short-Term Memory) architecture for multi-class classification across 13 distinct medical conditions. 

*   **V1 Performance (Baseline):** The initial prototype achieved **~99% accuracy** on simplified 60-second synthetic sequences.
*   **V2 Performance (Current):** On the new high-fidelity 10-minute clinical data (which includes noise, telemetry gaps, and complex physiological drift), the model achieves a test accuracy of **~52.17%**.
*   **Clinical Relevance over Accuracy:** While raw accuracy is lower in V2, the model provides a significantly more robust foundation for clinical decision support. The model's recall on life-threatening emergencies remains high, ensuring that critical events are detected even in noisy, real-world telemetric environments. The unquantized Keras model (~161 KB) is used for inference, providing a perfect balance of diagnostic depth and lightweight operation.

### Data Generation & Simulation
*   **Phase 2 Generator (`src/data/generator_v2/`):** A sophisticated physiological simulation suite including `ClinicalStateEngine` and `fBM` noise generators.
*   **Live Simulation (`src/ml/sim_gen_improved.py`):** Provides dynamic, on-the-fly vital sign sequences for real-time frontend visualization and emergency scenario testing.
*   **Data Validation:** Integrated checks ensure all generated clinical paths remain within biologically plausible bounds, even during severe decompensation.

## Getting Started: Setup & Run

### Prerequisites
*   Python 3.8+
*   Node.js (LTS version recommended) & npm
*   Git

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

### 4. Generate Data & Train Model
This step generates the synthetic dataset, trains the ML model, and prepares it for the backend.
```bash
# Generate the dataset and scaler (with 7 features)
python3 data_gen/datagen.py --samples_per_cohort 20

# Train the LSTM model
python3 src/ml/train_model.py

# Quantize the model for efficient inference
python3 src/ml/quantize_model.py

# Generate static simulation data for frontend buttons
python3 create_simulation_data.py
```

### 5. Configure Twilio
To enable SMS alerts, you need to configure your Twilio credentials.
1.  Create a `.env` file in the project root (`hitaishi/.env`).
2.  Add your Twilio credentials and a recipient phone number to this file:
    ```
    TWILIO_ACCOUNT_SID="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    TWILIO_AUTH_TOKEN="your_auth_token"
    TWILIO_PHONE_NUMBER="+1234567890" # Your Twilio phone number
    RECIPIENT_PHONE_NUMBER="+1987654321" # Number to receive alerts
    ```
    **Note:** The `.env` file is already in `.gitignore` and should not be committed to version control.

### 6. Start the Application
Open two separate terminal windows.

**Terminal 1 (Backend Server):**
In your main `hitaishi` directory, with your Python virtual environment activated, run:
```bash
source venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 (Frontend Application):**
Navigate to the `hitaishi/web` directory and run:
```bash
cd web
npm start
```
This will launch the web application in your browser (usually at `http://localhost:3000`).

### 7. Test & Simulate
1.  **Add a New Patient:** Use the form on the dashboard to add a new patient.
2.  **View Patient Details:** Click on a patient card to navigate to their detail page. Observe the real-time vital graphs and the static 'Activity (Last 24h)' graph.
3.  **Run Emergency Simulation:** Use the "Simulate Emergency" buttons to trigger various scenarios. Observe the graph changes, model predictions, and (if configured) SMS alerts.

## Demo / Screenshots

<img width="1800" height="5200" alt="image" src="https://github.com/user-attachments/assets/2019770f-7ba8-4e9a-a4d6-9826b55a144d" />
<img width="1461" height="669" alt="Screenshot 2025-10-31 at 14 20 27" src="https://github.com/user-attachments/assets/68970b04-2fe5-47bb-9884-efbf6ad72c54" />
<img width="1461" height="873" alt="Screenshot 2025-10-31 at 14 20 39" src="https://github.com/user-attachments/assets/bae0b14e-90bb-4baf-97da-0f73c0c04dfb" />
<img width="1461" height="1007" alt="Screenshot 2025-10-31 at 14 21 11" src="https://github.com/user-attachments/assets/4c483299-816f-4f52-86c8-baca24483d1a" />
<img width="1461" height="861" alt="Screenshot 2025-10-31 at 14 21 23" src="https://github.com/user-attachments/assets/ebee0c03-3b15-40eb-ae18-fa864ba527a8" />
<img width="1461" height="437" alt="Screenshot 2025-10-31 at 14 21 34" src="https://github.com/user-attachments/assets/8d2def91-5186-49e1-ab41-953f80dd0cc8" />
![IMG_FC5B84776032-1](https://github.com/user-attachments/assets/88c84856-f4fb-4e05-be57-ecb51194f32f)

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact / Contributors

Rushit Kansara
Aniket Kumar
Bhavya
Nitish Kumar Yadav
