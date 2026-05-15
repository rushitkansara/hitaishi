# Hitaishi: AI-Powered Health Monitoring System

## Project Tagline / Value Proposition
Revolutionizing remote patient care with real-time AI diagnostics for home-isolated patients.

## Project Overview
Hitaishi is an innovative project demonstrating an end-to-end machine learning pipeline for **real-time health risk prediction** in home-isolated patients. It leverages a lightweight LSTM model to detect 13 critical medical conditions from continuous vital sign sequences. The system is designed for resource-constrained environments, providing accessible, AI-powered diagnostic support through a web interface and real-time SMS alerts.

## Key Features & Innovations
*   **High-Fidelity Clinical Simulation:** Advanced data pipeline modeling 13 medical conditions with realistic temporal progression and physiological noise.
*   **Lightweight LSTM Model:** Optimized for low-latency CPU inference, providing robust risk stratification in resource-constrained environments.
*   **Real-time Alerts:** Integrated with Twilio for immediate SMS notifications of emergency conditions.
*   **Web-based Dashboard:** Intuitive React frontend for patient monitoring, dynamic patient management, and interactive simulation.

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
The Hitaishi pipeline covers the entire lifecycle from synthetic data generation to real-time inference and alerting. For a detailed technical deep dive into the physiological modeling and data validation logic, please refer to [DATA_VALIDATION_REWRITTEN.md](./DATA_VALIDATION_REWRITTEN.md).

### 1. Data Generation
The system uses a sophisticated simulation engine to generate thousands of unique physiological trajectories across 13 medical conditions (Stable, Sepsis, STEMI, Shock, etc.).

### 2. Model Training & Quantization
*   **Architecture:** 4-layer LSTM optimized for sequence classification.
*   **Performance:** Achieves high recall on life-threatening emergencies.
*   **Optimization:** Supports TFLite quantization (INT8) for edge deployment, reducing model size to ~35 KB.

### 3. Runtime System
*   **Inference Engine:** Loads the trained model and processes incoming vital sign streams.
*   **Alert System:** Triggers Twilio SMS notifications when critical risk thresholds are exceeded.

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
