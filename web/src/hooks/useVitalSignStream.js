import { useState, useEffect, useRef, useCallback } from 'react';
import { getSimulationTick, initSimulation, triggerEmergency } from '../services/api';

const VITAL_SIGN_HISTORY_LENGTH = 600;

// Hard-coded guaranteed default state
const DEFAULT_STREAM_DATA = {
    status: 'stable',
    mode: 'stable',
    simulation_active: false,
    emergency_scenario: null,
    prediction: { rationale: 'Analyzing...', confidence: 0, label: 'Stable' },
    alertStage: 'green',
    heart_rate: [],
    systolic_bp: [],
    diastolic_bp: [],
    spo2: [],
    temperature: [],
    resp_rate: [],
    blood_glucose: [],
    hasInitialData: false // Flag for UI sync
};

const useVitalSignStream = (patientId) => {
    // State is initialized synchronously with a guaranteed non-null object
    const [streamData, setStreamData] = useState(DEFAULT_STREAM_DATA);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const timerRef = useRef(null);

    const fetchTick = useCallback(async (isInitial = false) => {
        if (!patientId) return;
        try {
            // Request history only if this is an initial call (isInitial = true)
            const tickData = await getSimulationTick(patientId, isInitial);
            console.log("DEBUG: Raw API tickData:", tickData);
            if (!tickData) return;

            setStreamData(prev => {
                const updated = {
                    ...prev,
                    status: tickData.status || prev.status,
                    mode: tickData.mode || prev.mode,
                    simulation_active: tickData.simulation_active ?? prev.simulation_active,
                    emergency_scenario: tickData.emergency_scenario ?? prev.emergency_scenario,
                    prediction: tickData.prediction ? { ...tickData.prediction } : prev.prediction,
                    alertStage: tickData.alertStage || prev.alertStage
                };

                const expectedKeys = ['heart_rate','systolic_bp','diastolic_bp','spo2','temperature','resp_rate','blood_glucose'];
                const currentTimestamp = new Date(tickData.ts);
// History Injection
if (tickData.history && Array.isArray(tickData.history) && (!updated['heart_rate'] || updated['heart_rate'].length <= 1)) {
    console.log("DEBUG: Injecting prequel history of length:", tickData.history.length);
    tickData.history.forEach((dataPoint, index) => {
        const timestamp = new Date(currentTimestamp.getTime() - (tickData.history.length - index) * 1000);
        expectedKeys.forEach(key => {
            if (!updated[key]) updated[key] = [];
            if (dataPoint.hasOwnProperty(key)) updated[key].push({ x: timestamp, y: dataPoint[key] });
        });
    });
    updated.hasInitialData = true; 
    console.log("DEBUG: Prequel injection complete. New heart_rate length:", updated['heart_rate']?.length);
    return updated;
}

                // Tick Update (Only runs if history wasn't just injected)
                if (tickData.vitals) {
                    expectedKeys.forEach(key => {
                        if (tickData.vitals.hasOwnProperty(key)) {
                            const currentSeries = updated[key] ? [...updated[key]] : [];
                            currentSeries.push({ x: currentTimestamp, y: tickData.vitals[key] });
                            if (currentSeries.length > VITAL_SIGN_HISTORY_LENGTH) currentSeries.shift();
                            updated[key] = currentSeries;
                        }
                    });
                }
                return updated;
            });
        } catch (err) {
            console.error("Tick error:", err);
            setError(err.message);
        }
    }, [patientId]);

    const scheduleNextTick = useCallback(() => {
        if (timerRef.current) clearTimeout(timerRef.current);
        timerRef.current = setTimeout(async () => {
            await fetchTick();
            scheduleNextTick();
        }, 1000);
    }, [fetchTick]);

    const initializeSimulation = useCallback(async (profile) => {
        try {
            setLoading(true);
            await initSimulation(profile);
            setError(null);
            await fetchTick(true); // Pass true to request history
            scheduleNextTick();
            setLoading(false);
        } catch (err) {
            setError(err.message);
            setLoading(false);
        }
    }, [fetchTick, scheduleNextTick]);

    const abortSimulation = useCallback(async (pId) => {
        await fetch(`http://localhost:8000/api/v1_2/sim/abort/${pId}`, { method: 'POST' });
        await fetchTick(false);
    }, [fetchTick]);

    const triggerEmergencyForPatient = useCallback(async (emergencyId) => {
        await triggerEmergency(patientId, emergencyId);
        await fetchTick(false);
        scheduleNextTick();
    }, [patientId, fetchTick, scheduleNextTick]);

    useEffect(() => {
        if (patientId) scheduleNextTick();
        return () => { if (timerRef.current) clearTimeout(timerRef.current); };
    }, [patientId, scheduleNextTick]);

    return { streamData, loading, error, initializeSimulation, triggerEmergencyForPatient, abortSimulation, fetchTick };
};

export default useVitalSignStream;
