
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from typing import Dict, List, Any

from .engine import VitalsGeneratorV2
from .clinical_logic import ClinicalStateEngine

def visualize_v2_samples(output_path: str = 'v2_clinical_gallery.png'):
    """
    Generates 3 samples for each of the 13 classes using VitalsGeneratorV2
    and creates a comprehensive, beautified clinical gallery.
    """
    print("--- Generating Beautified V2.0 Clinical Gallery (600s episodes) ---")
    
    # Medical Color Scheme (High Contrast)
    COLORS = {
        'heart_rate': '#d35400', # Strong Orange-Red
        'systolic_bp': '#2980b9', # Blue
        'diastolic_bp': '#3498db', # Light Blue
        'spo2': '#27ae60', # Nephritis Green
        'temperature': '#c0392b', # Deep Red
        'respiratory_rate': '#8e44ad', # Wisteria Purple
        'blood_glucose': '#2c3e50'  # Midnight Blue/Gray
    }

    states = ClinicalStateEngine.STATES
    num_samples = 3
    
    # Setup ultra-wide figure for 10-minute resolution (96 inches wide)
    fig = plt.figure(figsize=(96, len(states) * 7))
    fig.patch.set_facecolor('white')
    
    title_text = 'Hitaishi V2.0: High-Fidelity Physiological Telemetry Gallery\n' \
                 '600s Multi-Layer Stochastic Simulation @ 1Hz'
    fig.suptitle(title_text, fontsize=32, fontweight='bold', y=0.99, color='#2c3e50')

    for state_code, state_name in states.items():
        print(f"  Generating samples for: {state_name}...")
        for s_idx in range(num_samples):
            # 1. Generate Episode
            age = random.uniform(20, 85)
            gender = random.choice(['male', 'female'])
            fitness = random.uniform(0.1, 0.9)
            
            gen = VitalsGeneratorV2(age=age, gender=gender, initial_state=state_code, fitness=fitness)
            data, meta = gen.generate_episode()
            
            # --- Panel A: Hemodynamics (HR, BP) ---
            ax_hemo = fig.add_subplot(len(states), num_samples * 3, (state_code * num_samples * 3) + (s_idx * 3) + 1)
            ax_hemo.plot(data[:, 0], color=COLORS['heart_rate'], lw=1.2, label='HR (BPM)')
            ax_hemo.plot(data[:, 1], color=COLORS['systolic_bp'], lw=1.0, ls='--', label='SBP (mmHg)')
            ax_hemo.plot(data[:, 2], color=COLORS['diastolic_bp'], lw=1.0, ls=':', label='DBP (mmHg)')
            
            if s_idx == 0: 
                ax_hemo.set_ylabel(f"CLASS: {state_name.replace('_', ' ')}\n\nHemodynamics", 
                                  fontsize=16, fontweight='bold', color='#2c3e50')
            
            ax_hemo.set_title(f"Sample {s_idx+1}: {int(age)}yo {gender}", fontsize=14, loc='left')
            ax_hemo.grid(True, alpha=0.2, linestyle='--')
            ax_hemo.legend(loc='upper right', fontsize=10, frameon=True, facecolor='white', framealpha=0.8)
            ax_hemo.tick_params(labelsize=10)

            # --- Panel B: Respiratory (SpO2, RR) ---
            ax_resp = fig.add_subplot(len(states), num_samples * 3, (state_code * num_samples * 3) + (s_idx * 3) + 2)
            ln1 = ax_resp.plot(data[:, 3], color=COLORS['spo2'], lw=1.2, label='SpO2 (%)')
            ax_resp.set_ylim(65, 105)
            ax_resp.set_ylabel('SpO2 (%)', color=COLORS['spo2'], fontsize=12)
            
            ax_resp_rr = ax_resp.twinx()
            ln2 = ax_resp_rr.plot(data[:, 5], color=COLORS['respiratory_rate'], lw=1.0, label='RR (br/min)')
            ax_resp_rr.set_ylabel('RR (br/min)', color=COLORS['respiratory_rate'], fontsize=12)
            
            ax_resp.set_title("Respiratory Dynamics", fontsize=14)
            ax_resp.grid(True, alpha=0.2, linestyle='--')
            
            # Combined Legend for twinx
            lns = ln1 + ln2
            labs = [l.get_label() for l in lns]
            ax_resp.legend(lns, labs, loc='upper right', fontsize=10, frameon=True, facecolor='white')

            # --- Panel C: Metabolic (Temp & Glucose) ---
            ax_meta = fig.add_subplot(len(states), num_samples * 3, (state_code * num_samples * 3) + (s_idx * 3) + 3)
            ln3 = ax_meta.plot(data[:, 4], color=COLORS['temperature'], lw=1.2, label='Temp (°C)')
            ax_meta.set_ylabel('Temp (°C)', color=COLORS['temperature'], fontsize=12)
            
            ax_meta_gl = ax_meta.twinx()
            ln4 = ax_meta_gl.plot(data[:, 6], color=COLORS['blood_glucose'], lw=1.0, label='Glucose (mg/dL)')
            ax_meta_gl.set_ylabel('Glucose (mg/dL)', color=COLORS['blood_glucose'], fontsize=12)
            
            ax_meta.set_title("Metabolic Trends", fontsize=14)
            ax_meta.grid(True, alpha=0.2, linestyle='--')
            
            # Combined Legend for twinx
            lns2 = ln3 + ln4
            labs2 = [l.get_label() for l in lns2]
            ax_meta.legend(lns2, labs2, loc='upper right', fontsize=10, frameon=True, facecolor='white')
            
            if state_code == len(states) - 1:
                ax_hemo.set_xlabel('Time (seconds)', fontsize=12)
                ax_resp.set_xlabel('Time (seconds)', fontsize=12)
                ax_meta.set_xlabel('Time (seconds)', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"--- Beautified Gallery saved to {output_path} ---")

if __name__ == '__main__':
    visualize_v2_samples()
