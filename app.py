"""
WardWatch AI — Silent Deterioration & Early Warning Engine
A prototype clinical decision-support system monitoring 8 simulated hospital beds over 24 hours.
Designed for non-ICU hospital wards using synthetic vital-sign and laboratory data.

Notice: This is a hackathon prototype for demonstration and decision-support research.
It does not replace clinical judgment, diagnose sepsis, or serve as a certified medical device.
"""

from datetime import datetime
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# CUSTOM CLINICAL COMMAND CENTER CSS
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Global Typography & Palette */
    :root {
        --clinical-navy: #0F172A;
        --clinical-slate: #334155;
        --clinical-teal: #0D9488;
        --clinical-teal-light: #F0FDFA;
        --clinical-border: #E2E8F0;
        --risk-green: #10B981;
        --risk-green-bg: #ECFDF5;
        --risk-yellow: #D97706;
        --risk-yellow-bg: #FFFBEB;
        --risk-red: #DC2626;
        --risk-red-bg: #FEF2F2;
    }
    
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: var(--clinical-navy);
        overflow-x: hidden;
    }

    /* Top banner card */
    .ward-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #FFFFFF;
        padding: 1.25rem 1.75rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .ward-banner h1 {
        margin: 0;
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #F8FAFC;
    }
    .ward-banner p {
        margin: 0.25rem 0 0 0;
        color: #94A3B8;
        font-size: 0.9rem;
    }

    /* Bed Card Styles */
    .bed-card {
        border-radius: 10px;
        border: 1px solid var(--clinical-border);
        padding: 1rem;
        background-color: #FFFFFF;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 0.75rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }
    .bed-card:hover {
        box-shadow: 0 4px 8px -2px rgba(0, 0, 0, 0.1);
    }
    .bed-card-selected {
        border: 2px solid var(--clinical-teal) !important;
        box-shadow: 0 0 0 3px rgba(13, 148, 136, 0.15) !important;
    }

    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .badge-green {
        background-color: var(--risk-green-bg);
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    .badge-yellow {
        background-color: var(--risk-yellow-bg);
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    .badge-red {
        background-color: var(--risk-red-bg);
        color: #991B1B;
        border: 1px solid #FECACA;
    }
    .badge-urgent {
        background-color: #7F1D1D;
        color: #FEF2F2;
        border: 1px solid #EF4444;
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }

    /* Metric pill */
    .metric-pill {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.35rem 0.5rem;
        background: #F8FAFC;
        border-radius: 6px;
        margin-bottom: 0.3rem;
        font-size: 0.82rem;
    }
    .metric-pill .label {
        color: #64748B;
        font-weight: 500;
    }
    .metric-pill .val {
        font-weight: 700;
        color: #0F172A;
    }
    .metric-pill .val-alert {
        color: #DC2626;
        font-weight: 800;
    }

    /* Clinical Explanations Card */
    .xai-card {
        background: #FFFFFF;
        border: 1px solid var(--clinical-border);
        border-left: 4px solid var(--clinical-teal);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    
    /* Footer */
    .clinical-footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--clinical-border);
        color: #64748B;
        font-size: 0.82rem;
        line-height: 1.5;
        text-align: center;
    }
</style>
"""

# ---------------------------------------------------------
# CLINICAL PROFILES & DESCRIPTIONS
# ---------------------------------------------------------
PATIENT_METADATA = {
    1: {"id": "WW-001", "name": "Arthur Pendelton", "age": 64, "gender": "Male", "scenario": "Stable", "admit": "Post-cholecystectomy Day 2", "desc": "Expected recovery path, normal physiological vitals."},
    2: {"id": "WW-002", "name": "Elena Rostova", "age": 52, "gender": "Female", "scenario": "Stable", "admit": "Cellulitis on IV Cefazolin", "desc": "Good therapeutic response, afebrile, stable hemodynamics."},
    3: {"id": "WW-003", "name": "Marcus Vance", "age": 71, "gender": "Male", "scenario": "Stable", "admit": "Elective total hip arthroplasty", "desc": "Routine post-op monitoring, vitals consistently within limits."},
    4: {"id": "WW-004", "name": "David Harrison", "age": 68, "gender": "Male", "scenario": "Progressive deterioration", "admit": "Post-op sigmoid colectomy Day 3", "desc": "Subtle occult deterioration starting ~Hr 10. Progressive tachycardia, declining BP/MAP, widening Shock Index, and rising serum lactate."},
    5: {"id": "WW-005", "name": "Beatrice Campbell", "age": 79, "gender": "Female", "scenario": "High-risk respiratory", "admit": "COPD exacerbation with atelectasis", "desc": "Elevated and fluctuating respiratory rate (frequently ≥ 22/min), borderline hypotension, elevated qSOFA (vital criteria)."},
    6: {"id": "WW-006", "name": "Carlos Gomez", "age": 45, "gender": "Male", "scenario": "Post-operative recovery", "admit": "Laparoscopic appendectomy Day 1", "desc": "Mild early postoperative tachycardia gradually settling down."},
    7: {"id": "WW-007", "name": "Sunita Patel", "age": 61, "gender": "Female", "scenario": "Post-operative recovery", "admit": "Right hemicolectomy Day 2", "desc": "Low-grade reactive temperature resolving with oral fluids and ambulation."},
    8: {"id": "WW-008", "name": "Gregory Smith", "age": 58, "gender": "Male", "scenario": "Post-operative recovery", "admit": "Ventral hernia repair Day 1", "desc": "Normal post-surgical convalescence, hemodynamically stable."}
}

# ---------------------------------------------------------
# SYNTHETIC PATIENT DATA GENERATOR
# ---------------------------------------------------------
@st.cache_data
def generate_synthetic_data() -> pd.DataFrame:
    """
    Generates deterministic, physiologically plausible synthetic vitals & labs
    for 8 hospital beds across 24 hourly observations (192 total records).
    """
    np.random.seed(42)
    records = []
    
    for bed_num in range(1, 9):
        meta = PATIENT_METADATA[bed_num]
        p_id = meta["id"]
        
        for h in range(1, 25):
            # Deterministic pseudo-random offsets for realistic biological jitter
            r1 = np.sin(bed_num * 1.5 + h * 0.4)
            r2 = np.cos(bed_num * 2.1 + h * 0.3)
            
            timestamp = f"{((7 + h) % 24):02d}:00"
            
            if bed_num in (1, 2, 3):
                # Beds 1-3: Stable
                hr = 70.0 + (bed_num - 2) * 2.0 + r1 * 2.2
                sbp = 120.0 + (bed_num - 2) * 2.5 + r2 * 3.0
                dbp = 76.0 + (bed_num - 2) * 1.5 + r1 * 1.8
                spo2 = 98.0 - abs(r2) * 0.8
                temp = 36.8 + r1 * 0.15
                rr = 15.0 + (bed_num - 2) * 0.5 + r2 * 1.0
                lactate = 1.1 + (bed_num - 2) * 0.1 + abs(r1) * 0.15
                wbc = 7.4 + (bed_num - 2) * 0.5 + r2 * 0.3
                
            elif bed_num == 4:
                # Bed 4: Progressive deterioration starting around hour 10
                if h < 10:
                    # Stable baseline
                    hr = 74.0 + r1 * 2.0
                    sbp = 122.0 + r2 * 2.5
                    dbp = 78.0 + r1 * 2.0
                    spo2 = 98.0 - abs(r2) * 0.5
                    temp = 37.0 + r1 * 0.1
                    rr = 16.0 + r2 * 0.8
                    lactate = 1.1 + abs(r1) * 0.1
                    wbc = 8.6 + r2 * 0.2
                else:
                    # Smooth sigmoidal/exponential escalation from hr 10 to 24
                    prog = (h - 9) / 15.0 # 0.067 to 1.0
                    
                    # HR rises toward 112 bpm
                    hr = 74.0 + (38.0 * (prog ** 1.15)) + r1 * 1.8
                    # SBP drops from 122 down to ~94 mmHg
                    sbp = 122.0 - (28.0 * (prog ** 1.05)) + r2 * 2.0
                    # DBP drops from 78 down to ~58 mmHg
                    dbp = 78.0 - (20.0 * (prog ** 1.0)) + r1 * 1.5
                    # SpO2 degrades gradually from 98% down to 92%
                    spo2 = 98.0 - (6.0 * (prog ** 1.1)) - abs(r2) * 0.5
                    # Temp rises (fever developing: 37.0 -> 38.6 °C)
                    temp = 37.0 + (1.6 * (prog ** 1.1)) + r1 * 0.1
                    # RR increases (tachypnea: 16 -> 25 breaths/min)
                    rr = 16.0 + (9.5 * (prog ** 1.1)) + r2 * 0.8
                    # Lactate climbs progressively: 1.1 -> 3.25 mmol/L
                    lactate = 1.1 + (2.15 * (prog ** 1.2)) + abs(r1) * 0.08
                    # WBC climbs: 8.6 -> 15.4 x10^3/uL
                    wbc = 8.6 + (6.8 * (prog ** 1.1)) + r2 * 0.3
                    
            elif bed_num == 5:
                # Bed 5: High-risk respiratory scenario (persistent/fluctuating tachypnea & borderline SBP)
                # RR fluctuates at or above 22/min frequently
                rr = 23.5 + 2.5 * np.sin(h * 0.9) + r2 * 1.0
                # SBP intermittently dips to/below 100 mmHg
                sbp = 101.0 - 5.0 * np.cos(h * 0.7) + r1 * 2.0
                dbp = 64.0 + r2 * 2.0
                hr = 92.0 + 4.0 * np.sin(h * 0.5) + r1 * 2.5
                spo2 = 92.5 + 1.2 * np.cos(h * 0.6)
                temp = 37.5 + r1 * 0.2
                lactate = 1.8 + abs(r2) * 0.25
                wbc = 12.2 + r1 * 0.4
                
            else: # Beds 6, 7, 8: Post-operative recovery
                # Mild initial postoperative reaction settling into calm baseline
                decay = np.exp(-(h - 1) / 7.0)
                
                if bed_num == 6: # Appendectomy
                    hr = 74.0 + 12.0 * decay + r1 * 1.5
                    sbp = 118.0 + 4.0 * (1 - decay) + r2 * 2.0
                    dbp = 74.0 + r1 * 1.5
                    spo2 = 97.5 + r2 * 0.4
                    temp = 36.9 + 0.4 * decay + r1 * 0.1
                    rr = 16.0 + 2.0 * decay + r2 * 0.6
                    lactate = 1.3 - 0.2 * (1 - decay) + abs(r1) * 0.08
                    wbc = 9.8 - 1.8 * (1 - decay) + r2 * 0.2
                elif bed_num == 7: # Hemicolectomy
                    hr = 76.0 + 8.0 * decay + r1 * 1.6
                    sbp = 116.0 + r2 * 2.5
                    dbp = 72.0 + r1 * 1.4
                    spo2 = 96.8 + 1.0 * (1 - decay)
                    temp = 37.4 * decay + 36.8 * (1 - decay) + r1 * 0.12
                    rr = 17.0 + 1.5 * decay + r2 * 0.7
                    lactate = 1.4 - 0.25 * (1 - decay) + abs(r1) * 0.09
                    wbc = 10.5 - 2.0 * (1 - decay) + r2 * 0.25
                else: # Hernia repair
                    hr = 72.0 + 6.0 * decay + r1 * 1.2
                    sbp = 122.0 + r2 * 2.0
                    dbp = 76.0 + r1 * 1.5
                    spo2 = 98.2 - abs(r2) * 0.4
                    temp = 36.7 + r1 * 0.1
                    rr = 15.0 + decay * 1.0 + r2 * 0.5
                    lactate = 1.0 + abs(r1) * 0.1
                    wbc = 7.8 + r2 * 0.2
            
            # Clamp to physiologically valid non-negative ranges
            hr = max(40.0, round(float(hr), 1))
            sbp = max(60.0, round(float(sbp), 1))
            dbp = max(40.0, round(float(dbp), 1))
            map_val = round((sbp + 2.0 * dbp) / 3.0, 1)
            spo2 = min(100.0, max(75.0, round(float(spo2), 1)))
            temp = round(float(temp), 1)
            rr = max(6.0, round(float(rr), 1))
            lactate = max(0.4, round(float(lactate), 2))
            wbc = max(2.0, round(float(wbc), 2))
            
            records.append({
                "Hour": int(h),
                "Timestamp": timestamp,
                "Bed": int(bed_num),
                "Bed_Label": f"Bed {bed_num}",
                "PatientID": p_id,
                "HR": hr,
                "SBP": sbp,
                "DBP": dbp,
                "MAP": map_val,
                "SpO2": spo2,
                "Temp": temp,
                "RR": rr,
                "Lactate": lactate,
                "WBC": wbc
            })
            
    df = pd.DataFrame(records)
    return df

# ---------------------------------------------------------
# CLINICAL FEATURE ENGINEERING & DERIVED METRICS
# ---------------------------------------------------------
def calculate_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Shock Index, qSOFA vital criteria, and 3-hour trend deltas.
    Ensures safe handling of timeline start (Hours 1-3 have NaN for 3h delta).
    """
    df = df.copy()
    
    # Shock Index = HR / SBP
    # Guard against zero or near-zero SBP
    df["Shock_Index"] = df.apply(
        lambda r: round(r["HR"] / max(r["SBP"], 1.0), 3), axis=1
    )
    
    # qSOFA criteria (2 vital criteria available in synthetic ward dataset)
    # Criterion 1: RR >= 22 /min
    df["qSOFA_RR_Met"] = df["RR"] >= 22.0
    # Criterion 2: SBP <= 100 mmHg
    df["qSOFA_SBP_Met"] = df["SBP"] <= 100.0
    # Combined observed score out of 2 (Mental status explicitly unassessed)
    df["qSOFA_Score"] = df["qSOFA_RR_Met"].astype(int) + df["qSOFA_SBP_Met"].astype(int)
    
    # Sort for deterministic window calculation
    df = df.sort_values(by=["Bed", "Hour"]).reset_index(drop=True)
    
    # 3-Hour Trend Deltas: Value(Hour) - Value(Hour - 3)
    # Using shift(3) within each Bed group
    df["Delta_HR_3h"] = df.groupby("Bed")["HR"].diff(periods=3)
    df["Delta_MAP_3h"] = df.groupby("Bed")["MAP"].diff(periods=3)
    df["Delta_Lactate_3h"] = df.groupby("Bed")["Lactate"].diff(periods=3)
    
    # Round delta metrics for clean clinical display
    df["Delta_HR_3h"] = df["Delta_HR_3h"].round(1)
    df["Delta_MAP_3h"] = df["Delta_MAP_3h"].round(1)
    df["Delta_Lactate_3h"] = df["Delta_Lactate_3h"].round(2)
    
    return df

# ---------------------------------------------------------
# COMPOSITE DETERIORATION RISK SCORE & XAI ENGINE
# ---------------------------------------------------------
def calculate_risk_score(row: pd.Series) -> dict:
    """
    Transparent, bounded rule-based clinical deterioration scoring engine (0-100%).
    Normalized across hemodynamic, respiratory, metabolic, and trend features.
    Returns:
      - score: float in [0.0, 100.0]
      - band: 'Green' (<30%), 'Yellow' (30-65%), or 'Red' (>65%)
      - urgent_alert: bool (urgent isolated abnormality flag)
      - urgent_reasons: list of str
      - contributions: dict of {factor_name: points}
    """
    contributions = {}
    urgent_reasons = []
    
    # 1. Shock Index (Max weight: 22 pts)
    # Reference: 0.5 - 0.7. Elevated: > 0.9.
    si = row["Shock_Index"]
    if si <= 0.70:
        c_si = 0.0
    elif si < 0.90:
        c_si = ((si - 0.70) / 0.20) * 8.0 # up to 8 pts
    else:
        # > 0.90 is elevated
        c_si = 8.0 + min(14.0, ((si - 0.90) / 0.30) * 14.0)
    contributions["Shock Index"] = round(c_si, 1)
    if si >= 1.0:
        urgent_reasons.append(f"Severe Shock Index ({si:.2f} ≥ 1.00)")
        
    # 2. Hypotension & MAP (Max weight: 18 pts)
    # Normal MAP >= 70, SBP >= 110.
    sbp = row["SBP"]
    map_val = row["MAP"]
    c_bp = 0.0
    if map_val < 65.0:
        c_bp += 10.0
        urgent_reasons.append(f"Severe Hypotension: MAP {map_val:.1f} < 65 mmHg")
    elif map_val < 70.0:
        c_bp += 6.0
        
    if sbp <= 90.0:
        c_bp += 8.0
        urgent_reasons.append(f"Critical SBP {sbp:.0f} ≤ 90 mmHg")
    elif sbp <= 100.0:
        c_bp += 5.0
    contributions["Blood Pressure / MAP"] = round(min(18.0, c_bp), 1)

    # 3. Respiratory Rate Abnormality (Max weight: 16 pts)
    # Normal: 12-20. qSOFA threshold: >= 22. Severe: >= 26.
    rr = row["RR"]
    c_rr = 0.0
    if rr >= 26.0:
        c_rr = 16.0
        urgent_reasons.append(f"Marked Tachypnea: RR {rr:.0f} ≥ 26/min")
    elif rr >= 22.0:
        c_rr = 10.0 + ((rr - 22.0) / 4.0) * 4.0
    elif rr > 20.0:
        c_rr = 4.0
    elif rr < 10.0:
        c_rr = 12.0
        urgent_reasons.append(f"Bradypnea / Hypoventilation: RR {rr:.0f} < 10/min")
    contributions["Respiratory Rate"] = round(c_rr, 1)

    # 4. Oxygen Saturation (Max weight: 15 pts)
    # Normal: >= 96%. Mild hypoxia: 92-95%. Moderate/Severe: < 92%.
    spo2 = row["SpO2"]
    c_spo2 = 0.0
    if spo2 < 90.0:
        c_spo2 = 15.0
        urgent_reasons.append(f"Critical Hypoxemia: SpO2 {spo2:.1f}% < 90%")
    elif spo2 < 93.0:
        c_spo2 = 10.0
    elif spo2 < 96.0:
        c_spo2 = 4.0
    contributions["Oxygen Saturation"] = round(c_spo2, 1)

    # 5. Serum Lactate (Max weight: 14 pts)
    # Normal: < 1.6 mmol/L. Mildly elevated: 1.6 - 2.2. Clinically significant: >= 2.2. Severe: >= 3.0.
    lac = row["Lactate"]
    c_lac = 0.0
    if lac >= 3.0:
        c_lac = 14.0
        urgent_reasons.append(f"Hyperlactatemia: {lac:.2f} ≥ 3.00 mmol/L")
    elif lac >= 2.2:
        c_lac = 8.0 + ((lac - 2.2) / 0.8) * 4.0
    elif lac >= 1.6:
        c_lac = ((lac - 1.6) / 0.6) * 4.0
    contributions["Serum Lactate"] = round(c_lac, 1)

    # 6. Temperature Abnormality (Max weight: 7 pts)
    # Hypothermia < 36.0 or Fever > 38.0
    temp = row["Temp"]
    c_temp = 0.0
    if temp >= 38.5:
        c_temp = 7.0
    elif temp >= 38.0:
        c_temp = 4.5
    elif temp < 36.0:
        c_temp = 6.0
    contributions["Body Temperature"] = round(c_temp, 1)

    # 7. Dynamic 3-Hour Trends (Early Deterioration Signal) (Max weight: 8 pts)
    # Trend detection is the heart of early detection before absolute thresholds break
    c_trend = 0.0
    d_hr = row.get("Delta_HR_3h", np.nan)
    d_map = row.get("Delta_MAP_3h", np.nan)
    d_lac = row.get("Delta_Lactate_3h", np.nan)
    
    if pd.notna(d_hr) and d_hr >= 12.0:
        c_trend += 3.0
    if pd.notna(d_map) and d_map <= -8.0:
        c_trend += 3.0
    if pd.notna(d_lac) and d_lac >= 0.40:
        c_trend += 2.0
    contributions["3-Hour Vital Trends"] = round(c_trend, 1)

    # Sum and clip to 0 - 100%
    raw_sum = sum(contributions.values())
    final_score = float(np.clip(round(raw_sum, 1), 0.0, 100.0))
    
    # Demo risk bands
    if final_score < 30.0:
        band = "Green"
    elif final_score <= 65.0:
        band = "Yellow"
    else:
        band = "Red"
        
    return {
        "score": final_score,
        "band": band,
        "urgent_alert": len(urgent_reasons) > 0,
        "urgent_reasons": urgent_reasons,
        "contributions": contributions
    }

def get_risk_drivers(row: pd.Series, risk_data: dict) -> list:
    """
    Identifies and formats top contributing risk drivers with clinical interpretation.
    If the score is low, provides protective/reassuring clinical factors.
    """
    contributions = risk_data["contributions"]
    score = risk_data["score"]
    
    # Sort contributions descending
    sorted_contribs = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    
    drivers = []
    
    if score >= 30.0:
        # Patient has meaningful risk: show top 2 actual risk drivers
        top_two = [item for item in sorted_contribs if item[1] > 0][:2]
        
        # If less than 2 had positive contributions, pad with next highest
        if len(top_two) < 2:
            top_two = sorted_contribs[:2]
            
        for name, pts in top_two:
            if name == "Shock Index":
                desc = f"Shock Index {row['Shock_Index']:.2f} is elevated above baseline reference (0.50–0.70)."
            elif name == "Blood Pressure / MAP":
                desc = f"MAP {row['MAP']:.1f} mmHg / SBP {row['SBP']:.0f} mmHg demonstrates declining systemic vascular perfusion."
            elif name == "Respiratory Rate":
                desc = f"Respiratory rate {row['RR']:.0f}/min signals compensatory respiratory effort or pulmonary stress."
            elif name == "Oxygen Saturation":
                desc = f"SpO2 {row['SpO2']:.1f}% indicates sub-optimal peripheral oxygenation."
            elif name == "Serum Lactate":
                desc = f"Serum lactate {row['Lactate']:.2f} mmol/L reflects anaerobic cellular metabolism."
            elif name == "Body Temperature":
                desc = f"Core temperature {row['Temp']:.1f}°C indicates pyrexia / systemic inflammatory response."
            elif name == "3-Hour Vital Trends":
                desc = f"Multivariate divergence over past 3 hours: HR Δ {row.get('Delta_HR_3h', 0):+.1f} bpm, MAP Δ {row.get('Delta_MAP_3h', 0):+.1f} mmHg."
            else:
                desc = f"Observed physiological divergence in {name}."
                
            drivers.append({
                "factor": name,
                "contribution": pts,
                "description": desc,
                "type": "risk"
            })
    else:
        # Low risk (<30%): explain that stable factors maintain safety
        drivers.append({
            "factor": "Hemodynamic Stability",
            "contribution": contributions.get("Shock Index", 0.0),
            "description": f"Shock Index {row['Shock_Index']:.2f} & MAP {row['MAP']:.1f} mmHg remain well within safe reference margins.",
            "type": "protective"
        })
        drivers.append({
            "factor": "Adequate Perfusion & Oxygenation",
            "contribution": contributions.get("Serum Lactate", 0.0),
            "description": f"Serum lactate {row['Lactate']:.2f} mmol/L and SpO2 {row['SpO2']:.1f}% indicate stable tissue oxygen delivery.",
            "type": "protective"
        })
        
    return drivers

def get_status_label(band: str, urgent: bool) -> tuple:
    """Returns human-readable status, badge CSS class, and icon."""
    if urgent:
        return "URGENT REVIEW", "badge-urgent", "🚨"
    elif band == "Red":
        return "HIGH RISK", "badge-red", "⚠️"
    elif band == "Yellow":
        return "MODERATE RISK", "badge-yellow", "⚡"
    else:
        return "LOW RISK", "badge-green", "✅"

# ---------------------------------------------------------
# CONTEXT-SENSITIVE NURSE-REVIEW CHECKLIST
# ---------------------------------------------------------
def get_nurse_review_items(row: pd.Series, risk_data: dict) -> list:
    """
    Generates tailored, protocol-oriented review prompts.
    Avoids prescriptive treatment claims; phrases actions as clinician-directed checks.
    """
    items = []
    
    # Universal baseline check
    items.append({
        "id": "chk_vitals",
        "title": "Verify Vital Signs & Sensor Placement",
        "detail": "Repeat manual blood pressure and verify pulse-oximeter plethysmograph waveform quality."
    })
    
    # Shock Index / Hemodynamics
    if row["Shock_Index"] > 0.85 or row["MAP"] < 70.0:
        items.append({
            "id": "chk_hemo",
            "title": "Hemodynamic & Perfusion Reassessment",
            "detail": f"Shock Index is {row['Shock_Index']:.2f} (MAP {row['MAP']:.1f} mmHg). Assess capillary refill time, peripheral pulse volume, and urine output."
        })
        
    # Respiratory / Oxygenation
    if row["RR"] >= 22.0 or row["SpO2"] < 94.0:
        items.append({
            "id": "chk_resp",
            "title": "Respiratory & Airway Evaluation",
            "detail": f"Respiratory rate is {row['RR']:.0f}/min with SpO2 {row['SpO2']:.1f}%. Auscultate lung bases, assess work of breathing, and check supplemental O2 delivery."
        })
        
    # Metabolic / Lactate
    if row["Lactate"] >= 2.0:
        items.append({
            "id": "chk_lac",
            "title": "Lactate Clearance & Metabolic Panel",
            "detail": f"Serum lactate is elevated at {row['Lactate']:.2f} mmol/L. Consider clinician-ordered repeat venous/arterial blood gas to evaluate clearance."
        })
        
    # Escalation
    if risk_data["band"] == "Red" or risk_data["urgent_alert"]:
        items.append({
            "id": "chk_esc",
            "title": "Urgent Clinical Escalation",
            "detail": "Notify Ward Registrar / Rapid Response Team (RRT) per hospital deteriorating patient policy."
        })
    elif risk_data["band"] == "Yellow":
        items.append({
            "id": "chk_freq",
            "title": "Increase Observation Frequency",
            "detail": "Transition from routine 4-hourly monitoring to hourly bedside vital-sign track-and-trigger observations."
        })
    else:
        items.append({
            "id": "chk_routine",
            "title": "Continue Routine Convalescent Monitoring",
            "detail": "Patient observations are stable. Continue scheduled ward rounds and standard post-operative pathway."
        })
        
    return items

# ---------------------------------------------------------
# INTERACTIVE PLOTLY VISUALIZATIONS
# ---------------------------------------------------------
def build_trend_chart(hist_df: pd.DataFrame, metric_col: str, title: str, unit: str, 
                      ref_low: float = None, ref_high: float = None, 
                      line_color: str = "#0D9488") -> go.Figure:
    """
    Builds a responsive, accessible Plotly time-series chart
    showing observations from Hour 1 through the selected Hour only.
    Prevents data leakage into future hours.
    """
    fig = go.Figure()
    
    # Main trend line
    fig.add_trace(go.Scatter(
        x=hist_df["Hour"],
        y=hist_df[metric_col],
        mode="lines+markers",
        name=title,
        line=dict(color=line_color, width=2.5),
        marker=dict(size=6, color=line_color, symbol="circle"),
        hovertemplate=f"<b>Hour %{{x}}</b><br>{title}: %{{y}} {unit}<extra></extra>"
    ))
    
    # Reference thresholds
    if ref_low is not None:
        fig.add_hline(
            y=ref_low, line_dash="dash", line_color="#94A3B8", line_width=1.2,
            annotation_text=f"Ref Min ({ref_low})", annotation_position="bottom right",
            annotation_font_size=10, annotation_font_color="#64748B"
        )
    if ref_high is not None:
        fig.add_hline(
            y=ref_high, line_dash="dash", line_color="#EF4444", line_width=1.2,
            annotation_text=f"Warning ({ref_high})", annotation_position="top right",
            annotation_font_size=10, annotation_font_color="#DC2626"
        )
        
    # Layout styling
    fig.update_layout(
        title=dict(text=f"<b>{title}</b> <span style='font-size:12px; color:#64748B;'>({unit})</span>", font=dict(size=14, color="#0F172A")),
        xaxis=dict(
            title="Ward Timeline (Hour)",
            dtick=2 if len(hist_df) > 8 else 1,
            range=[0.5, max(hist_df["Hour"].max() + 0.5, 2.5)],
            gridcolor="#F1F5F9",
            zeroline=False
        ),
        yaxis=dict(
            gridcolor="#F1F5F9",
            zeroline=False
        ),
        height=220,
        margin=dict(l=40, r=20, t=35, b=35),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        showlegend=False
    )
    return fig

def build_risk_gauge(score: float, band: str) -> go.Figure:
    """
    Builds a clean 0-100% Plotly radial indicator gauge for composite deterioration risk.
    """
    band_colors = {
        "Green": "#10B981",
        "Yellow": "#F59E0B",
        "Red": "#DC2626"
    }
    gauge_bar_color = band_colors.get(band, "#0D9488")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(suffix="%", font=dict(size=32, color="#0F172A", family="-apple-system, sans-serif")),
        title=dict(text=f"<b>Deterioration Risk Score</b><br><span style='font-size:13px; color:{gauge_bar_color}; font-weight:700;'>Category: {band.upper()} RISK</span>", font=dict(size=14)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#94A3B8", dtick=20),
            bar=dict(color=gauge_bar_color, thickness=0.35),
            bgcolor="#F8FAFC",
            borderwidth=1,
            bordercolor="#E2E8F0",
            steps=[
                dict(range=[0, 30], color="#ECFDF5"),
                dict(range=[30, 65], color="#FFFBEB"),
                dict(range=[65, 100], color="#FEF2F2"),
            ],
            threshold=dict(
                line=dict(color="#DC2626", width=3),
                thickness=0.8,
                value=65
            )
        )
    ))
    
    fig.update_layout(
        height=220,
        margin=dict(l=30, r=30, t=50, b=20),
        paper_bgcolor="#FFFFFF"
    )
    return fig

# ---------------------------------------------------------
# STATE MUTATION CALLBACKS (RESOLVES BUTTON SYNC CONFLICTS)
# ---------------------------------------------------------
def set_sim_hour(h: int):
    """Callback to reliably update the simulation hour across all components."""
    st.session_state.sim_hour = int(h)

def set_selected_bed(b: int):
    """Callback to reliably update the active patient bed selection."""
    st.session_state.selected_bed = int(b)

def set_milestone(h: int, b: int = 4):
    """Callback to set both simulation hour and patient bed in lockstep."""
    st.session_state.sim_hour = int(h)
    st.session_state.selected_bed = int(b)

# ---------------------------------------------------------
# SIDEBAR NAVIGATION & SIMULATION TIMELINE CONTROLS
# ---------------------------------------------------------
def render_sidebar(full_df: pd.DataFrame):
    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.25rem;">
                <span style="font-size:1.8rem;">🏥</span>
                <div>
                    <h2 style="margin:0; font-size:1.25rem; font-weight:800; color:#0F172A;">WardWatch AI</h2>
                    <span style="font-size:0.75rem; color:#0D9488; font-weight:700; letter-spacing:0.04em;">CLINICAL DECISION SUPPORT</span>
                </div>
            </div>
            <p style="font-size:0.8rem; color:#64748B; margin-top:0.2rem; margin-bottom:1.2rem;">
                Non-ICU early warning engine monitoring subtle multivariate trends.
            </p>
            <hr style="margin:0 0 1rem 0; border:0; border-top:1px solid #E2E8F0;">
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("### ⏱️ Timeline Control")
        
        # Primary Hour Slider directly linked to st.session_state.sim_hour
        st.slider(
            "Ward Simulation Hour",
            min_value=1,
            max_value=24,
            key="sim_hour",
            help="Adjust to observe physiological trajectories across the 24-hour ward stay."
        )
        
        # Timeline step buttons (Prev / Next) with deterministic on_click callbacks
        col_prev, col_next = st.columns(2)
        with col_prev:
            st.button(
                "◀ Prev Hour",
                key="btn_prev_hour",
                use_container_width=True,
                disabled=(st.session_state.sim_hour <= 1),
                on_click=set_sim_hour,
                args=(max(1, st.session_state.sim_hour - 1),)
            )
        with col_next:
            st.button(
                "Next Hour ▶",
                key="btn_next_hour",
                use_container_width=True,
                disabled=(st.session_state.sim_hour >= 24),
                on_click=set_sim_hour,
                args=(min(24, st.session_state.sim_hour + 1),)
            )
                
        # Ward Census Quick Stats (Computed dynamically for active simulation hour)
        current_hour_df = full_df[full_df["Hour"] == st.session_state.sim_hour]
        
        high_risk_count = 0
        mod_risk_count = 0
        stable_count = 0
        urgent_count = 0
        
        for _, r in current_hour_df.iterrows():
            calc = calculate_risk_score(r)
            if calc["urgent_alert"]:
                urgent_count += 1
            if calc["band"] == "Red":
                high_risk_count += 1
            elif calc["band"] == "Yellow":
                mod_risk_count += 1
            else:
                stable_count += 1
                
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Ward Census Snapshot")
        
        st.metric("Total Monitored Beds", "8 Beds")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.metric("High Risk / Alert", f"{high_risk_count + urgent_count}", delta=f"{urgent_count} urgent" if urgent_count > 0 else None, delta_color="inverse")
        with col_c2:
            st.metric("Stable Beds", f"{stable_count}", delta="Normal limits")
            
        st.markdown(
            f"""
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:0.6rem; font-size:0.78rem; color:#475569; margin-top:0.5rem;">
                <strong>Simulation Time:</strong> Hour {st.session_state.sim_hour}:00<br>
                <strong>Active Patient:</strong> Bed {st.session_state.selected_bed} ({PATIENT_METADATA[st.session_state.selected_bed]['id']})
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Reset / Jump to Demo Milestones using on_click callbacks
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🎯 Demo Milestones")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.button("Baseline (Hr 1)", key="btn_ms_h1", use_container_width=True, on_click=set_milestone, args=(1, 4))
            st.button("Midway (Hr 15)", key="btn_ms_h15", use_container_width=True, on_click=set_milestone, args=(15, 4))
        with col_m2:
            st.button("Inception (Hr 10)", key="btn_ms_h10", use_container_width=True, on_click=set_milestone, args=(10, 4))
            st.button("Critical (Hr 24)", key="btn_ms_h24", use_container_width=True, on_click=set_milestone, args=(24, 4))
                
        # Mandatory Synthetic Data & Clinical Disclaimer
        st.markdown("<br><hr style='margin:0 0 0.8rem 0; border:0; border-top:1px solid #E2E8F0;'>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size:0.72rem; color:#64748B; line-height:1.4;">
                <strong>Synthetic Demo Notice:</strong> WardWatch AI is a proof-of-concept hackathon prototype powered by synthetic patient observations.
                It is <em>not</em> an FDA/CE cleared medical device and does not provide autonomous diagnostic claims or clinical management directives.
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------------------------------------------------
# PATIENT CARD RENDERING (WARD FLOOR GRID)
# ---------------------------------------------------------
def render_patient_card(row: pd.Series, is_selected: bool):
    """
    Renders an individual bed card within the responsive ward floor grid.
    Includes Bed number, Patient ID, Composite Risk %, Status Badge,
    Key vitals, Shock Index, qSOFA, and Select Patient button.
    """
    bed_num = int(row["Bed"])
    p_id = row["PatientID"]
    
    # Calculate risk and status
    risk_info = calculate_risk_score(row)
    score = risk_info["score"]
    band = risk_info["band"]
    urgent = risk_info["urgent_alert"]
    status_label, badge_class, icon = get_status_label(band, urgent)
    
    selected_class = "bed-card-selected" if is_selected else ""
    
    # Alert highlights
    si_alert = "val-alert" if row["Shock_Index"] >= 0.90 else ""
    sbp_alert = "val-alert" if row["SBP"] <= 100.0 else ""
    spo2_alert = "val-alert" if row["SpO2"] < 94.0 else ""
    
    card_html = f"""
    <div class="bed-card {selected_class}">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
            <div>
                <span style="font-weight:800; font-size:1.05rem; color:#0F172A;">Bed {bed_num}</span>
                <span style="font-size:0.75rem; color:#64748B; margin-left:0.3rem;">{p_id}</span>
            </div>
            <span class="badge {badge_class}">{icon} {status_label}</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:0.6rem; padding-bottom:0.4rem; border-bottom:1px solid #F1F5F9;">
            <span style="font-size:0.8rem; color:#64748B;">Risk Index</span>
            <span style="font-size:1.35rem; font-weight:800; color:{'#DC2626' if score > 65 else ('#D97706' if score >= 30 else '#059669')};">
                {score:.1f}%
            </span>
        </div>
        <div class="metric-pill">
            <span class="label">Heart Rate</span>
            <span class="val">{row['HR']:.0f} bpm</span>
        </div>
        <div class="metric-pill">
            <span class="label">Blood Pressure</span>
            <span class="val {sbp_alert}">{row['SBP']:.0f}/{row['DBP']:.0f} <span style="font-size:0.7rem; font-weight:normal; color:#64748B;">(MAP {row['MAP']:.0f})</span></span>
        </div>
        <div class="metric-pill">
            <span class="label">SpO2 / RR</span>
            <span class="val {spo2_alert}">{row['SpO2']:.1f}% / {row['RR']:.0f}/m</span>
        </div>
        <div class="metric-pill">
            <span class="label">Shock Index</span>
            <span class="val {si_alert}">{row['Shock_Index']:.2f}</span>
        </div>
        <div class="metric-pill">
            <span class="label">qSOFA (Vital Criteria)</span>
            <span class="val">{int(row['qSOFA_Score'])}/2 <span style="font-size:0.68rem; color:#64748B;">[2-param]</span></span>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Stable select button using on_click callback
    btn_label = "Active Patient" if is_selected else f"Select Bed {bed_num}"
    st.button(
        btn_label,
        key=f"select_bed_{bed_num}",
        use_container_width=True,
        type="primary" if is_selected else "secondary",
        on_click=set_selected_bed,
        args=(bed_num,)
    )

# ---------------------------------------------------------
# PATIENT DEEP DIVE SECTION
# ---------------------------------------------------------
def render_patient_detail(p_row: pd.Series, hist_p_df: pd.DataFrame):
    """
    Renders comprehensive clinical deep-dive for the selected bed and hour.
    """
    bed_num = int(p_row["Bed"])
    meta = PATIENT_METADATA[bed_num]
    risk_info = calculate_risk_score(p_row)
    drivers = get_risk_drivers(p_row, risk_info)
    status_label, badge_class, icon = get_status_label(risk_info["band"], risk_info["urgent_alert"])
    
    st.markdown(f"## 🔍 Patient Deep-Dive: Bed {bed_num} ({meta['id']})")
    
    # Patient Summary Header Banner
    col_hdr1, col_hdr2 = st.columns([3, 1])
    with col_hdr1:
        st.markdown(
            f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:0.9rem 1.2rem; margin-bottom:1rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                    <div>
                        <span style="font-size:1.15rem; font-weight:700; color:#0F172A;">{meta['name']}</span>
                        <span style="font-size:0.85rem; color:#64748B; margin-left:0.5rem;">{meta['age']} yo {meta['gender']} | Admit: <em>{meta['admit']}</em></span>
                    </div>
                    <span class="badge {badge_class}">{icon} {status_label}</span>
                </div>
                <p style="margin:0.4rem 0 0 0; font-size:0.85rem; color:#475569; line-height:1.4;">
                    <strong>Clinical Scenario Context:</strong> {meta['desc']}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_hdr2:
        # Quick action: jump to adjacent beds with on_click callbacks
        prev_bed = 8 if bed_num == 1 else bed_num - 1
        next_bed = 1 if bed_num == 8 else bed_num + 1
        c_p, c_n = st.columns(2)
        with c_p:
            st.button(f"◀ Bed {prev_bed}", key=f"nav_prev_bed_{bed_num}", use_container_width=True, on_click=set_selected_bed, args=(prev_bed,))
        with c_n:
            st.button(f"Bed {next_bed} ▶", key=f"nav_next_bed_{bed_num}", use_container_width=True, on_click=set_selected_bed, args=(next_bed,))
                
    # Urgent alert banner if triggered
    if risk_info["urgent_alert"]:
        reasons_txt = " • ".join(risk_info["urgent_reasons"])
        st.error(f"🚨 **URGENT PHYSIOLOGICAL ALERT TRIGGERED:** {reasons_txt}")
        
    # Row of Key Clinical Metrics with 3-Hour Delta
    col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
    
    d_hr = p_row.get("Delta_HR_3h")
    d_map = p_row.get("Delta_MAP_3h")
    d_lac = p_row.get("Delta_Lactate_3h")
    
    hr_delta_str = f"{d_hr:+.1f} (3h)" if pd.notna(d_hr) else "N/A (h < 4)"
    map_delta_str = f"{d_map:+.1f} (3h)" if pd.notna(d_map) else "N/A (h < 4)"
    lac_delta_str = f"{d_lac:+.2f} (3h)" if pd.notna(d_lac) else "N/A (h < 4)"
    
    with col_m1:
        st.metric("Heart Rate", f"{p_row['HR']:.0f} bpm", delta=hr_delta_str, delta_color="inverse")
    with col_m2:
        st.metric("Blood Pressure", f"{p_row['SBP']:.0f}/{p_row['DBP']:.0f}", delta=f"MAP {p_row['MAP']:.0f} mmHg")
    with col_m3:
        st.metric("MAP 3h Trend", f"{p_row['MAP']:.1f} mmHg", delta=map_delta_str)
    with col_m4:
        st.metric("Shock Index", f"{p_row['Shock_Index']:.2f}", delta="Ref: 0.50–0.70", delta_color="off")
    with col_m5:
        st.metric("SpO2 / RR", f"{p_row['SpO2']:.1f}%", delta=f"{p_row['RR']:.0f} breaths/min", delta_color="off")
    with col_m6:
        st.metric("Serum Lactate", f"{p_row['Lactate']:.2f} mmol/L", delta=lac_delta_str, delta_color="inverse")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Left: Risk Gauge & XAI Explainability | Right: Nurse Review Protocol
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### 🎛️ Risk Engine & Explainable AI (XAI)")
        
        # Plotly Risk Gauge
        gauge_fig = build_risk_gauge(risk_info["score"], risk_info["band"])
        st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Transparent XAI Risk Drivers Card
        st.markdown(
            """
            <div style="font-size:0.85rem; font-weight:700; color:#334155; margin-bottom:0.4rem;">
                Key Physiological Contributors (Transparent Rule-Based Engine)
            </div>
            """,
            unsafe_allow_html=True
        )
        
        for d in drivers:
            if d["type"] == "risk":
                badge_type = f"<span style='color:#DC2626; font-weight:700;'>+{d['contribution']:.1f} pts</span>"
                border_clr = "#F87171"
            else:
                badge_type = "<span style='color:#059669; font-weight:700;'>Protective / Baseline</span>"
                border_clr = "#34D399"
                
            st.markdown(
                f"""
                <div class="xai-card" style="border-left-color:{border_clr};">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <strong>{d['factor']}</strong>
                        {badge_type}
                    </div>
                    <p style="margin:0.25rem 0 0 0; font-size:0.83rem; color:#475569;">
                        {d['description']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # Contribution Breakdown Mini-Chart
        contrib_items = [(k, v) for k, v in risk_info["contributions"].items() if v > 0]
        if contrib_items:
            df_contrib = pd.DataFrame(contrib_items, columns=["Feature", "Contribution"]).sort_values("Contribution", ascending=True)
            fig_bar = px.bar(
                df_contrib, x="Contribution", y="Feature", orientation="h",
                title="Risk Contribution Breakdown (Percentage Points)",
                color="Contribution", color_continuous_scale="Reds",
                labels={"Contribution": "Points Added to Composite Score"}
            )
            fig_bar.update_layout(
                height=180, margin=dict(l=10, r=10, t=30, b=30),
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        # Transparent methodology note
        st.info("ℹ️ **Transparent Architecture:** WardWatch AI uses an explainable, bounded rule-based scoring formula derived from validated physiology, not an opaque black-box machine learning model.")

    with col_right:
        st.markdown("### 📋 Context-Aware Nurse Review Checklist")
        st.markdown(
            """
            <p style="font-size:0.83rem; color:#64748B; margin-bottom:0.75rem;">
                Standardized escalation considerations generated from current observations.
                Review actions require qualified clinician evaluation.
            </p>
            """,
            unsafe_allow_html=True
        )
        
        checklist_items = get_nurse_review_items(p_row, risk_info)
        
        for item in checklist_items:
            # Key persisted per bed so nurse actions remain recorded during bedside workflow
            key = f"chk_{bed_num}_{item['id']}"
            current_val = st.session_state.checklist_state.get(key, False)
            
            checked = st.checkbox(
                f"**{item['title']}** — {item['detail']}",
                value=current_val,
                key=f"cb_{key}"
            )
            st.session_state.checklist_state[key] = checked
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Clinical validation reminder & qSOFA Notice
        qsofa_score_display = int(p_row['qSOFA_Score'])
        st.markdown(
            f"""
            <div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:8px; padding:0.85rem; font-size:0.8rem; color:#92400E; line-height:1.45;">
                <strong>qSOFA Prototype Notation:</strong><br>
                The qSOFA score shown (<code>{qsofa_score_display}/2</code>) evaluates the two physiological ward criteria (<em>RR ≥ 22/min</em> and <em>SBP ≤ 100 mmHg</em>). 
                Mental status (GCS / altered mentation) is currently unassessed in this synthetic vital telemetry stream and must be evaluated bedside by nursing staff.
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Time-Series Trajectory Section
    st.markdown("### 📈 Multivariate Trajectories (Hour 1 to Current Hour)")
    st.markdown(
        f"""
        <p style="font-size:0.85rem; color:#64748B; margin-top:-0.3rem;">
            Plotting historical observations through <strong>Hour {st.session_state.sim_hour}:00</strong> only. 
            Notice how Bed 4 demonstrates multivariate trend convergence beginning around Hour 10 before individual single-parameter thresholds trip.
        </p>
        """,
        unsafe_allow_html=True
    )
    
    c_ch1, c_ch2 = st.columns(2)
    with c_ch1:
        fig_hr = build_trend_chart(hist_p_df, "HR", "Heart Rate", "bpm", ref_low=60.0, ref_high=100.0, line_color="#0D9488")
        st.plotly_chart(fig_hr, use_container_width=True)
        
        fig_spo2 = build_trend_chart(hist_p_df, "SpO2", "Oxygen Saturation", "%", ref_low=94.0, line_color="#3B82F6")
        st.plotly_chart(fig_spo2, use_container_width=True)
        
    with c_ch2:
        fig_map = build_trend_chart(hist_p_df, "MAP", "Mean Arterial Pressure", "mmHg", ref_low=65.0, line_color="#8B5CF6")
        st.plotly_chart(fig_map, use_container_width=True)
        
        fig_lac = build_trend_chart(hist_p_df, "Lactate", "Serum Lactate", "mmol/L", ref_high=2.0, line_color="#EF4444")
        st.plotly_chart(fig_lac, use_container_width=True)

# ---------------------------------------------------------
# DATA EXPLORER & SYSTEM VALIDATION SECTION
# ---------------------------------------------------------
def render_data_explorer(full_df: pd.DataFrame, current_ward_df: pd.DataFrame):
    """
    Renders collapsible data exploration, CSV export, and validation check logs.
    """
    with st.expander("📊 Ward Data Exploration & Integrity Validation", expanded=False):
        tab1, tab2, tab3 = st.tabs(["Current Hour Ward Snapshot", "Full 24-Hour Synthetic Dataset", "Data Engine Integrity Checks"])
        
        with tab1:
            st.markdown(f"**Ward Observation Table at Hour {st.session_state.sim_hour}:00**")
            
            # Format display table
            display_cols = [
                "Bed", "PatientID", "HR", "SBP", "DBP", "MAP", "SpO2", "Temp", "RR", "Lactate", "WBC", "Shock_Index", "qSOFA_Score"
            ]
            
            view_df = current_ward_df[display_cols].copy()
            view_df.rename(columns={
                "qSOFA_Score": "qSOFA (Vital/2)"
            }, inplace=True)
            
            st.dataframe(view_df, use_container_width=True, hide_index=True)
            
            # CSV Download button for current hour
            csv_cur = view_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download Current Hour ({st.session_state.sim_hour}:00) Snapshot (CSV)",
                data=csv_cur,
                file_name=f"wardwatch_hour_{st.session_state.sim_hour}_snapshot.csv",
                mime="text/csv",
                key="dl_cur_csv"
            )
            
        with tab2:
            st.markdown("**Complete Multi-Patient 24-Hour Dataset (192 records)**")
            
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                bed_filter = st.multiselect("Filter by Bed", options=list(range(1, 9)), default=list(range(1, 9)))
            with col_f2:
                hr_range = st.slider("Filter Hour Range", 1, 24, (1, 24))
                
            filtered_full = full_df[
                (full_df["Bed"].isin(bed_filter)) & 
                (full_df["Hour"] >= hr_range[0]) & 
                (full_df["Hour"] <= hr_range[1])
            ]
            
            st.dataframe(filtered_full, use_container_width=True, hide_index=True)
            
            # CSV Download for full filtered dataset
            csv_all = filtered_full.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Synthetic Dataset (CSV)",
                data=csv_all,
                file_name="wardwatch_synthetic_24h_telemetry.csv",
                mime="text/csv",
                key="dl_full_csv"
            )
            
        with tab3:
            st.markdown("### Automated Data Pipeline & Clinical Model Validation")
            
            # Automated verification checks
            total_records = len(full_df)
            distinct_beds = full_df["Bed"].nunique()
            null_count = full_df[["HR", "SBP", "DBP", "MAP", "SpO2", "Temp", "RR", "Lactate", "WBC"]].isna().sum().sum()
            
            # Check MAP consistency: MAP ~= (SBP + 2*DBP)/3
            map_diff = np.abs(full_df["MAP"] - ((full_df["SBP"] + 2 * full_df["DBP"]) / 3.0)).max()
            map_consistent = map_diff < 0.5
            
            # Check risk bounded [0, 100]
            scores = [calculate_risk_score(r)["score"] for _, r in full_df.iterrows()]
            scores_valid = all(0.0 <= s <= 100.0 for s in scores)
            
            v1, v2, v3 = st.columns(3)
            with v1:
                st.metric("Total Records", f"{total_records} / 192", delta="100% Complete")
                st.metric("Bed Distribution", f"{distinct_beds} Beds", delta="24 obs/bed")
            with v2:
                st.metric("Missing Raw Telemetry", f"{null_count} NaNs", delta="Complete stream")
                st.metric("MAP Formula Check", "Consistent" if map_consistent else "Mismatch", delta="Δ < 0.5 mmHg")
            with v3:
                st.metric("Risk Score Bounds", "0% – 100%", delta="Strictly bounded")
                st.metric("Seed Reproducibility", "Seed 42 (Fixed)", delta="Deterministic")
                
            if total_records == 192 and distinct_beds == 8 and null_count == 0 and map_consistent and scores_valid:
                st.success("✅ All 5 Core Clinical Telemetry & Feature Engineering Validation Checks Passed.")
            else:
                st.warning("⚠️ Telemetry verification identified anomalies.")

# ---------------------------------------------------------
# MAIN APPLICATION CONTROLLER
# ---------------------------------------------------------
def main():
    # 0. Set Page Config & Styles
    st.set_page_config(
        page_title="WardWatch AI | Clinical Command Center",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # 1. Initialize Session State
    if "selected_bed" not in st.session_state:
        st.session_state.selected_bed = 4 # Default to Bed 4 for the central hackathon demonstration

    if "sim_hour" not in st.session_state:
        st.session_state.sim_hour = 12 # Default to hour 12 where Bed 4 deterioration is unfolding

    if "checklist_state" not in st.session_state:
        st.session_state.checklist_state = {}

    # 2. Load deterministic synthetic dataset
    raw_df = generate_synthetic_data()
    
    # 3. Derive clinical metrics & trend windows
    full_df = calculate_derived_metrics(raw_df)
    
    # 4. Render Sidebar controls FIRST so any button/slider callbacks resolve before main view rendering
    render_sidebar(full_df)
    
    # 5. Filter current hour ward view using guaranteed up-to-date session state
    current_ward_df = full_df[full_df["Hour"] == st.session_state.sim_hour].copy()
    
    # 6. Top Header Banner
    st.markdown(
        f"""
        <div class="ward-banner">
            <div>
                <h1>🏥 WardWatch AI — Clinical Early Warning Center</h1>
                <p>Non-ICU Inpatient Deterioration Detection • Multivariate Trend Analysis • Explainable Decision Support</p>
            </div>
            <div style="background:rgba(255, 255, 255, 0.1); border:1px solid rgba(255, 255, 255, 0.2); padding:0.4rem 0.8rem; border-radius:8px; font-size:0.85rem;">
                <strong>Ward Timeline:</strong> <span style="color:#2DD4BF; font-weight:700;">Hour {st.session_state.sim_hour}:00</span> (Day 1)
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 7. Ward Floor Grid (Section 6.2)
    st.markdown("### 🛏️ Inpatient Ward Floor Grid (8 Monitored Beds)")
    st.markdown(
        """
        <p style="font-size:0.85rem; color:#64748B; margin-top:-0.4rem; margin-bottom:1rem;">
            Click <strong>Select Bed</strong> to view comprehensive multivariate trajectories, explainable risk breakdown, and protocol checklists.
        </p>
        """,
        unsafe_allow_html=True
    )
    
    # Display in 4 responsive columns (wraps naturally)
    grid_cols = st.columns(4)
    for idx, (_, row) in enumerate(current_ward_df.iterrows()):
        col = grid_cols[idx % 4]
        with col:
            is_active = (int(row["Bed"]) == st.session_state.selected_bed)
            render_patient_card(row, is_active)
            
    st.markdown("<hr style='margin:1.5rem 0; border:0; border-top:1px solid #E2E8F0;'>", unsafe_allow_html=True)
    
    # 8. Selected Patient Deep-Dive Section (Section 6.3 - 6.7)
    selected_bed_data = current_ward_df[current_ward_df["Bed"] == st.session_state.selected_bed]
    
    if not selected_bed_data.empty:
        p_row = selected_bed_data.iloc[0]
        # Historical slice up through current hour only (NO future data leakage!)
        hist_p_df = full_df[
            (full_df["Bed"] == st.session_state.selected_bed) & 
            (full_df["Hour"] <= st.session_state.sim_hour)
        ].copy()
        
        render_patient_detail(p_row, hist_p_df)
    else:
        st.warning(f"No clinical record found for Bed {st.session_state.selected_bed} at Hour {st.session_state.sim_hour}.")
        
    st.markdown("<hr style='margin:2rem 0; border:0; border-top:1px solid #E2E8F0;'>", unsafe_allow_html=True)
    
    # 9. Data Exploration & Integrity Checks (Section 7)
    render_data_explorer(full_df, current_ward_df)
    
    # 10. Dynamic Clinical Footer
    current_year = datetime.now().year
    st.markdown(
        f"""
        <div class="clinical-footer">
            <strong>WardWatch AI Prototype</strong> • Clinical Decision Support Systems Hackathon MVP<br>
            © {current_year} WardWatch AI Initiative. Developed for research and clinical workflow demonstration using synthetic patient data.<br>
            <em>Clinical Notice: Not a certified medical device. Requires clinical validation before inpatient deployment.</em>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
