# WardWatch AI — Silent Deterioration & Early Warning Engine
### Prototype Clinical Decision Support System (CDSS) for Non-ICU Hospital Wards

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Online-brightgreen.svg?style=for-the-badge&logo=cloudflare)](https://marble-hall-qualifying-courses.trycloudflare.com)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Streamlit 1.35+](https://img.shields.io/badge/streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/plotly-interactive-3F4F75.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 🌐 **Interactive Live Public Deployment:** [https://marble-hall-qualifying-courses.trycloudflare.com](https://marble-hall-qualifying-courses.trycloudflare.com)
> 
> *Directly accessible on any desktop, tablet, or smartphone without installation.*

---

## 1. Executive Summary

**WardWatch AI** is a prototype clinical early-warning and decision-support engine engineered to detect subtle, multi-parameter patterns of physiological deterioration in general inpatient wards. In non-ICU hospital settings, episodic manual vital-sign recordings frequently miss insidious trajectories of occult sepsis, hypovolemia, and respiratory failure until acute decompensation occurs.

WardWatch AI continuously ingests synthetic vital signs and laboratory biomarkers across **8 simulated hospital beds** over a **24-hour timeline** (192 discrete hourly observations). Rather than waiting for isolated single-variable thresholds to breach critical limits, WardWatch AI synthesizes:
- **Multivariate Hemodynamic Trends** (Heart Rate, Systolic/Diastolic BP, Mean Arterial Pressure)
- **Shock Index** ($SI = \frac{HR}{SBP}$)
- **Quick Sequential Organ Failure Assessment (qSOFA)** vital criteria
- **Dynamic 3-Hour Rate-of-Change Deltas** ($\Delta HR_{3h}, \Delta MAP_{3h}, \Delta Lactate_{3h}$)
- **A Transparent, Bounded Composite Risk Score (0–100%)**
- **Explainable AI (XAI) Feature Attribution** with actionable nurse-review protocols

> **Clinical Research & Hackathon Notice:** WardWatch AI is an investigational prototype built with deterministic synthetic patient data for workflow evaluation and clinical UI/UX design. It is **not** an FDA/CE cleared medical device, does not provide autonomous diagnostic claims, and does not replace the judgment of licensed healthcare professionals.

---

## 2. Monitored Patients & Clinical Scenarios

WardWatch AI tracks 8 patient beds (`WW-001` through `WW-008`) reflecting real-world general ward conditions:

| Bed | Patient ID | Patient Name | Age / Sex | Clinical Scenario & Diagnosis | Trajectory Description |
|---|---|---|---|---|---|
| **Bed 1** | `WW-001` | Arthur Pendelton | 64 M | Post-cholecystectomy Day 2 | **Stable:** Normal physiological vitals, low composite risk. |
| **Bed 2** | `WW-002` | Elena Rostova | 52 F | Cellulitis on IV Cefazolin | **Stable:** Responsive to antibiotics, afebrile, stable hemodynamics. |
| **Bed 3** | `WW-003` | Marcus Vance | 71 M | Elective Total Hip Arthroplasty | **Stable:** Routine post-operative recovery, vitals within reference ranges. |
| **Bed 4** | `WW-004` | David Harrison | 68 M | Post-op Sigmoid Colectomy Day 3 | **Progressive Deterioration:** Stable baseline prior to Hr 10. Subtle anastomotic leak / occult sepsis initiates around Hr 10. HR rises (75 $\to$ 110+ bpm), SBP declines (120 $\to$ 95 mmHg), Shock Index climbs $> 0.90$, and serum lactate elevates $\to$ 3.2 mmol/L. |
| **Bed 5** | `WW-005` | Beatrice Campbell | 79 F | COPD Exacerbation with Atelectasis | **High-Risk Respiratory:** Persistent tachypnea (RR $\ge 22$/min), borderline hypotension, consistently triggering qSOFA vital criteria. |
| **Bed 6** | `WW-006` | Carlos Gomez | 45 M | Laparoscopic Appendectomy Day 1 | **Post-Op Recovery:** Mild early tachycardia settling down gradually. |
| **Bed 7** | `WW-007` | Sunita Patel | 61 F | Right Hemicolectomy Day 2 | **Post-Op Recovery:** Low-grade transient reactive pyrexia resolving with hydration. |
| **Bed 8** | `WW-008` | Gregory Smith | 58 M | Ventral Hernia Repair Day 1 | **Post-Op Recovery:** Stable surgical convalescence. |

---

## 3. Clinical Feature Engineering & Math

### 3.1 Shock Index (SI)
$$\text{Shock Index} = \frac{\text{Heart Rate (bpm)}}{\text{Systolic Blood Pressure (mmHg)}}$$
- **Reference Range:** $0.50 - 0.70$
- **Warning Threshold:** $> 0.90$ (signals impending hemodynamic collapse or left ventricular uncoupling)

### 3.2 qSOFA (Quick Sepsis-related Organ Failure Assessment)
WardWatch AI evaluates the two bedside vital criteria available in standard ward telemetry:
1. **Respiratory Rate $\ge 22$ breaths/min** (1 point)
2. **Systolic Blood Pressure $\le 100$ mmHg** (1 point)

*Note:* In accordance with clinical accuracy standards, mental status (GCS / altered mentation) is labeled as **unassessed** rather than fabricated, producing a transparent **two-component observed score out of 2**.

### 3.3 Dynamic 3-Hour Trend Deltas
To avoid late detection, the engine calculates sliding 3-hour differentials:
$$\Delta HR_{3h} = HR_t - HR_{t-3}$$
$$\Delta MAP_{3h} = MAP_t - MAP_{t-3}$$
$$\Delta Lactate_{3h} = \text{Lactate}_t - \text{Lactate}_{t-3}$$
*(For Hours 1–3, trends are safely designated as `N/A` without generating false alarms).*

### 3.4 Composite Deterioration Risk Score (0–100%)
A deterministic, bounded scoring algorithm that assigns normalized weights to physiological deviations:
- **Shock Index Elevation** (up to 22 pts)
- **Blood Pressure & MAP Decline** (up to 18 pts)
- **Respiratory Rate Abnormality** (up to 16 pts)
- **Oxygen Desaturation ($SpO_2$)** (up to 15 pts)
- **Serum Lactate Elevation** (up to 14 pts)
- **Dynamic 3-Hour Trend Divergence** (up to 8 pts)
- **Pyrexia / Hypothermia** (up to 7 pts)

**Risk Triage Categories:**
- 🟢 **Green (Low Risk):** $< 30\%$
- 🟡 **Yellow (Moderate Risk):** $30\% - 65\%$
- 🔴 **Red (High Risk):** $> 65\%$
- 🚨 **Urgent Physiological Override:** Triggered immediately if any critical threshold is breached (e.g., $SBP \le 90$, $SpO_2 < 90\%$, $\text{Shock Index} \ge 1.0$, $\text{Lactate} \ge 3.0$), preventing aggregate averaging from masking an acute abnormality.

---

## 4. Implementation of 20 Quality Fixes

1. **Remove horizontal scrolling:** Implemented clean CSS containers with responsive wrapping and fixed-height Plotly charts.
2. **Find broken links:** Verified zero broken external links; all assets and scripts are fully self-contained.
3. **Add a mobile-friendly menu:** Utilized Streamlit's native responsive sidebar and flexbox card grid.
4. **Add a favicon:** Configured `page_icon="🏥"` in `st.set_page_config()`.
5. **Fix page titles:** Set official title `WardWatch AI | Clinical Command Center`.
6. **Add meta descriptions where supported:** Documented product description in headers and README.
7. **Fix footer links:** Clean clinical copyright footer with dynamic year and safety notices.
8. **Create a custom 404 page where applicable:** Provided robust empty-state handling if bed selection is invalid.
9. **Fix copyright year:** Generated dynamically via `datetime.now().year`.
10. **Compress images:** Avoided heavy static raster assets; utilized lightweight vector/CSS components and Plotly WebGL/SVG charts.
11. **Fix broken buttons:** Configured unique session-state keys for all 8 bed cards, milestone shortcuts, and CSV exports.
12. **Add success messages:** Displays `st.success` upon successful data verification.
13. **Add error messages:** Uses `st.error` for urgent clinical overrides and `st.warning` for borderline conditions.
14. **Remove placeholder text:** Completely populated with realistic clinical histories, diagnoses, and lab values.
15. **Remove unused navigation:** Focused single-page command center with seamless deep-dive drilldown.
16. **Fix mobile overflow:** Adaptive 4-column layout that gracefully wraps on phones and tablets.
17. **Make the logo clickable:** Home milestone shortcut provided in the sidebar.
18. **Make the phone number clickable:** Omitted fabricated contact numbers per instructions.
19. **Make the email clickable:** Omitted fabricated contact emails per instructions.
20. **Make every page mobile-responsive:** Responsive Plotly layouts (`use_container_width=True`) and fluid metrics.

---

## 5. Getting Started & Installation

### Prerequisites
- Python 3.10 or 3.11 installed.

### Step 1: Clone / Navigate to Directory
```powershell
cd c:\Users\hp\HACK111
```

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run the Application
```powershell
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 6. Automated Verification & Testing

To execute the automated clinical engine validation suite:
```powershell
.venv\Scripts\python.exe test_app.py
```

### Validation Report Summary
- **192 Observations:** Exactly 8 beds $\times$ 24 hourly records.
- **Data Integrity:** 0 null values, deterministic seed 42.
- **Formula Accuracy:** Mean Arterial Pressure verifies $\frac{SBP + 2 \times DBP}{3}$ within 0.5 mmHg.
- **Risk Score Bounds:** 100% of generated scores fall within $[0.0, 100.0]\%$.
- **Bed 4 Trajectory:** Verifies occult deterioration begins at Hour 10, Shock Index exceeds 0.90, Lactate reaches $\ge 3.0$ mmol/L, and score escalates into Red ($>65\%$).
- **Bed 5 Scenario:** Verifies frequent tachypnea and qSOFA vital criteria triggers.
- **Explainability:** Generates top 2 transparent physiological drivers for every patient.

---

## 7. Hackathon Live Demonstration Script

1. **Hour 1 (Baseline):** Click `Baseline (Hr 1)` in sidebar. Note that all 8 beds show green badges with low composite risk ($<25\%$).
2. **Hour 10 (Deterioration Inception):** Click `Inception (Hr 10)`. Notice Bed 4's vitals begin subtle trend divergence: HR begins rising to 82 bpm, SBP trends to 116 mmHg, and 3-hour trend indicators detect the shift.
3. **Hour 15 (Midway Progression):** Click `Midway (Hr 15)`. Bed 4 escalates to Yellow ($48\%$). Shock Index reaches 0.88, Lactate climbs to 1.95 mmol/L. Click **Select Bed 4** to inspect the XAI Risk Breakdown.
4. **Hour 24 (Critical Decompensation):** Click `Critical (Hr 24)`. Bed 4 reaches Red ($>75\%$) with urgent alerts for tachycardia (111 bpm), hypotension (94/58 mmHg), Shock Index 1.18, and Lactate 3.2 mmol/L.
5. **Respiratory Comparison (Bed 5):** Click **Select Bed 5** to view how persistent tachypnea ($RR \ge 24$) and borderline BP trigger qSOFA vital criteria.
6. **Data Export:** Expand the **Ward Data Exploration** section at the bottom and download the snapshot CSV.

---

## 8. Clinical Limitations & Future Work

1. **Synthetic Telemetry:** Telemetry is generated via physiologically modeled mathematical curves. Clinical trial data (e.g., MIMIC-IV or eICU) is required for rigorous predictive validation.
2. **Mental Status Evaluation:** Standard qSOFA requires GCS evaluation; future versions can integrate nurse bedside charting for cognitive assessment.
3. **EHR Integration:** Future production roadmaps envision HL7 FHIR / SMART-on-FHIR interfaces with hospital electronic health records (Epic, Cerner).
