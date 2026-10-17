"""
Automated Validation Suite for WardWatch AI
Validates synthetic patient data, clinical metric calculations,
deterioration risk scoring, and clinical trajectory expectations.
"""

import sys
import numpy as np
import pandas as pd

# Import functions from app.py
from app import (
    generate_synthetic_data,
    calculate_derived_metrics,
    calculate_risk_score,
    get_risk_drivers,
    get_nurse_review_items,
    PATIENT_METADATA
)

def run_tests():
    print("=" * 60)
    print("RUNNING WARDWATCH AI CLINICAL ENGINE VALIDATION SUITE")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    def check(desc, condition):
        nonlocal passed, failed
        if condition:
            print(f" [PASS] {desc}")
            passed += 1
        else:
            print(f" [FAIL] {desc}")
            failed += 1

    # 1. Test Data Generation
    df_raw = generate_synthetic_data()
    check("Synthetic dataset contains exactly 192 records", len(df_raw) == 192)
    check("Dataset contains exactly 8 distinct beds", df_raw["Bed"].nunique() == 8)
    check("Dataset contains exactly 24 observations per bed", (df_raw.groupby("Bed")["Hour"].count() == 24).all())
    
    patient_ids = df_raw["PatientID"].unique()
    expected_pids = [f"WW-00{i}" for i in range(1, 9)]
    check("Patient IDs match WW-001 through WW-008", sorted(list(patient_ids)) == expected_pids)
    
    # 2. Check for missing or impossible values
    raw_vitals = ["HR", "SBP", "DBP", "MAP", "SpO2", "Temp", "RR", "Lactate", "WBC"]
    has_nulls = df_raw[raw_vitals].isna().any().any()
    check("Raw telemetry contains zero nulls or missing values", not has_nulls)
    
    # Check MAP formula: MAP ~= (SBP + 2 * DBP) / 3
    map_discrepancy = np.abs(df_raw["MAP"] - ((df_raw["SBP"] + 2.0 * df_raw["DBP"]) / 3.0)).max()
    check(f"MAP is consistent with (SBP + 2*DBP)/3 (max diff: {map_discrepancy:.2f})", map_discrepancy < 0.5)

    # 3. Test Derived Clinical Metrics
    df = calculate_derived_metrics(df_raw)
    
    # Shock Index
    check("Shock Index computed for all records", df["Shock_Index"].notna().all())
    check("Shock Index is strictly positive and finite", (df["Shock_Index"] > 0).all() and np.isfinite(df["Shock_Index"]).all())
    
    # qSOFA vital criteria
    check("qSOFA score ranges between 0 and 2", df["qSOFA_Score"].between(0, 2).all())
    
    # 3-Hour Deltas
    # For Hours 1-3, deltas should be NaN
    hr1_3_deltas = df[df["Hour"] <= 3][["Delta_HR_3h", "Delta_MAP_3h", "Delta_Lactate_3h"]]
    check("Hours 1-3 have NaN for 3-hour delta (safe handling)", hr1_3_deltas.isna().all().all())
    
    # For Hours 4-24, deltas should be numeric and finite
    hr4_24_deltas = df[df["Hour"] >= 4][["Delta_HR_3h", "Delta_MAP_3h", "Delta_Lactate_3h"]]
    check("Hours 4-24 have finite numeric 3-hour deltas", hr4_24_deltas.notna().all().all())

    # 4. Test Composite Risk Score Boundedness
    scores = []
    urgent_flags = []
    for _, row in df.iterrows():
        res = calculate_risk_score(row)
        scores.append(res["score"])
        urgent_flags.append(res["urgent_alert"])
        
    check("All risk scores are bounded strictly between 0 and 100%", all(0.0 <= s <= 100.0 for s in scores))
    
    # 5. Clinical Scenarios Validation
    # Beds 1-3: Stable
    b1_3 = df[df["Bed"].isin([1, 2, 3])]
    b1_3_scores = [calculate_risk_score(r)["score"] for _, r in b1_3.iterrows()]
    check("Beds 1-3 (Stable) average risk is low (<25%)", np.mean(b1_3_scores) < 25.0)
    check("Beds 1-3 never exceed yellow risk band (>65%)", max(b1_3_scores) < 65.0)

    # Bed 4: Progressive Deterioration starting around Hour 10
    b4 = df[df["Bed"] == 4].sort_values("Hour").reset_index(drop=True)
    b4_h1 = b4[b4["Hour"] == 1].iloc[0]
    b4_h9 = b4[b4["Hour"] == 9].iloc[0]
    b4_h15 = b4[b4["Hour"] == 15].iloc[0]
    b4_h24 = b4[b4["Hour"] == 24].iloc[0]
    
    score_b4_h1 = calculate_risk_score(b4_h1)["score"]
    score_b4_h9 = calculate_risk_score(b4_h9)["score"]
    score_b4_h15 = calculate_risk_score(b4_h15)["score"]
    score_b4_h24 = calculate_risk_score(b4_h24)["score"]
    
    check(f"Bed 4 is stable before Hr 10 (Hr 1: {score_b4_h1}%, Hr 9: {score_b4_h9}%)", score_b4_h1 < 25.0 and score_b4_h9 < 25.0)
    check(f"Bed 4 HR rises from ~75 to >105 bpm (Hr 1: {b4_h1['HR']} -> Hr 24: {b4_h24['HR']})", b4_h24["HR"] > 105.0)
    check(f"Bed 4 SBP falls from ~120 to <100 mmHg (Hr 1: {b4_h1['SBP']} -> Hr 24: {b4_h24['SBP']})", b4_h24["SBP"] < 100.0)
    check(f"Bed 4 Shock Index crosses 0.90 threshold (Hr 24 SI: {b4_h24['Shock_Index']})", b4_h24["Shock_Index"] > 0.90)
    check(f"Bed 4 Lactate rises toward >3.0 mmol/L (Hr 24 Lac: {b4_h24['Lactate']})", b4_h24["Lactate"] >= 3.0)
    check(f"Bed 4 risk score progressively escalates to Red (>65%) at Hr 24 ({score_b4_h24}%)", score_b4_h24 > 65.0)
    
    # Bed 5: High-Risk Respiratory Scenario
    b5 = df[df["Bed"] == 5]
    b5_high_rr_count = (b5["RR"] >= 22.0).sum()
    b5_qsofa_high_count = (b5["qSOFA_Score"] >= 1).sum()
    check(f"Bed 5 exhibits frequent tachypnea RR >= 22 ({b5_high_rr_count}/24 hours)", b5_high_rr_count >= 15)
    check(f"Bed 5 frequently triggers qSOFA vital criteria ({b5_qsofa_high_count}/24 hours)", b5_qsofa_high_count >= 15)
    
    # Beds 6-8: Post-operative recovery
    b6_8 = df[df["Bed"].isin([6, 7, 8])]
    b6_8_scores = [calculate_risk_score(r)["score"] for _, r in b6_8.iterrows()]
    check("Beds 6-8 (Post-op) stay generally in Green/Yellow band (<65%)", max(b6_8_scores) < 65.0)

    # 6. Test Explainable AI Drivers
    b4_crit_risk = calculate_risk_score(b4_h24)
    drivers = get_risk_drivers(b4_h24, b4_crit_risk)
    check("XAI identifies top 2 risk drivers for deteriorating patient", len(drivers) == 2)
    check("XAI driver contains contribution and description", "contribution" in drivers[0] and "description" in drivers[0])

    # 7. Test Nurse-Review Checklist
    chk_items = get_nurse_review_items(b4_h24, b4_crit_risk)
    check("Nurse-review checklist generated with actionable items", len(chk_items) >= 3)
    has_escalation = any("Escalation" in item["title"] for item in chk_items)
    check("Critical patient checklist includes escalation recommendation", has_escalation)
    
    print("=" * 60)
    print(f"VALIDATION SUMMARY: {passed} PASSED, {failed} FAILED")
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
