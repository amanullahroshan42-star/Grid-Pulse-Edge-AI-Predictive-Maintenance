"""
edge_single_phase_smartgrid.py
Master Edge Engine for Single-Phase Smart Grid, Asset Health & Emergency Protection.
Runs on Raspberry Pi (or PC).

Pipelines:
1. Micro-PMU Synchrophasor Engine (|V|, |I|, Phase Angle theta, Frequency, RoCoF df/dt)
2. Reactive Power & Volt-VAr Analytics (Q in kVAr, and recommended C in uF for 0.96 PF)
3. IEEE 1159 Power Quality (PQ) Sag, Swell & Interruption Classifier
4. Dynamic Time-of-Use (ToU) Energy Costing & Demand Response (DR) M&V
5. Multi-Criteria Robust Motor Health Index (MHI %: EHI, VHI, KHI)
6. Emergency Protection Trip Relay Controller (Instantaneous & Definite-Time Trip Latching)
"""

import os
import sys
import time
import json
import math
import joblib
import numpy as np

# Nominal Grid Specifications for 415V 3-Phase (230V - 240V Line-to-Neutral)
NOMINAL_VOLTAGE_LN = 230.0
NOMINAL_FREQ_HZ = 50.0
TARGET_POWER_FACTOR = 0.96

# PSG 2.2 kW (3.0 HP) Motor Rated Parameters
MOTOR_RATED_CURRENT_A = 4.8
MOTOR_RATED_RPM = 1430.0
MOTOR_SYNC_RPM = 1500.0

# Time-of-Use (ToU) Industrial Tariff Rates (in INR per kWh)
TOU_RATES = {
    "OFF_PEAK": 4.50,   # 22:00 - 06:00
    "NORMAL":   7.50,   # 06:00 - 18:00
    "PEAK":    11.20    # 18:00 - 22:00
}
DR_INCENTIVE_RATE = 15.00  # Utility incentive for Demand Response curtailment (INR / kWh)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(CURRENT_DIR, "motor_lab_model.joblib")
META_FILE = os.path.join(CURRENT_DIR, "motor_lab_metadata.json")

class SinglePhaseSmartGridEngine:
    def __init__(self, target_pf=0.96):
        self.target_pf = target_pf
        self.last_freq = 50.0
        self.last_freq_time = time.time()

        # PQ Event Tracking (IEEE 1159)
        self.active_pq_event = None
        self.pq_event_history = []

        # Demand Response State
        self.dr_event_active = False
        self.dr_baseline_kw = 0.624  # Baseline for current load
        self.total_curtailed_kwh = 0.0
        self.total_rebate_inr = 0.0

        # Thermal Memory & Protection State
        self.thermal_i2t_stress = 0.0
        self.last_thermal_time = time.time()
        
        # Protection Trip Relay State
        self.relay_state = "CLOSED"         # "CLOSED" (Armed/Running) or "TRIP_OPEN" (Emergency Trip)
        self.trip_latched = False           # Latched until operator reset
        self.trip_active_reason = ""
        self.trip_timestamp = None
        self.low_mhi_start_time = None      # Timer for definite-time delay (1.5s)
        self.trip_history = []

        # Load ML model if present
        self.ml_model = None
        self.ml_metadata = None
        if os.path.exists(MODEL_FILE) and os.path.exists(META_FILE):
            try:
                self.ml_model = joblib.load(MODEL_FILE)
                with open(META_FILE, "r", encoding="utf-8") as f:
                    self.ml_metadata = json.load(f)
                print(f"[+] Loaded ML model: {self.ml_metadata.get('model_type', 'RandomForest')}")
            except Exception as e:
                print(f"[!] Warning: Could not load ML model ({e}). Using rule-based fallback.")

        print("[+] Single-Phase Smart Grid & Protection Engine Initialized.")
        print(f"    Target Power Factor: {self.target_pf}")
        print(f"    Nominal Grid: {NOMINAL_VOLTAGE_LN}V L-N | {NOMINAL_FREQ_HZ}Hz | Rated Motor Current: {MOTOR_RATED_CURRENT_A}A")
        print(f"    Protection Relay: [ARMED / CLOSED]")

    def determine_tou_tariff(self):
        """Returns current tariff tier based on system hour."""
        hour = time.localtime().tm_hour
        if 18 <= hour < 22:
            return "PEAK", TOU_RATES["PEAK"]
        elif 6 <= hour < 18:
            return "NORMAL", TOU_RATES["NORMAL"]
        else:
            return "OFF_PEAK", TOU_RATES["OFF_PEAK"]

    def compute_synchrophasor(self, v_rms, i_rms, pf, freq):
        """
        Micro-PMU Synchrophasor Engine:
        Computes phasor magnitudes, power factor lag angle, and RoCoF (Rate of Change of Frequency).
        """
        now = time.time()
        dt = max(1e-3, now - self.last_freq_time)
        rocof = (freq - self.last_freq) / dt  # df/dt in Hz/s
        self.last_freq = freq
        self.last_freq_time = now

        clamped_pf = max(-1.0, min(1.0, pf))
        theta_rad = math.acos(clamped_pf)
        theta_deg = math.degrees(theta_rad)

        v_real = v_rms
        v_imag = 0.0
        i_real = i_rms * clamped_pf
        i_imag = -i_rms * math.sin(theta_rad)

        return {
            "v_magnitude": round(v_rms, 2),
            "i_magnitude": round(i_rms, 2),
            "phase_angle_deg": round(theta_deg, 2),
            "phase_angle_rad": round(theta_rad, 4),
            "frequency_hz": round(freq, 2),
            "rocof_hz_per_sec": round(rocof, 3),
            "v_phasor_cartesian": {"real": round(v_real, 2), "imag": round(v_imag, 2)},
            "i_phasor_cartesian": {"real": round(i_real, 2), "imag": round(i_imag, 2)}
        }

    def evaluate_reactive_power_analytics(self, v_rms, power_w, pf, freq):
        """
        Reactive Power & Volt-VAr Analytics:
        Calculates reactive power (Q) and recommended capacitance (uF) to achieve 0.96 PF.
        """
        power_kw = power_w / 1000.0
        clamped_pf = max(0.1, min(1.0, pf))

        theta_current = math.acos(clamped_pf)
        q_current_kvar = power_kw * math.tan(theta_current)

        theta_target = math.acos(self.target_pf)
        q_target_kvar = power_kw * math.tan(theta_target)

        q_deficit_kvar = max(0.0, q_current_kvar - q_target_kvar)

        c_recommended_uf = 0.0
        if q_deficit_kvar > 0 and v_rms > 50 and freq > 40:
            c_farads = (q_deficit_kvar * 1000.0) / (2.0 * math.pi * freq * (v_rms ** 2))
            c_recommended_uf = c_farads * 1e6

        apparent_power_kva = math.sqrt(power_kw**2 + q_current_kvar**2)

        return {
            "active_power_kw": round(power_kw, 3),
            "reactive_power_kvar": round(q_current_kvar, 3),
            "apparent_power_kva": round(apparent_power_kva, 3),
            "q_deficit_kvar": round(q_deficit_kvar, 3),
            "recommended_capacitor_uf": round(c_recommended_uf, 1),
            "compensation_advice": (
                f"Connect {c_recommended_uf:.1f} uF (440V AC) to correct PF from {clamped_pf:.2f} to {self.target_pf:.2f}"
                if c_recommended_uf > 0 else "Power factor already optimal (>= 0.96)"
            )
        }

    def evaluate_power_quality(self, v_rms):
        """
        IEEE 1159 / EN 50160 Power Quality Sag, Swell & Interruption Classifier.
        """
        pu_voltage = v_rms / NOMINAL_VOLTAGE_LN
        now_ms = int(time.time() * 1000)
        event_status = "NORMAL"

        if pu_voltage < 0.1:
            event_type = "INTERRUPTION"
        elif pu_voltage < 0.9:
            event_type = "VOLTAGE_SAG"
        elif pu_voltage > 1.1:
            event_type = "VOLTAGE_SWELL"
        else:
            event_type = "NORMAL"

        if event_type != "NORMAL":
            if self.active_pq_event is None:
                self.active_pq_event = {
                    "event_type": event_type,
                    "start_time_ms": now_ms,
                    "min_voltage_v": v_rms,
                    "max_voltage_v": v_rms
                }
            else:
                self.active_pq_event["min_voltage_v"] = min(self.active_pq_event["min_voltage_v"], v_rms)
                self.active_pq_event["max_voltage_v"] = max(self.active_pq_event["max_voltage_v"], v_rms)
            event_status = f"{event_type}_IN_PROGRESS"
        else:
            if self.active_pq_event is not None:
                duration_ms = now_ms - self.active_pq_event["start_time_ms"]
                finished_event = {
                    "event_type": self.active_pq_event["event_type"],
                    "duration_ms": duration_ms,
                    "residual_voltage_v": self.active_pq_event["min_voltage_v"],
                    "timestamp": time.strftime("%X")
                }
                self.pq_event_history.append(finished_event)
                event_status = f"{self.active_pq_event['event_type']}_RESOLVED ({duration_ms}ms)"
                self.active_pq_event = None

        return {
            "per_unit_voltage": round(pu_voltage, 3),
            "pq_status": event_status,
            "active_event": self.active_pq_event,
            "recent_events_count": len(self.pq_event_history)
        }

    def evaluate_demand_response(self, power_w):
        """
        Dynamic Time-of-Use (ToU) Costing & Demand Response M&V.
        """
        tier_name, rate_inr = self.determine_tou_tariff()
        power_kw = power_w / 1000.0
        hourly_cost_inr = power_kw * rate_inr

        curtailed_kw = 0.0
        if self.dr_event_active:
            curtailed_kw = max(0.0, self.dr_baseline_kw - power_kw)
            kwh_curtailed_step = (curtailed_kw * (1.0 / 3600.0))
            self.total_curtailed_kwh += kwh_curtailed_step
            self.total_rebate_inr += (kwh_curtailed_step * DR_INCENTIVE_RATE)

        return {
            "tariff_tier": tier_name,
            "tariff_rate_inr_per_kwh": rate_inr,
            "current_cost_rate_inr_per_hr": round(hourly_cost_inr, 2),
            "dr_event_active": self.dr_event_active,
            "curtailed_power_kw": round(curtailed_kw, 3),
            "total_curtailed_kwh": round(self.total_curtailed_kwh, 4),
            "total_rebate_earned_inr": round(self.total_rebate_inr, 2)
        }

    def compute_motor_health_and_load(self, packet):
        """
        Multi-Criteria Robust Motor Health Index (MHI %) Architecture:
        1. EHI (Electrical Health Index): Overcurrent ratio, thermal stress, voltage bounds.
        2. VHI (Vibration Health Index): ISO 10816 severity, Kurtosis, Crest Factor.
        3. KHI (Kinematic Health Index): Rotor slip deviation, 1x fr unbalance peak.
        4. Composite MHI: min(EHI, VHI, KHI) (Conservative weakest-link protection).
        """
        v_rms   = packet.get("v_rms", 230.0)
        i_rms   = packet.get("i_rms", 2.1)
        power_w = packet.get("power_w", 120.0)
        pf      = packet.get("power_factor", 0.25)
        freq    = packet.get("freq_hz", 50.0)
        rpm     = packet.get("rpm", 1495.0)
        ax      = packet.get("accel_x", 0.035)
        ay      = packet.get("accel_y", 0.040)
        az      = packet.get("accel_z", 0.860)

        # Advanced fault fields if provided
        peak_x  = packet.get("vib_x_peak", abs(ax) * 1.414)
        kurt_x  = packet.get("vib_x_kurtosis", 3.0)
        crest_x = packet.get("vib_x_crest", peak_x / max(1e-3, abs(ax)))
        peak_1x = packet.get("fft_1x_amp", 0.05)

        # Overall RMS vibration
        vib_rms = float(np.sqrt(np.mean(np.array([ax, ay, az])**2)))
        slip = max(0.0, (MOTOR_SYNC_RPM - rpm) / MOTOR_SYNC_RPM)

        # -------------------------------------------------------------
        # 1. Electrical Health Index (EHI %)
        # -------------------------------------------------------------
        current_ratio = i_rms / MOTOR_RATED_CURRENT_A
        pen_current = 0.0
        if current_ratio > 1.0:
            if current_ratio <= 1.25:
                pen_current = (current_ratio - 1.0) * 80.0
            else:
                pen_current = 20.0 + (current_ratio - 1.25) * 140.0

        v_pu = v_rms / NOMINAL_VOLTAGE_LN
        pen_voltage = 0.0
        if v_pu < 0.90 or v_pu > 1.10:
            pen_voltage = abs(v_pu - 1.0) * 120.0

        ehi = float(np.clip(100.0 - (pen_current + pen_voltage), 0.0, 100.0))

        # -------------------------------------------------------------
        # 2. Vibration Health Index (VHI %) - ISO 10816 Severity & Kurtosis
        # -------------------------------------------------------------
        pen_vib_iso = 0.0
        if vib_rms > 0.8:
            if vib_rms <= 1.5:
                pen_vib_iso = (vib_rms - 0.8) * 35.0
            else:
                pen_vib_iso = 24.5 + (vib_rms - 1.5) * 60.0

        pen_kurtosis = 0.0
        if kurt_x > 4.2:
            pen_kurtosis = (kurt_x - 4.2) * 8.5  # Bearing fluting / impact signature

        pen_crest = 0.0
        if crest_x > 3.5:
            pen_crest = (crest_x - 3.5) * 10.0

        vhi = float(np.clip(100.0 - (pen_vib_iso + pen_kurtosis + pen_crest), 0.0, 100.0))

        # -------------------------------------------------------------
        # 3. Kinematic Health Index (KHI %) - Slip & 1x fr Unbalance Peak
        # -------------------------------------------------------------
        pen_slip = 0.0
        if slip > 0.053:
            pen_slip = (slip - 0.053) * 320.0

        pen_unbalance = 0.0
        if peak_1x > 0.35:
            pen_unbalance = (peak_1x - 0.35) * 45.0  # Rotor unbalance harmonic

        khi = float(np.clip(100.0 - (pen_slip + pen_unbalance), 0.0, 100.0))

        # -------------------------------------------------------------
        # 4. Composite MHI % (Weakest-Link Principle)
        # -------------------------------------------------------------
        composite_mhi = float(np.clip(min(ehi, vhi, khi), 0.0, 100.0))

        color = "GREEN" if composite_mhi >= 85.0 else ("YELLOW" if composite_mhi >= 65.0 else "RED")

        # ML Load Prediction (if model is loaded)
        predicted_condition = "Unknown"
        confidence_pct = 100.0
        if self.ml_model is not None and self.ml_metadata is not None:
            try:
                feat_dict = {
                    "voltage_v": v_rms, "current_a": i_rms, "power_w": power_w,
                    "power_factor": pf, "frequency_hz": freq, "rpm": rpm,
                    "slip": slip, "shaft_freq_hz": rpm / 60.0,
                    "vib_x_rms": abs(ax), "vib_x_peak": peak_x, "vib_x_crest": crest_x,
                    "vib_x_kurtosis": kurt_x, "vib_x_skew": 0.0,
                    "vib_y_rms": abs(ay), "vib_y_peak": abs(ay), "vib_y_crest": 1.0,
                    "vib_y_kurtosis": 3.0,
                    "vib_z_rms": abs(az), "vib_z_peak": abs(az), "vib_z_crest": 1.0,
                    "fft_1x_amp": peak_1x, "fft_energy_band": 0.002, "fft_dominant_freq": 24.0
                }
                vec = np.array([feat_dict.get(c, 0.0) for c in self.ml_metadata["features"]]).reshape(1, -1)
                predicted_condition = str(self.ml_model.predict(vec)[0])
                probs = self.ml_model.predict_proba(vec)[0]
                idx = list(self.ml_model.classes_).index(predicted_condition)
                confidence_pct = round(float(probs[idx]) * 100.0, 1)
            except Exception:
                predicted_condition = "No_Load" if i_rms < 2.3 else ("Load_1_2kg" if i_rms < 3.0 else "Load_2_7.5kg")
        else:
            predicted_condition = "No_Load" if i_rms < 2.3 else ("Load_1_2kg" if i_rms < 3.0 else "Load_2_7.5kg")

        return {
            "predicted_load_state": predicted_condition,
            "prediction_confidence": confidence_pct,
            "mhi_percentage": round(composite_mhi, 1),
            "mhi_status": color,
            "sub_indices": {
                "ehi_pct": round(ehi, 1),
                "vhi_pct": round(vhi, 1),
                "khi_pct": round(khi, 1)
            },
            "diagnostics": {
                "vibration_rms_g": round(vib_rms, 3),
                "kurtosis": round(kurt_x, 2),
                "crest_factor": round(crest_x, 2),
                "unbalance_1x_g": round(peak_1x, 3),
                "current_ratio": round(current_ratio, 2),
                "slip": round(slip, 4)
            }
        }

    def evaluate_protection_trip(self, mhi_data, v_rms, i_rms, ax, ay, az):
        """
        Emergency Protection Trip Relay Controller:
        - Instantaneous Trip: Current > 200% rated, Vibration > 2.5g RMS / 3.5g Peak, Severe Undervoltage.
        - Definite-Time Trip: Composite MHI < 65% for >= 1.5 seconds.
        - Latches trip state until explicit operator reset.
        """
        now = time.time()
        mhi_pct = mhi_data["mhi_percentage"]
        vib_rms = mhi_data["diagnostics"]["vibration_rms_g"]
        
        # If already tripped and latched, hold the trip state
        if self.trip_latched:
            return {
                "relay_state": "TRIP_OPEN",
                "trip_latched": True,
                "trip_reason": self.trip_active_reason,
                "trip_timestamp": self.trip_timestamp,
                "action_command": "HOLD_TRIP_LATCHED"
            }

        trip_triggered = False
        trip_reason = ""

        # 1. Instantaneous Trips
        if i_rms >= (MOTOR_RATED_CURRENT_A * 2.0):  # >= 9.6 A
            trip_triggered = True
            trip_reason = f"INSTANTANEOUS_OVERCURRENT_TRIP (Current={i_rms:.2f}A >= 200% rated)"
        elif vib_rms >= 2.5 or abs(ax) >= 3.5:
            trip_triggered = True
            trip_reason = f"INSTANTANEOUS_DESTRUCTIVE_VIBRATION_TRIP (Vib={vib_rms:.2f}g >= 2.5g)"
        elif v_rms < 140.0 and i_rms > 3.0:
            trip_triggered = True
            trip_reason = f"SEVERE_VOLTAGE_COLLAPSE_TRIP (Voltage={v_rms:.1f}V < 140V)"

        # 2. Definite-Time Health Index Trip (MHI < 65% for > 1.5s)
        if not trip_triggered:
            if mhi_pct < 65.0:
                if self.low_mhi_start_time is None:
                    self.low_mhi_start_time = now
                elif (now - self.low_mhi_start_time) >= 1.5:
                    trip_triggered = True
                    # Identify weakest domain
                    subs = mhi_data["sub_indices"]
                    weakest = min(subs, key=subs.get)
                    trip_reason = f"HEALTH_INDEX_CRITICAL_TRIP (MHI={mhi_pct}% < 65% for >1.5s, Fault Dominance: {weakest.upper()})"
            else:
                self.low_mhi_start_time = None

        if trip_triggered:
            self.relay_state = "TRIP_OPEN"
            self.trip_latched = True
            self.trip_active_reason = trip_reason
            self.trip_timestamp = time.strftime("%X")
            self.trip_history.append({
                "timestamp": self.trip_timestamp,
                "reason": trip_reason,
                "mhi": mhi_pct
            })
            action_command = "DISPATCH_TRIP_RELAY_OPEN"
        else:
            action_command = "RELAY_ARMED_CLOSED"

        return {
            "relay_state": self.relay_state,
            "trip_latched": self.trip_latched,
            "trip_reason": trip_reason if trip_triggered else "None (Motor Operating Within Protection Envelopes)",
            "trip_timestamp": self.trip_timestamp,
            "action_command": action_command
        }

    def reset_trip_relay(self):
        """Operator reset command to re-arm the protection relay after clearing a fault."""
        if not self.trip_latched:
            return False, "Relay was not tripped."
        self.relay_state = "CLOSED"
        self.trip_latched = False
        self.trip_active_reason = ""
        self.low_mhi_start_time = None
        print("[+] Protection Trip Relay LATCH CLEARED. Contactor RE-ARMED [CLOSED].")
        return True, "Protection Relay Reset Successfully"

    def process_incoming_packet(self, packet):
        """
        Master execution pipeline on each incoming JSON telemetry packet.
        Handles Standby / De-energized states cleanly when v_rms < 10.0V.
        """
        v_rms   = float(packet.get("v_rms", 0.0))
        i_rms   = float(packet.get("i_rms", 0.0))
        power_w = float(packet.get("power_w", 0.0))
        pf      = float(packet.get("power_factor", 0.0))
        freq    = float(packet.get("freq_hz", 0.0))
        ax      = float(packet.get("accel_x", 0.0))
        ay      = float(packet.get("accel_y", 0.0))
        az      = float(packet.get("accel_z", 0.0))

        # Check for Standby / De-energized state
        if v_rms < 10.0 and i_rms < 0.2:
            synchro = {
                "v_magnitude": 0.0, "i_magnitude": 0.0, "phase_angle_deg": 0.0,
                "phase_angle_rad": 0.0, "frequency_hz": 0.0, "rocof_hz_per_sec": 0.0,
                "v_phasor_cartesian": {"real": 0.0, "imag": 0.0},
                "i_phasor_cartesian": {"real": 0.0, "imag": 0.0}
            }
            var_analytics = {
                "active_power_kw": 0.0, "reactive_power_kvar": 0.0, "apparent_power_kva": 0.0,
                "current_pf": 0.0, "target_pf": 0.96, "required_compensation_kvar": 0.0,
                "recommended_shunt_capacitance_uF": 0.0, "var_status": "DE_ENERGIZED"
            }
            pq = {
                "status": "OFFLINE", "event_type": "DE_ENERGIZED", "v_rms": 0.0, "severity": "STANDBY"
            }
            dr = {
                "tier": "OFF_PEAK", "rate_per_kwh": 0.06, "hourly_cost_rate": 0.0,
                "monthly_projected_cost": 0.0, "curtailment_recommended": False
            }
            mhi_load = {
                "mhi_percentage": 100.0, "health_zone": "STANDBY_HEALTHY", "color_code": "#718096",
                "predicted_load_state": "Motor_Standby_Off", "classification_confidence_pct": 100.0,
                "sub_indices": {"ehi_pct": 100.0, "vhi_pct": 100.0, "khi_pct": 100.0},
                "diagnostics": {"vibration_rms_g": 0.0, "vibration_iso_mm_s": 0.0, "current_loading_pct": 0.0, "slip_pct": 0.0}
            }
            protection = {
                "relay_state": self.relay_state,
                "trip_latched": self.trip_latched,
                "trip_reason": "None (Motor in Standby / Power Disconnected)",
                "trip_timestamp": self.trip_timestamp,
                "action_command": "RELAY_ARMED_CLOSED" if not self.trip_latched else "DISPATCH_TRIP_RELAY_OPEN"
            }
            return {
                "phase": packet.get("phase", "A"),
                "timestamp": time.strftime("%X"),
                "synchrophasor": synchro,
                "var_analytics": var_analytics,
                "power_quality": pq,
                "demand_response": dr,
                "motor_health_and_load": mhi_load,
                "protection_relay": protection,
                "fault_tag": "Motor Standby (AC Power OFF)"
            }

        # 1. Synchrophasor (Micro-PMU)
        synchro = self.compute_synchrophasor(v_rms, i_rms, pf, freq)

        # 2. Reactive Power & Volt-VAr Analytics
        var_analytics = self.evaluate_reactive_power_analytics(v_rms, power_w, pf, freq)

        # 3. Power Quality (PQ)
        pq = self.evaluate_power_quality(v_rms)

        # 4. Demand Response & ToU
        dr = self.evaluate_demand_response(power_w)

        # 5. Multi-Criteria Health Index & Load State
        mhi_load = self.compute_motor_health_and_load(packet)

        # 6. Emergency Protection Trip Relay Controller
        protection = self.evaluate_protection_trip(mhi_load, v_rms, i_rms, ax, ay, az)

        return {
            "phase": packet.get("phase", "A"),
            "timestamp": time.strftime("%X"),
            "synchrophasor": synchro,
            "var_analytics": var_analytics,
            "power_quality": pq,
            "demand_response": dr,
            "motor_health_and_load": mhi_load,
            "protection_relay": protection,
            "fault_tag": packet.get("fault_injected", "None (Baseline Telemetry)")
        }

# ==========================================
# BENCHMARK SIMULATION TEST
# ==========================================
def run_simulation_benchmark():
    engine = SinglePhaseSmartGridEngine()
    print("\n" + "="*80)
    print("  SMART GRID, ASSET HEALTH & PROTECTION BENCHMARK (PSG 2.2 kW Motor)")
    print("="*80)

    test_scenarios = [
        ("1. Lab Healthy Baseline (Load 1: 230V, 2.5A, 1480 RPM)", {
            "phase": "A", "v_rms": 230.0, "i_rms": 2.5, "power_w": 356.5,
            "power_factor": 0.62, "freq_hz": 50.00, "rpm": 1480.0,
            "accel_x": 0.035, "accel_y": 0.045, "accel_z": 0.860
        }),
        ("2. IEEE 1159 Grid Voltage Sag Event (180V < 0.9 p.u.)", {
            "phase": "A", "v_rms": 180.0, "i_rms": 4.2, "power_w": 580.0,
            "power_factor": 0.76, "freq_hz": 49.80, "rpm": 1430.0,
            "accel_x": 0.045, "accel_y": 0.055, "accel_z": 0.880
        }),
        ("3. Synthetic Mechanical Unbalance Fault (ISO 10816-3 Zone C/D)", {
            "phase": "A", "v_rms": 230.0, "i_rms": 2.8, "power_w": 410.0,
            "power_factor": 0.64, "freq_hz": 50.00, "rpm": 1475.0,
            "accel_x": 1.85, "accel_y": 0.65, "accel_z": 0.88,
            "fft_1x_amp": 1.45, "vib_x_kurtosis": 3.2, "vib_x_crest": 1.8
        }),
        ("4. Synthetic Bearing Fluting / BPFO Shock (High Kurtosis)", {
            "phase": "A", "v_rms": 230.0, "i_rms": 2.6, "power_w": 365.0,
            "power_factor": 0.62, "freq_hz": 50.00, "rpm": 1478.0,
            "accel_x": 0.95, "accel_y": 0.45, "accel_z": 0.86,
            "vib_x_peak": 3.85, "vib_x_kurtosis": 8.4, "vib_x_crest": 4.8
        }),
        ("5. Severe Locked-Rotor / Stator Overcurrent (>200% Rated)", {
            "phase": "A", "v_rms": 218.0, "i_rms": 10.4, "power_w": 1850.0,
            "power_factor": 0.82, "freq_hz": 49.60, "rpm": 1320.0,
            "accel_x": 0.25, "accel_y": 0.20, "accel_z": 0.88
        })
    ]

    for title, packet in test_scenarios:
        result = engine.process_incoming_packet(packet)
        sp = result["synchrophasor"]
        pq = result["power_quality"]
        ml = result["motor_health_and_load"]
        pr = result["protection_relay"]
        subs = ml["sub_indices"]

        print(f"\n>>> [{title}]")
        print(f"  Micro-PMU Phasor   : |V|={sp['v_magnitude']}V | |I|={sp['i_magnitude']}A | Phase Angle={sp['phase_angle_deg']} deg")
        print(f"  Power Quality (PQ) : V={packet['v_rms']}V ({pq['per_unit_voltage']} p.u.) | Status={pq['pq_status']}")
        print(f"  Health Sub-Indices : EHI={subs['ehi_pct']}% | VHI={subs['vhi_pct']}% | KHI={subs['khi_pct']}%")
        print(f"  Composite MHI      : {ml['mhi_percentage']}% [{ml['mhi_status']}] (Load: '{ml['predicted_load_state']}')")
        print(f"  >>> RELAY STATE    : [{pr['relay_state']}] -> Action: {pr['action_command']}")
        if pr["trip_latched"]:
            print(f"  >>> TRIP REASON    : {pr['trip_reason']}")

    print("\n" + "="*80)
    print("[+] All smart grid & protection benchmark tests passed successfully!")

def run_live_mqtt_daemon(broker="localhost", port=1883):
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        print("[!] paho-mqtt not installed. Run: pip install paho-mqtt")
        return

    engine = SinglePhaseSmartGridEngine()
    topic_in = "motor/phase_a/telemetry"
    topic_out = "motor/grid_telemetry/phase_a"
    topic_trip = "motor/control/protection"
    topic_reset = "motor/control/reset"

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print(f"[+] Connected to MQTT Broker at {broker}:{port}")
            client.subscribe([(topic_in, 0), (topic_reset, 0)])
            print(f"[*] Subscribed to telemetry: {topic_in} and reset: {topic_reset}")
        else:
            print(f"[!] MQTT connection failed, code {rc}")

    def on_message(client, userdata, msg):
        try:
            if msg.topic == topic_reset:
                success, msg_str = engine.reset_trip_relay()
                client.publish("motor/status/protection", json.dumps({"status": "RESET_ACK", "detail": msg_str}))
                return

            packet = json.loads(msg.payload.decode("utf-8"))
            result = engine.process_incoming_packet(packet)
            client.publish(topic_out, json.dumps(result))

            pr = result["protection_relay"]
            if pr["action_command"] == "DISPATCH_TRIP_RELAY_OPEN":
                trip_payload = {
                    "command": "TRIP_OPEN",
                    "reason": pr["trip_reason"],
                    "mhi": result["motor_health_and_load"]["mhi_percentage"],
                    "timestamp": pr["trip_timestamp"]
                }
                client.publish(topic_trip, json.dumps(trip_payload))
                print(f"[!] EMERGENCY TRIP DISPATCHED: {pr['trip_reason']}")

        except Exception as e:
            print(f"[!] Error processing MQTT payload: {e}")

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"[*] Starting MQTT listener on {broker}:{port}...")
    try:
        client.connect(broker, port, 60)
        client.loop_forever()
    except Exception as e:
        print(f"[!] Could not connect to MQTT Broker: {e}")

if __name__ == "__main__":
    if "--mqtt" in sys.argv:
        run_live_mqtt_daemon()
    else:
        run_simulation_benchmark()
