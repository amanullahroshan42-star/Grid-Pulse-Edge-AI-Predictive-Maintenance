"""
fault_injection_engine.py
Physics-grounded Digital Twin Fault Injection Engine for 3-Phase Induction Motor Asset.
Enables non-destructive laboratory testing of predictive maintenance algorithms
and emergency protection trip relays.

Standards Referenced:
- ISO 10816-3 / ISO 20816-1: Mechanical Vibration Severity
- IEEE 1415: Induction Motor Maintenance & Bearing Testing
- IEEE 1159 / EN 50160: Voltage Quality & Sag Dynamics
- IEC 60034-1 / NEMA MG-1: Motor Thermal Ratings & Overload Protection
"""

import copy
import math
import numpy as np

class MotorFaultInjector:
    """
    Transforms clean/healthy motor telemetry into synthetic fault signatures
    grounded in motor electromechanical physics.
    """

    def __init__(self, rated_current_a=4.8, sync_rpm=1500.0):
        self.rated_current_a = rated_current_a
        self.sync_rpm = sync_rpm

    def inject_mechanical_unbalance(self, packet, severity=1.0):
        """
        Fault 1: Rotor Mass Asymmetry / Mechanical Unbalance (ISO 10816-3).
        - Generates strong radial harmonic vibration at 1x fr running frequency (~24.5 Hz).
        - Increases radial acceleration RMS (ax) and introduces slight current pulsation.
        """
        p = copy.deepcopy(packet)
        # Scale severity: 1.0 = moderate unbalance (1.6g), 2.0 = severe unbalance (2.8g)
        intensity = max(0.2, min(3.0, severity))
        
        # Radial vibration surges
        base_ax = p.get("accel_x", 0.035)
        p["accel_x"] = round(float(base_ax + 1.25 * intensity + np.random.normal(0, 0.08)), 3)
        p["accel_y"] = round(float(p.get("accel_y", 0.04) + 0.45 * intensity), 3)
        p["accel_z"] = round(float(p.get("accel_z", 0.86) + 0.15 * intensity), 3)

        # Spectral 1x fr amplitude surges (unbalance signature)
        p["fft_1x_amp"] = round(float(0.05 + 0.95 * intensity), 4)
        p["fft_energy_band"] = round(float(0.002 + 0.45 * (intensity**2)), 4)
        
        # Torque pulsation modulates current slightly
        base_i = p.get("i_rms", 2.5)
        p["i_rms"] = round(float(base_i + 0.15 * intensity), 3)
        p["power_w"] = round(float(p["v_rms"] * p["i_rms"] * p["power_factor"]), 1)
        p["fault_injected"] = f"MECHANICAL_UNBALANCE (Severity {intensity:.1f})"
        return p

    def inject_bearing_fault(self, packet, severity=1.0):
        """
        Fault 2: Bearing Outer Race Defect / Fluting (IEEE 1415 / BPFO).
        - Generates high-frequency impulsive impact pulses.
        - Kurtosis surges (> 6.0 to 12.0), Crest Factor surges (> 4.0).
        """
        p = copy.deepcopy(packet)
        intensity = max(0.2, min(3.0, severity))

        # Peak impact acceleration spikes significantly while RMS rises moderately
        p["accel_x"] = round(float(p.get("accel_x", 0.035) + 0.65 * intensity), 3)
        p["vib_x_peak"] = round(float(0.12 + 2.40 * intensity + np.random.uniform(0.2, 0.8)), 3)
        
        # Impulsiveness indicators
        p["vib_x_kurtosis"] = round(float(3.0 + 4.5 * intensity + np.random.normal(0, 0.4)), 2)
        p["vib_x_crest"] = round(float(1.5 + 2.5 * intensity), 2)
        
        p["fault_injected"] = f"BEARING_DEFECT_BPFO (Kurtosis {p['vib_x_kurtosis']:.1f}, Crest {p['vib_x_crest']:.1f})"
        return p

    def inject_thermal_overload(self, packet, target_current_a=6.5):
        """
        Fault 3: Mechanical Binding / Stator Thermal Overload (IEC 60034-1).
        - Motor is mechanically overloaded or binding, pulling current well above rated (4.8A).
        - Rotor speed sags below nominal breakdown speed (~1410 RPM).
        """
        p = copy.deepcopy(packet)
        i_overload = max(self.rated_current_a * 1.1, target_current_a)
        
        p["i_rms"] = round(float(i_overload + np.random.normal(0, 0.04)), 3)
        
        # Speed drops under extreme torque demand
        # Nominal full load speed is 1430 RPM; severe overload drops speed to 1380-1410 RPM
        p["rpm"] = round(float(1405.0 - (i_overload - self.rated_current_a) * 20.0), 1)
        p["power_factor"] = round(min(0.92, max(0.82, 0.80 + (i_overload - 4.8) * 0.03)), 3)
        p["power_w"] = round(float(p["v_rms"] * p["i_rms"] * p["power_factor"]), 1)
        
        p["fault_injected"] = f"THERMAL_OVERLOAD (Current {p['i_rms']:.2f}A > Rated {self.rated_current_a}A)"
        return p

    def inject_voltage_sag(self, packet, target_voltage_v=170.0):
        """
        Fault 4: Severe Grid Voltage Sag / Phase Dip (IEEE 1159).
        - Supply voltage dips to 0.70 - 0.80 p.u. (e.g. 170V).
        - Stator current surges to maintain mechanical power; high RoCoF transient.
        """
        p = copy.deepcopy(packet)
        p["v_rms"] = round(float(target_voltage_v + np.random.normal(0, 0.5)), 1)
        p["freq_hz"] = round(float(p.get("freq_hz", 50.0) - 0.45), 2)
        
        # Constant power characteristic: Current increases inversely with voltage
        base_p = p.get("power_w", 350.0)
        p["i_rms"] = round(float((base_p / max(50.0, p["v_rms"] * p["power_factor"])) * 1.25), 3)
        p["fault_injected"] = f"GRID_VOLTAGE_SAG ({p['v_rms']:.1f}V < 0.9 p.u.)"
        return p

    def inject_single_phasing(self, packet):
        """
        Fault 5: Supply Single-Phasing (One supply fuse blown).
        - Extreme current surge on healthy phase (> 2x rated current).
        - Heavy 100 Hz electromagnetic torque oscillation and acoustic hum.
        """
        p = copy.deepcopy(packet)
        p["i_rms"] = round(float(self.rated_current_a * 1.85 + np.random.normal(0, 0.1)), 2)
        p["power_factor"] = 0.55
        p["rpm"] = 1350.0
        p["accel_x"] = round(float(p.get("accel_x", 0.035) + 1.85), 3)
        p["power_w"] = round(float(p["v_rms"] * p["i_rms"] * p["power_factor"]), 1)
        p["fault_injected"] = f"SINGLE_PHASING (Current {p['i_rms']:.1f}A, 100Hz Stator Ripple)"
        return p
