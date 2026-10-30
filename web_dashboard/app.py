"""
web_dashboard/app.py
Production-grade Real-Time Web Dashboard for:
PSG 2.2 kW Three-Phase Induction Motor Testbed & Single-Phase Smart Grid Gateway.

Features:
- Dual-Mode Auto-Switching Architecture:
    * LIVE_HARDWARE: When ESP32 / PZEM is actively streaming (via MQTT, Serial, or REST)
    * DIGITAL_TWIN: Seamless automated fallback when hardware is disconnected/unpowered
- Live 1-second streaming telemetry with SVG sparklines & gauges
- Real-time Chart.js timeseries (Voltage, Current, Vibration, Power)
- Multi-Criteria Health Index (EHI, VHI, KHI, Composite MHI %)
- Micro-PMU Synchrophasor Polar Vector Display
- Interactive Digital Twin Fault Injector (Unbalance, Bearing, Overload, Sag)
- Emergency Protection Trip Relay Controller & Latching Reset
- Time-of-Use (ToU) Costing & Demand Response (DR) M&V
"""

import os
import sys
import time
import json
import threading
from flask import Flask, render_template, jsonify, request

# Add parent directory to path to import core engines
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from edge_single_phase_smartgrid import SinglePhaseSmartGridEngine
from fault_injection_engine import MotorFaultInjector
from edge_filter_population import EdgePopulationFilter

app = Flask(__name__, template_folder="templates")

# Core engine singletons
smartgrid_engine = SinglePhaseSmartGridEngine()
fault_injector = MotorFaultInjector(rated_current_a=4.8)
edge_filter = EdgePopulationFilter()

# Dashboard Global State
active_fault_mode = "none"  # "none", "unbalance", "bearing", "overload", "sag", "single_phase"
active_fault_intensity = 1.0
current_load_condition = "Load_1_2kg"  # "No_Load", "Load_1_2kg", "Load_2_7.5kg"

# Hardware Watchdog & Source Tracking
last_hardware_time = 0.0
HARDWARE_TIMEOUT_SEC = 3.0  # Fall back to Digital Twin if no hardware packet for 3s
current_data_source = "DIGITAL_TWIN"  # "LIVE_HARDWARE" or "DIGITAL_TWIN"

history_buffer = {
    "timestamps": [],
    "voltage": [],
    "current": [],
    "power": [],
    "vibration_x": [],
    "mhi": []
}
MAX_HISTORY = 40

# Lock for thread-safe state updates
data_lock = threading.Lock()
latest_telemetry = {}

def get_base_packet():
    """Generates baseline lab telemetry corresponding to current selected load condition."""
    global current_load_condition
    if current_load_condition == "No_Load":
        base = {
            "phase": "A", "v_rms": 230.1, "i_rms": 2.10, "power_w": 120.7,
            "power_factor": 0.24, "freq_hz": 50.00, "rpm": 1495.0,
            "accel_x": 0.032, "accel_y": 0.043, "accel_z": 0.859
        }
    elif current_load_condition == "Load_2_7.5kg":
        base = {
            "phase": "A", "v_rms": 220.2, "i_rms": 3.50, "power_w": 623.7,
            "power_factor": 0.81, "freq_hz": 49.98, "rpm": 1460.0,
            "accel_x": 0.036, "accel_y": 0.048, "accel_z": 0.860
        }
    else:  # Load 1 (2 kg) default
        base = {
            "phase": "A", "v_rms": 230.0, "i_rms": 2.50, "power_w": 356.5,
            "power_factor": 0.62, "freq_hz": 50.00, "rpm": 1480.0,
            "accel_x": 0.035, "accel_y": 0.045, "accel_z": 0.860
        }
    return base

def commit_telemetry_packet(raw_packet, source_label):
    """Common ingestion pipeline for both live hardware and digital twin packets."""
    global latest_telemetry, current_data_source, active_fault_mode, active_fault_intensity

    # INJECT FAULT INTO LIVE TELEMETRY (Software-in-the-Loop on Live Stream)
    if active_fault_mode != "none":
        # Make a copy to avoid mutating original
        raw_packet = dict(raw_packet)
        if active_fault_mode == "unbalance":
            raw_packet = fault_injector.inject_mechanical_unbalance(raw_packet, severity=active_fault_intensity)
        elif active_fault_mode == "bearing":
            raw_packet = fault_injector.inject_bearing_fault(raw_packet, severity=active_fault_intensity)
        elif active_fault_mode == "overload":
            raw_packet = fault_injector.inject_thermal_overload(raw_packet, target_current_a=6.8)
        elif active_fault_mode == "sag":
            raw_packet = fault_injector.inject_voltage_sag(raw_packet, target_voltage_v=172.0)
        elif active_fault_mode == "single_phase":
            raw_packet = fault_injector.inject_single_phasing(raw_packet)
        
        # Tag the source to show hybrid hardware + fault injection
        if source_label == "LIVE_HARDWARE":
            source_label = "LIVE_HARDWARE_FAULT_INJECTED"

    # 1. Apply population-based edge filter
    filtered_packet = edge_filter.filter_telemetry_packet(raw_packet)

    # 2. Process through Smart Grid Engine
    result = smartgrid_engine.process_incoming_packet(filtered_packet)
    result["data_source"] = source_label

    # Check if automated protection trip occurred in engine (latched or trip_triggered)
    prot = result.get("protection_relay", {})
    if prot.get("trip_latched", False) or prot.get("relay_state") == "TRIP_OPEN":
        send_actuator_command("TRIP")

    # 3. Commit to global state & rolling history
    with data_lock:
        current_data_source = source_label
        latest_telemetry = result
        t_str = time.strftime("%H:%M:%S")
        history_buffer["timestamps"].append(t_str)
        history_buffer["voltage"].append(result["synchrophasor"]["v_magnitude"])
        history_buffer["current"].append(result["synchrophasor"]["i_magnitude"])
        history_buffer["power"].append(result["var_analytics"]["active_power_kw"] * 1000.0)
        history_buffer["vibration_x"].append(result["motor_health_and_load"]["diagnostics"]["vibration_rms_g"])
        history_buffer["mhi"].append(result["motor_health_and_load"]["mhi_percentage"])

        if len(history_buffer["timestamps"]) > MAX_HISTORY:
            for k in history_buffer:
                history_buffer[k] = history_buffer[k][-MAX_HISTORY:]

def telemetry_stream_worker():
    """
    Background worker thread:
    - Only runs Digital Twin if a fault injection is manually triggered by user, OR if user selects Digital Twin mode.
    - When hardware is disconnected and no fault is active, it reports 0 values (no artificial fake numbers).
    """
    global active_fault_mode, active_fault_intensity, last_hardware_time

    while True:
        try:
            now = time.time()
            hardware_is_active = (now - last_hardware_time) < HARDWARE_TIMEOUT_SEC

            if not hardware_is_active:
                if active_fault_mode != "none":
                    # Only simulate if user intentionally clicks a Fault Injection button
                    base = get_base_packet()
                    import numpy as np
                    base["v_rms"] = round(float(base["v_rms"] + np.random.normal(0, 0.15)), 2)
                    base["i_rms"] = round(float(base["i_rms"] + np.random.normal(0, 0.02)), 3)
                    base["rpm"] = round(float(base["rpm"] + np.random.normal(0, 1.2)), 1)
                    base["power_w"] = round(float(base["v_rms"] * base["i_rms"] * base["power_factor"]), 1)

                    if active_fault_mode == "unbalance":
                        packet = fault_injector.inject_mechanical_unbalance(base, severity=active_fault_intensity)
                    elif active_fault_mode == "bearing":
                        packet = fault_injector.inject_bearing_fault(base, severity=active_fault_intensity)
                    elif active_fault_mode == "overload":
                        packet = fault_injector.inject_thermal_overload(base, target_current_a=6.8)
                    elif active_fault_mode == "sag":
                        packet = fault_injector.inject_voltage_sag(base, target_voltage_v=172.0)
                    elif active_fault_mode == "single_phase":
                        packet = fault_injector.inject_single_phasing(base)
                    commit_telemetry_packet(packet, "DIGITAL_TWIN")
                else:
                    # No hardware and no fault injected: Do NOT generate artificial data! Show zeros / idle
                    idle_packet = {
                        "phase": "A",
                        "v_rms": 0.0,
                        "i_rms": 0.0,
                        "power_w": 0.0,
                        "power_factor": 1.0,
                        "freq_hz": 50.00,
                        "rpm": 0.0,
                        "accel_x": 0.0,
                        "accel_y": 0.0,
                        "accel_z": 0.0,
                        "relay_state": "CLOSED",
                        "fault_injected": "None (Awaiting Hardware Connection)"
                    }
                    commit_telemetry_packet(idle_packet, "STANDBY_NO_HARDWARE")

        except Exception as e:
            print(f"[!] Error in telemetry generator: {e}")

        time.sleep(1.0)

# Global handles for outbound actuation commands to ESP32
mqtt_publisher_client = None
serial_conn = None

def send_actuator_command(cmd_str):
    """Dispatches command ('TRIP' or 'RESET') to ESP32 via MQTT and/or USB Serial."""
    global mqtt_publisher_client, serial_conn
    dispatched = False

    # 1. Dispatch over MQTT
    if mqtt_publisher_client is not None:
        try:
            mqtt_publisher_client.publish("motor/phase_a/command", cmd_str)
            print(f"[+] Dispatched '{cmd_str}' to topic 'motor/phase_a/command' via MQTT")
            dispatched = True
        except Exception as e:
            print(f"[!] MQTT publish failed: {e}")

    # 2. Dispatch over USB Serial
    if serial_conn is not None:
        try:
            serial_conn.write(f"{cmd_str}\n".encode("utf-8"))
            serial_conn.flush()
            print(f"[+] Dispatched '{cmd_str}' over USB Serial")
            dispatched = True
        except Exception as e:
            print(f"[!] USB Serial write failed: {e}")

    return dispatched

# Optional MQTT Listener Thread to ingest real packets from ESP32
def mqtt_listener_worker():
    global last_hardware_time, mqtt_publisher_client
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        return

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            client.subscribe("motor/phase_a/telemetry")
            print("[+] MQTT Connected to localhost:1883 - Subscribed to motor/phase_a/telemetry")

    def on_message(client, userdata, msg):
        global last_hardware_time
        try:
            packet = json.loads(msg.payload.decode("utf-8"))
            last_hardware_time = time.time()  # Mark hardware alive!
            commit_telemetry_packet(packet, "LIVE_HARDWARE")
        except Exception:
            pass

    client = mqtt.Client()
    mqtt_publisher_client = client
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect("localhost", 1883, 60)
        client.loop_forever()
    except Exception:
        pass  # If no broker running locally, continue silently

# Background Serial Ingestion Worker (Auto-detects USB-connected ESP32)
serial_conn = None

def serial_listener_worker():
    global last_hardware_time, serial_conn
    try:
        import serial
        import glob
    except ImportError:
        return

    while True:
        try:
            # Find USB Serial ports
            ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
            if not ports:
                import serial.tools.list_ports
                com_ports = [p.device for p in serial.tools.list_ports.comports()]
                ports = [p for p in com_ports if "COM" in p or "tty" in p]

            if ports:
                port = ports[0]
                with serial.Serial(port, 115200, timeout=2.0) as ser:
                    serial_conn = ser
                    print(f"[+] Connected to ESP32 on USB Serial: {port}")
                    while True:
                        line = ser.readline().decode("utf-8", errors="ignore").strip()
                        if line.startswith("{") and line.endswith("}"):
                            try:
                                packet = json.loads(line)
                                last_hardware_time = time.time()
                                commit_telemetry_packet(packet, "LIVE_HARDWARE")
                            except json.JSONDecodeError:
                                pass
            else:
                time.sleep(2.0)
        except Exception:
            time.sleep(2.0)

# Start background workers
stream_thread = threading.Thread(target=telemetry_stream_worker, daemon=True)
stream_thread.start()

mqtt_thread = threading.Thread(target=mqtt_listener_worker, daemon=True)
mqtt_thread.start()

serial_thread = threading.Thread(target=serial_listener_worker, daemon=True)
serial_thread.start()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/telemetry")
def get_telemetry():
    with data_lock:
        data = dict(latest_telemetry)
        data["history"] = history_buffer
        data["active_fault_mode"] = active_fault_mode
        data["current_load_condition"] = current_load_condition
        data["data_source"] = current_data_source
    return jsonify(data)

@app.route("/api/ingest_hardware", methods=["POST"])
def ingest_hardware():
    """Endpoint for ESP32 or serial forwarders to post physical telemetry."""
    global last_hardware_time
    packet = request.get_json() or {}
    if packet:
        last_hardware_time = time.time()
        commit_telemetry_packet(packet, "LIVE_HARDWARE")
        return jsonify({"status": "SUCCESS", "mode": "LIVE_HARDWARE"})
    return jsonify({"status": "ERROR", "message": "Empty packet"}), 400

@app.route("/api/inject_fault", methods=["POST"])
def set_fault():
    global active_fault_mode, active_fault_intensity
    body = request.get_json() or {}
    mode = body.get("fault_mode", "none").lower()
    intensity = float(body.get("intensity", 1.0))

    if mode in ["none", "unbalance", "bearing", "overload", "sag", "single_phase"]:
        active_fault_mode = mode
        active_fault_intensity = intensity
        return jsonify({"status": "SUCCESS", "active_fault_mode": active_fault_mode, "intensity": active_fault_intensity})
    return jsonify({"status": "ERROR", "message": "Invalid fault mode"}), 400

@app.route("/api/set_load", methods=["POST"])
def set_load():
    global current_load_condition
    body = request.get_json() or {}
    load = body.get("load_condition", "Load_1_2kg")
    if load in ["No_Load", "Load_1_2kg", "Load_2_7.5kg"]:
        current_load_condition = load
        return jsonify({"status": "SUCCESS", "current_load_condition": current_load_condition})
    return jsonify({"status": "ERROR", "message": "Invalid load condition"}), 400

@app.route("/api/reset_relay", methods=["POST"])
def reset_relay():
    global active_fault_mode
    active_fault_mode = "none"  # Automatically clear any injected fault upon reset
    success, msg = smartgrid_engine.reset_trip_relay()
    if success:
        send_actuator_command("RESET")
    return jsonify({"status": "SUCCESS" if success else "NOTICE", "message": msg})

@app.route("/api/trip_relay_manual", methods=["POST"])
def trip_relay_manual():
    smartgrid_engine.relay_state = "TRIP_OPEN"
    smartgrid_engine.trip_latched = True
    smartgrid_engine.trip_active_reason = "MANUAL_OPERATOR_EMERGENCY_STOP"
    smartgrid_engine.trip_timestamp = time.strftime("%X")
    send_actuator_command("TRIP")
    return jsonify({"status": "SUCCESS", "message": "Emergency Trip Engaged"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f"  SMART GRID & MOTOR INTELLIGENCE DASHBOARD READY")
    print(f"  Access locally: http://localhost:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
