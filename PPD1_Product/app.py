"""
======================================================================================
PPD-1 FLASK WEB APPLICATION — app.py
Version: 5.8 (Render Free-Tier Memory & Timeout Optimizations)
======================================================================================
"""

from __future__ import annotations

import base64
import io
import json
import os
import pickle
import time
import uuid
import warnings
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.spatial import distance as dist

# ── Force TensorFlow to use minimal CPU memory for Render Free Tier ──
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MP_TASK_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
V5_MODEL_PATH = os.path.join(BASE_DIR, "vision_model_v5.tflite")
V5_SCALER_PATH = os.path.join(BASE_DIR, "vision_scaler_v5.pkl")
PHYS_MODEL_PATH = os.path.join(BASE_DIR, "stress_model.tflite")
PHYS_SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  
UPLOAD_FOLDER = Path(BASE_DIR) / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

EAR_THRESHOLD      = 0.20
DARK_CIRCLE_RATIO  = 0.85
PERCLOS_THRESHOLD  = 0.15

LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]
LEFT_BROW  = 107
RIGHT_BROW = 336
OUTER_EYES = [33, 263]
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 61, 291

MODELS_LOADED = False
v_scaler = p_scaler = v_interp = p_interp = detector = None

def load_models():
    global MODELS_LOADED, v_scaler, p_scaler, v_interp, p_interp, detector
    
    print("[SYSTEM] Initializing AI Models using Absolute Paths...")

    if not os.path.exists(MP_TASK_PATH):
        print(f"[FATAL ERROR] {MP_TASK_PATH} not found!")
        MODELS_LOADED = False
        return

    base_opts = python.BaseOptions(model_asset_path=MP_TASK_PATH)
    lm_opts   = vision.FaceLandmarkerOptions(
        base_options=base_opts,
        num_faces=1,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.FaceLandmarker.create_from_options(lm_opts)

    if TF_AVAILABLE and os.path.exists(V5_MODEL_PATH) and os.path.exists(V5_SCALER_PATH):
        try:
            with open(V5_SCALER_PATH, "rb") as f:
                v_scaler = pickle.load(f)
            v_interp = tf.lite.Interpreter(model_path=V5_MODEL_PATH)
            v_interp.allocate_tensors()
            print("[SYSTEM] V5 Vision ResNet loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load Vision Model. Error: {e}")
            v_scaler = None
            v_interp = None

    if TF_AVAILABLE and os.path.exists(PHYS_MODEL_PATH) and os.path.exists(PHYS_SCALER_PATH):
        try:
            with open(PHYS_SCALER_PATH, "rb") as f:
                p_scaler = pickle.load(f)
            p_interp = tf.lite.Interpreter(model_path=PHYS_MODEL_PATH)
            p_interp.allocate_tensors()
            print("[SYSTEM] Physical sensor AI (WESAD) loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load Physical Sensor AI. Error: {e}")
            p_scaler = None
            p_interp = None

    MODELS_LOADED = (detector is not None)
    print(f"[SYSTEM] PPD-1 V5 Engine ready. Models loaded: {MODELS_LOADED}")

load_models()

def calc_ear(eye_pts: np.ndarray) -> float:
    A = dist.euclidean(eye_pts[1], eye_pts[5])
    B = dist.euclidean(eye_pts[2], eye_pts[4])
    C = dist.euclidean(eye_pts[0], eye_pts[3])
    return (A + B) / (2.0 * C) if C > 1e-6 else 0.0

def check_dark_circles_lab(frame, landmarks, w, h):
    try:
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab_frame[:, :, 0] 
        l_eye_y, l_eye_x = int(landmarks[145].y * h), int(landmarks[145].x * w)
        l_cheek_y, l_cheek_x = int(landmarks[205].y * h), int(landmarks[205].x * w)
        r_eye_y, r_eye_x = int(landmarks[374].y * h), int(landmarks[374].x * w)
        r_cheek_y, r_cheek_x = int(landmarks[425].y * h), int(landmarks[425].x * w)
        
        l_ratio = np.mean(l_channel[max(0, l_eye_y-5):l_eye_y+5, max(0, l_eye_x-5):l_eye_x+5]) / (np.mean(l_channel[max(0, l_cheek_y-5):l_cheek_y+5, max(0, l_cheek_x-5):l_cheek_x+5]) + 1e-6)
        r_ratio = np.mean(l_channel[max(0, r_eye_y-5):r_eye_y+5, max(0, r_eye_x-5):r_eye_x+5]) / (np.mean(l_channel[max(0, r_cheek_y-5):r_cheek_y+5, max(0, r_cheek_x-5):r_cheek_x+5]) + 1e-6)
        return 1.0 if (l_ratio < 0.85 or r_ratio < 0.85) else 0.0
    except:
        return 0.0

def compute_v5_features(lm: list, blendshapes: list, h: int, w: int) -> dict:
    l_eye = np.array([(lm[i].x * w, lm[i].y * h) for i in LEFT_EYE])
    r_eye = np.array([(lm[i].x * w, lm[i].y * h) for i in RIGHT_EYE])
    ear   = (calc_ear(l_eye) + calc_ear(r_eye)) / 2.0
    pt_lb = np.array([lm[LEFT_BROW].x * w,  lm[LEFT_BROW].y * h])
    pt_rb = np.array([lm[RIGHT_BROW].x * w, lm[RIGHT_BROW].y * h])
    face_w   = dist.euclidean(np.array([lm[OUTER_EYES[0]].x * w, lm[OUTER_EYES[0]].y * h]), np.array([lm[OUTER_EYES[1]].x * w, lm[OUTER_EYES[1]].y * h])) + 1e-6
    etr = dist.euclidean(pt_lb, pt_rb) / face_w
    mouth_w_val = dist.euclidean((lm[MOUTH_LEFT].x * w, lm[MOUTH_LEFT].y * h), (lm[MOUTH_RIGHT].x * w, lm[MOUTH_RIGHT].y * h))
    mar = dist.euclidean((lm[MOUTH_TOP].x * w, lm[MOUTH_TOP].y * h), (lm[MOUTH_BOTTOM].x * w, lm[MOUTH_BOTTOM].y * h)) / mouth_w_val if mouth_w_val > 1e-6 else 0.0
    moe = mar / ear if ear > 1e-6 else 0.0
    far = dist.euclidean((lm[10].x * w, lm[10].y * h), (lm[152].x * w, lm[152].y * h)) / face_w
    blend_scores = [b.score for b in blendshapes] if blendshapes else [0.0] * 52
    v5_vector = np.array([ear, etr, mar, moe, dist.euclidean(pt_lb, np.mean(l_eye, axis=0)) / face_w, dist.euclidean(pt_rb, np.mean(r_eye, axis=0)) / face_w, far] + (blend_scores + [0.0] * 52)[:52], dtype=np.float32)
    return {"ear": float(ear), "etr": float(etr), "mar": float(mar), "moe": float(moe), "far": float(far), "v5_vector": v5_vector, "blendshapes": {b.category_name: round(b.score, 4) for b in blendshapes} if blendshapes else {}}

def run_vision_model(v5_vector: np.ndarray) -> float:
    if v_scaler is not None and v_interp is not None:
        try:
            v_interp.set_tensor(v_interp.get_input_details()[0]["index"], v_scaler.transform([v5_vector]).astype(np.float32))
            v_interp.invoke()
            return float(np.clip(float(v_interp.get_tensor(v_interp.get_output_details()[0]["index"])[0][0]), 0.0, 1.0))
        except: pass
    ear, etr, mar, moe = float(v5_vector[0]), float(v5_vector[1]), float(v5_vector[2]), float(v5_vector[3])
    return float(np.clip(max(0.0, (0.28 - ear) / 0.15) * 0.45 + max(0.0, (0.42 - etr) / 0.15) * 0.35 + min(1.0, mar * 2.5) * 0.20, 0.0, 1.0))

def run_physiology_model(gsr: float, hrv: float, hr: float) -> float:
    if p_scaler is not None and p_interp is not None:
        try:
            p_interp.set_tensor(p_interp.get_input_details()[0]["index"], p_scaler.transform([[gsr, hrv, hr]]).astype(np.float32))
            p_interp.invoke()
            return float(np.clip(float(p_interp.get_tensor(p_interp.get_output_details()[0]["index"])[0][0]), 0.0, 1.0))
        except: pass
    return float(np.clip(min(1.0, (gsr - 2.0) / 10.0) * 0.55 + min(1.0, (hr  - 60.0) / 60.0) * 0.45, 0.0, 1.0))

def get_medical_category(probability: float) -> dict:
    p = float(np.clip(probability, 0.0, 1.0)) * 100
    if p <= 40: return {"label": "Optimal Baseline", "sub": "Relaxed", "band": "00–40%", "color": "#22c55e", "bg": "rgba(34,197,94,0.08)", "bd": "rgba(34,197,94,0.22)"}
    elif p <= 55: return {"label": "Sub-Clinical", "sub": "Low Stress", "band": "41–55%", "color": "#14b8a6", "bg": "rgba(20,184,166,0.08)", "bd": "rgba(20,184,166,0.22)"}
    elif p <= 70: return {"label": "Elevated Sympathetic Tone", "sub": "Moderate Stress", "band": "56–70%", "color": "#f59e0b", "bg": "rgba(245,158,11,0.08)", "bd": "rgba(245,158,11,0.22)"}
    elif p <= 85: return {"label": "Acute Fatigue / Stress", "sub": "High", "band": "71–85%", "color": "#f97316", "bg": "rgba(249,115,22,0.08)", "bd": "rgba(249,115,22,0.22)"}
    else: return {"label": "Critical", "sub": "Consult Specialist", "band": "86–100%", "color": "#ef4444", "bg": "rgba(239,68,68,0.08)", "bd": "rgba(239,68,68,0.22)"}

def compute_holistic_score(v_prob: float, dc_flag: float, p_prob: float | None, perclos: float) -> float:
    score = v_prob + (dc_flag * 0.05)
    if p_prob is not None: score = (score * 0.70) + (p_prob * 0.30)
    if perclos >= PERCLOS_THRESHOLD: score += (perclos - PERCLOS_THRESHOLD) * 0.5
    return float(np.clip(score, 0.0, 1.0))

def process_frame(frame_bgr: np.ndarray) -> dict:
    if detector is None: 
        return {"face_detected": False, "error": "Server AI models failed to load. Check logs."}

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    if not result.face_landmarks: 
        return {"face_detected": False}

    lm = result.face_landmarks[0]
    h, w, _ = frame_bgr.shape
    feats = compute_v5_features(lm, result.face_blendshapes[0] if result.face_blendshapes else [], h, w)
    dc_flag = check_dark_circles_lab(frame_bgr, lm, w, h)
    v_prob = run_vision_model(feats["v5_vector"])
    holistic = compute_holistic_score(v_prob, dc_flag, None, 0.0)

    return {
        "face_detected": True,
        "ear": feats["ear"], "etr": feats["etr"], "mar": feats["mar"], "moe": feats["moe"], "far": feats["far"],
        "dc_flag": dc_flag,
        "v_prob": round(v_prob, 4), "holistic": round(holistic, 4),
        "blendshapes": feats["blendshapes"],
        "category": get_medical_category(holistic)
    }

sessions: dict[str, dict] = {}
def get_or_create_session(sid: str) -> dict:
    if sid not in sessions: sessions[sid] = {"id": sid, "frames": [], "blinks": 0, "eye_closed": False, "perclos_buf": deque(maxlen=300)}
    return sessions[sid]

report_archive: list[dict] = []

def build_report(sid: str, session_meta: dict, sensor_data: dict | None = None) -> dict:
    s = sessions.get(sid, {})
    frames = s.get("frames", [])
    if not frames: return {}

    avg = lambda key: round(float(np.mean([f[key] for f in frames if key in f])), 4)
    v_p = avg("v_prob")
    dc_f = avg("dc_flag")
    perc = round(float(sum(list(s.get("perclos_buf", []))) / len(list(s.get("perclos_buf", [])))) if list(s.get("perclos_buf", [])) else 0.0, 4)

    p_prob = run_physiology_model(sensor_data.get("gsr", 4.5), sensor_data.get("hrv", 0.5), sensor_data.get("hr", 80.0)) if sensor_data else None
    holistic = compute_holistic_score(v_p, dc_f, p_prob, perc)
    duration = session_meta.get("duration_s", 0)
    bpm_rate = round((s.get("blinks", 0) / duration * 60) if duration > 0 else 0.0, 1)

    combined = {}
    for f in frames:
        for k, v in f.get("blendshapes", {}).items(): combined.setdefault(k, []).append(v)
    avgs = {k: round(float(np.mean(v)), 4) for k, v in combined.items()}
    bs_summary = dict(sorted(avgs.items(), key=lambda x: x[1], reverse=True)[:10])

    rec = {
        "report_id": str(uuid.uuid4())[:8].upper(),
        "date": datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
        "session_id": session_meta.get("patient_id", "ANON"),
        "mode": session_meta.get("mode", "Vision Only"),
        "duration_s": duration,
        "total_blinks": s.get("blinks", 0),
        "blink_rate_bpm": bpm_rate,
        "perclos_pct": round(perc * 100, 1),
        "avg_ear": avg("ear"), "avg_etr": avg("etr"), "avg_mar": avg("mar"), "avg_moe": avg("moe"), "avg_far": avg("far"),
        "dark_circle_pct": round(dc_f * 100, 1),
        "vision_stress_pct": round(v_p * 100, 1),
        "phys_stress_pct": round(p_prob * 100, 1) if p_prob else None,
        "holistic_pct": round(holistic * 100, 1),
        "verdict": get_medical_category(holistic),
        "blendshape_summary": bs_summary,
    }
    report_archive.append(rec)
    return rec

@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/analyze/live", methods=["POST"])
def analyze_live():
    try:
        data = request.get_json(force=True)
        b64 = data.get("frame", "")
        sid = data.get("session_id", "default")
        phys = data.get("sensor", None)  

        img_bytes = base64.b64decode(b64.split(",")[-1])
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None: return jsonify({"error": "Failed to decode frame from frontend."})
        
        metrics = process_frame(frame)
        if "error" in metrics: return jsonify(metrics)

        if metrics.get("face_detected"):
            sess = get_or_create_session(sid)
            sess["frames"].append(metrics)
            ear = metrics.get("ear", 1.0)
            if ear < EAR_THRESHOLD:
                if not sess["eye_closed"]: sess["blinks"] += 1; sess["eye_closed"] = True
            else: sess["eye_closed"] = False
                
            sess["perclos_buf"].append(1 if ear < EAR_THRESHOLD else 0)
            perclos = float(sum(list(sess["perclos_buf"])) / len(sess["perclos_buf"]))
            metrics["perclos"] = round(perclos, 4)
            metrics["blinks"] = sess["blinks"]
            dc_val = metrics.get("dc_flag", 0.0)

            if phys:
                p_prob = run_physiology_model(phys.get("gsr", 4.5), phys.get("hrv", 0.5), phys.get("hr", 80.0))
                metrics["p_prob"] = round(p_prob, 4)
                metrics["holistic"] = round(compute_holistic_score(metrics["v_prob"], dc_val, p_prob, perclos), 4)
            else:
                metrics["holistic"] = round(compute_holistic_score(metrics["v_prob"], dc_val, None, perclos), 4)
                
            metrics["category"] = get_medical_category(metrics["holistic"])
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": f"Server crash during frame processing: {str(e)}"})

@app.route("/api/analyze/image", methods=["POST"])
def analyze_image():
    if "file" not in request.files: return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    sid = request.form.get("session_id", str(uuid.uuid4())[:8])
    pid = request.form.get("patient_id", "ANON")
    path = UPLOAD_FOLDER / f"{sid}_{f.filename}"
    f.save(str(path))
    frame = cv2.imread(str(path))
    metrics = process_frame(frame)
    path.unlink(missing_ok=True)
    if metrics.get("face_detected"):
        sess = get_or_create_session(sid)
        sess["frames"].append(metrics)
        report = build_report(sid, {"patient_id": pid, "mode": "Image Analysis", "duration_s": 0})
        return jsonify({"metrics": metrics, "report": report})
    return jsonify({"metrics": metrics})

@app.route("/api/analyze/video", methods=["POST"])
def analyze_video():
    if "file" not in request.files: return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    sid = request.form.get("session_id", str(uuid.uuid4())[:8])
    pid = request.form.get("patient_id", "ANON")
    path = UPLOAD_FOLDER / f"{sid}_{f.filename}"
    f.save(str(path))
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration_s = round(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps, 1)
    sample_every = max(1, int(fps // 4))   
    sess = get_or_create_session(sid)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % sample_every != 0: continue
        frame = cv2.resize(frame, (640, 480))
        m = process_frame(frame)
        if m.get("face_detected"):
            sess["frames"].append(m)
            ear = m.get("ear", 1.0)
            if ear < EAR_THRESHOLD:
                if not sess["eye_closed"]: sess["blinks"] += 1; sess["eye_closed"] = True
            else: sess["eye_closed"] = False
            sess["perclos_buf"].append(1 if ear < EAR_THRESHOLD else 0)
    cap.release()
    path.unlink(missing_ok=True)
    report = build_report(sid, {"patient_id": pid, "mode": "Video Analysis", "duration_s": duration_s})
    return jsonify({"report": report})

@app.route("/api/report/export/csv", methods=["GET"])
def export_csv():
    import csv, io
    buf = io.StringIO()
    fields = ["report_id","date","session_id","mode","duration_s","total_blinks","blink_rate_bpm","perclos_pct","avg_ear","avg_etr","avg_mar","avg_moe","avg_far","vision_stress_pct","phys_stress_pct","holistic_pct","verdict_label"]
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in report_archive:
        row = dict(r); row["verdict_label"] = r.get("verdict", {}).get("label","")
        w.writerow(row)
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode()), mimetype="text/csv", as_attachment=True, download_name=f"PPD1_Records.csv")

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000, threaded=True)
