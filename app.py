# ==========================================
# 0. CRITICAL HARDWARE CONFIGURATION
# ==========================================
import os

# This forces the Roboflow/ONNX package to completely ignore your GPU.
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "CPUExecutionProvider"

# Standard library imports
import shutil
import base64
import json
import time
import uuid
import gc

# Third-party imports
import cv2
import torch
import requests
import paho.mqtt.client as mqtt
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from threading import Lock, Thread
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from dotenv import load_dotenv
# AI Model imports
from inference import get_model

# ==========================================
# 1. INITIALIZATION & SETUP
# ==========================================

# Load hidden secrets (like Telegram tokens) from a .env file
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')

# --- Trackers to prevent notification spam ---
sensor_alert_history = {"motion": 0, "tilt": 0, "gunshot": 0}
SENSOR_COOLDOWN = 5  # Wait 5 seconds before repeating sensor alerts
# confidence
WEAPON_CONFIDENCE_THRESHOLD = 0.30
# --- Define Folder Paths ---
BASE_DIR = os.getcwd()
MODEL_FOLDER = os.path.join(BASE_DIR, "speciesnet_model")
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
DETECTIONS_DIR = os.path.join(BASE_DIR, "static", "detections")
VIDEO_DIR = os.path.join(BASE_DIR, "uploads", "videos")
TEMP_DIR = os.path.join(BASE_DIR, "temp_inference", "frames")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads") 

# Tell PyTorch and SpeciesNet to use our custom cache folder
os.environ["XDG_CACHE_HOME"] = CACHE_DIR
os.environ["SPECIESNET_CACHE"] = CACHE_DIR
os.environ["TORCH_HOME"] = CACHE_DIR

# Create all necessary folders if they don't exist yet
for folder in [DETECTIONS_DIR, VIDEO_DIR, TEMP_DIR, UPLOAD_DIR, os.path.join(BASE_DIR, "templates")]:
    os.makedirs(folder, exist_ok=True)

# Initialize the Flask Web App
app = Flask(__name__)
CORS(app) # Allows front-end to talk to back-end easily

# Detect if we have a GPU available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("\n" + "="*60)
if DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU DETECTED AND ACTIVE: {gpu_name}")
    print(f"✅ TOTAL VRAM: {vram_gb:.2f} GB")
else:
    print("❌ NO GPU DETECTED! PyTorch is falling back to the slow CPU.")
    print("⚠️  Check your CUDA installation and PyTorch version.")
print("="*60 + "\n")

print(f"📂 WORKING DIR: {BASE_DIR}")
print(f"🖥️ MAIN AI DEVICE: {DEVICE}")

# Initialize SQLite Database for storing results
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(BASE_DIR, "species_data.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

video_processor = None # Will hold our AI processing class later

# ==========================================
# 2. HELPER FUNCTIONS & DATABASE MODELS
# ==========================================

def send_telegram_alert(message):
    """Sends a push notification to your phone via Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        requests.post(url, data=payload)
        print(f"✅ Telegram sent: {message}")
    except Exception as e: 
        print(f"❌ Telegram Error: {e}")

class VideoRecord(db.Model):
    """Database table to store uploaded video details."""
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(255), nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    processed = db.Column(db.Boolean, default=False)
    detections = db.relationship('DetectionResult', backref='video', lazy=True)

class DetectionResult(db.Model):
    """Database table to store specific animals/weapons found in videos."""
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey('video_record.id'), nullable=False)
    species = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp_in_video = db.Column(db.Float, nullable=False)
    image_url = db.Column(db.String(255), nullable=True)
class SensorEvent(db.Model):
    """Database table to store ESP32 hardware sensor alerts."""
    id = db.Column(db.Integer, primary_key=True)
    event_type = db.Column(db.String(50), nullable=False)  # e.g., 'motion', 'tilt', 'gunshot'
    value = db.Column(db.Float, nullable=True)             # e.g., the specific tilt angle
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Create the database file if it doesn't exist
with app.app_context(): db.create_all()


# ==========================================
# 3. THE ASYMMETRIC AI MANAGER
# ==========================================
class ModelManager:
    """Loads and manages both the GPU and CPU models safely."""
    _instance = None
    _lock = Lock()

    def __new__(cls):
        # Singleton pattern: Ensures we only load the heavy AI models ONCE into memory.
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.model = None          # SpeciesNet (GPU)
                    cls._instance.weapon_model = None   # Roboflow (CPU)
                    cls._instance.initialized = False
        return cls._instance

    def initialize(self):
        """Loads models into hardware during server startup."""
        if self.initialized: return True
        if not os.path.exists(MODEL_FOLDER):
            print(f"❌ ERROR: SPECIESNET MODEL NOT FOUND AT: {MODEL_FOLDER}")
            return False

        print("="*60)
        print("🧠 STARTING ASYMMETRIC AI ARCHITECTURE...")
        print(f"📍 SpeciesNet loading to: {DEVICE} (GPU)")
        print(f"📍 Weapon Model loading to: CPU (Intel i5)")
        print("="*60)

        try:
            # 1. Load SpeciesNet to GPU
            from speciesnet import SpeciesNet
            self.model = SpeciesNet(model_name=MODEL_FOLDER, components='all', geofence=True, multiprocessing=False)
            print("✅ SpeciesNet Loaded Successfully (GPU)")
            
            # 2. Load Weapon Model to CPU
            print("⏳ Loading Weapon Model...")
            self.weapon_model = get_model(model_id="rifle-1b8vx/2", api_key=ROBOFLOW_API_KEY)
            print("✅ Weapon Model Loaded Successfully (CPU)")
            
            self._warmup()
            self.initialized = True
            return True
        except Exception as e:
            print(f"❌ Model load error: {e}")
            return False

    def _warmup(self):
        """Runs a fake image to wake up the GPU so the first real video isn't slow."""
        print("🔥 Warming up GPU...")
        try:
            import numpy as np
            from PIL import Image
            dummy_path = os.path.join(BASE_DIR, "warmup.jpg")
            Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)).save(dummy_path)
            _ = self.model.predict(filepaths=[dummy_path], country="IND", run_mode='single_thread', progress_bars=False)
            if os.path.exists(dummy_path): os.remove(dummy_path)
            print("🔥 Warmup complete")
        except Exception as e:
            print(f"⚠️ Warmup failed: {e}")
            
    def predict(self, filepath):
        """Used for single image uploads via the web interface."""
        if not self.model: return None
        try:
            return self.model.predict(filepaths=[filepath], country="IND", run_mode='single_thread', batch_size=1, progress_bars=False)
        except Exception as e:
            print(f"Prediction Error: {e}")
            return None

model_manager = ModelManager()


# ==========================================
# 4. THE TWO-STAGE VIDEO PROCESSOR
# ==========================================
class BatchVideoProcessor:
    """Handles splitting videos into frames and passing them to the models."""
    def __init__(self, model_manager):
        self.model_manager = model_manager

    def process_video_batched(self, video_path, batch_size=8, sample_fps=1, min_confidence=0.3, country='IND', rotate_video=False):
        # Note: batch_size=2 protects your 4GB VRAM from crashing
        print(f"\n🏆 PROCESSING VIDEO: {video_path}")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps == 0: return []
        
        frame_interval = int(max(1, fps / sample_fps)) # How many frames to skip
        paths_buffer, timestamps_buffer, all_detections = [], [], []
        video_alert_history = {} 
        
        # ⏱️ THE COOLDOWN TRACKER
        # Tracks when the CPU last fired, and if we already sent a Telegram alert for this video
        video_state = {"last_gun_time": -999, "alert_sent": False}
        
        frame_count = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break 
                
                # Only process 1 frame every second
                if frame_count % frame_interval == 0:
                    current_time_sec = frame_count / fps
                    
                    # Optional: Rotate video if AMB82 is mounted sideways
                    # Optional: Rotate video if AMB82 is mounted sideways
                    if rotate_video:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    
                    # Save frame temporarily for AI to read
                    frame_path = os.path.join(TEMP_DIR, f"frame_{uuid.uuid4().hex}.jpg")
                    cv2.imwrite(frame_path, frame)
                    
                    paths_buffer.append(frame_path)
                    timestamps_buffer.append(current_time_sec)
                    
                    # When buffer is full (2 frames), send to AI
                    if len(paths_buffer) >= batch_size:
                        batch_detections = self._process_batch(
                            paths_buffer, timestamps_buffer, country, min_confidence, video_alert_history, video_state
                        )
                        all_detections.extend(batch_detections)
                        
                        # Clean up temp frames to save hard drive space
                        for p in paths_buffer:
                            if os.path.exists(p): os.remove(p)
                        paths_buffer, timestamps_buffer = [], []
                        print(f"⏳ Progress: {frame_count/total_frames:.1%}")
                frame_count += 1
                
            # Process any remaining frames at the end of the video
            if paths_buffer:
                batch_detections = self._process_batch(
                    paths_buffer, timestamps_buffer, country, min_confidence, video_alert_history, video_state
                )
                all_detections.extend(batch_detections)
                for p in paths_buffer:
                    if os.path.exists(p): os.remove(p)
        finally:
            cap.release()
            
        print(f"✅ Video complete. Found {len(all_detections)} detections.")
        return all_detections

    def _process_batch(self, filepaths, timestamps, country, min_confidence, alert_history, video_state):
        if not self.model_manager.model: return []
        try:
            path_to_time = {fp: ts for fp, ts in zip(filepaths, timestamps)}
            
            # --- STAGE 1: SPECIESNET (GPU) ---
            result = self.model_manager.model.predict(filepaths=filepaths, country=country, batch_size=len(filepaths))
            valid_detections = []
            predictions = result.get('predictions', {})
            
            # Formatting magic to handle SpeciesNet output
            if isinstance(predictions, list):
                iterator = zip(filepaths, predictions)
                is_dict = False
            elif isinstance(predictions, dict):
                iterator = predictions.items()
                is_dict = True
            else: return []
            
            for item in iterator:
                path_key, pred_data = item if not is_dict else item
                if not pred_data: continue
                class_data = pred_data.get("classifications", {})
                if not class_data: continue
                
                top_score = class_data.get("scores", [0])[0]
                if top_score >= min_confidence:
                    top_class = class_data.get("classes", ["Unknown"])[0]
                    
                    # Clean up scientific name (e.g. "Panthera tigris; Tiger" -> "Tiger")
                    if ";" in top_class:
                        parts = [p.strip() for p in top_class.split(";") if p.strip()]
                        common_name = parts[-1].title()
                    else:
                        common_name = top_class.title()
                        
                    time_sec = path_to_time.get(path_key, 0)
                    unique_name = f"det_{uuid.uuid4().hex[:8]}.jpg"
                    save_path = os.path.join(DETECTIONS_DIR, unique_name)
                    image_url = None
                    is_weapon_threat = False
                    weapon_conf = 0.0

                    # Read original image to draw boxes and resize
                    if os.path.exists(path_key):
                        img = cv2.imread(path_key)
                        if img is not None:
                            # 🚀 PERFORMANCE FIX: Resize the image BEFORE making the CPU think
                            h, w = img.shape[:2]
                            if w > 640:
                                scale = 640 / w
                                img = cv2.resize(img, (640, int(h * scale)))
                            # --- STAGE 2: WEAPON DETECTION (CPU) ---
                            # Rules: Only run if it's a Human AND 3 seconds have passed since last gun detection
                            if common_name.lower() == "human" and (time_sec - video_state["last_gun_time"]) > 3.0:
                                print(f"👤 Human spotted at {time_sec:.1f}s. Running Weapon Scan on CPU...")
                                
                                # Run inference on the Intel i5 CPU
                                weapon_res = self.model_manager.weapon_model.infer(img,confidence=WEAPON_CONFIDENCE_THRESHOLD)
                                
                                # If weapon found...
                                if len(weapon_res[0].predictions) > 0:
                                    print("🚨 LETHAL WEAPON DETECTED!")
                                    
                                    # Reset the 3-second cooldown timer
                                    video_state["last_gun_time"] = time_sec 
                                    is_weapon_threat = True
                                    weapon_conf = float(weapon_res[0].predictions[0].confidence)
                                    
                                    # Draw Red Bounding Boxes on the image
                                    for pred in weapon_res[0].predictions:
                                        x1 = int(pred.x - (pred.width / 2))
                                        y1 = int(pred.y - (pred.height / 2))
                                        x2 = int(pred.x + (pred.width / 2))
                                        y2 = int(pred.y + (pred.height / 2))
                                        label = f"{pred.class_name} {pred.confidence:.2f}"
                                        
                                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red Box
                                        cv2.putText(img, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                    
                                    # Send Telegram Alert (ONLY ONCE per video to prevent spam)
                                    if not video_state["alert_sent"]:
                                        msg = f"🚨 *LETHAL WEAPON DETECTED* 🚨\n\n🎯 *Threat:* Armed Human\n⏱️ *Video Time:* {int(time_sec)}s\n📍 *Unit:* Field Cam 01"
                                        Thread(target=send_telegram_alert, args=(msg,)).start()
                                        video_state["alert_sent"] = True 
                            
                            # Save final image to static folder
                            cv2.imwrite(save_path, img)
                            image_url = f"/static/detections/{unique_name}"
                            
                            # Standard Wildlife Telegram Alert (Ignores humans to prevent spam)
# Standard Wildlife Telegram Alert (Ignores humans and empty frames)
                            excluded_categories = ["human", "blank", "unknown", "none"]       
                            if top_score > 0.75 and not is_weapon_threat and common_name.lower() not in excluded_categories:
                                last_alert = alert_history.get(common_name, -999)
                                if (time_sec - last_alert) > 30: # Don't spam if a monkey sits in front of camera
                                    msg = f"🐾 *WILDLIFE SIGHTING*\n\n🦁 *Species:* {common_name}\n🎯 *Confidence:* {top_score:.1%}\n⏱️ *Video Time:* {int(time_sec)}s"
                                    Thread(target=send_telegram_alert, args=(msg,)).start()
                                    alert_history[common_name] = time_sec
                    # Log the original SpeciesNet detection to the database
                   # Log the original SpeciesNet detection to the database (excluding blanks)
                    # If it's a human WITH a weapon, completely override the normal human log
                    if is_weapon_threat:
                        valid_detections.append({
                            "species": "ARMED HUMAN", "confidence": weapon_conf,
                            "timestamp": time_sec, "image_url": image_url
                        })
                    else:
                        # Log normal wildlife and unarmed humans
                        if common_name.lower() not in ["blank", "unknown", "none"]:
                            valid_detections.append({
                                "species": common_name, "confidence": float(top_score),
                                "timestamp": time_sec, "image_url": image_url
                            })

            return valid_detections
        except Exception as e:
            print(f"❌ Batch error: {e}")
            return []


# ==========================================
# 5. SENSORS (MQTT) & FLASK WEB ROUTES
# ==========================================

# --- Sensor Globals ---
last_status = {
    "motion": 0, "tilt": 0.0, "gunshot": 0, "temp": None,
    "free_heap": None, "min_heap": None, "rssi": None, "uptime": None
}
last_seen = 0
gunshot_timestamp = 0

def on_mqtt_connect(client, userdata, flags, rc):
    """Fires when local MQTT broker connects."""
    print(f"✅ Connected to MQTT Broker with result code {rc}")
    client.subscribe([("security/events", 0), ("security/heartbeat", 0)])

def on_mqtt_message(client, userdata, msg):
    """Processes incoming sensor data from ESP32."""
    global last_status, last_seen, gunshot_timestamp, sensor_alert_history
    current_time = time.time()
    last_seen = current_time

    # Helper function to save to DB from a background thread
    def log_event_to_db(event_type, value=None):
        with app.app_context():
            new_event = SensorEvent(event_type=event_type, value=value)
            db.session.add(new_event)
            db.session.commit()
            print(f"💾 Logged {event_type.upper()} event to database.")

    try:
        payload = msg.payload.decode('utf-8')
        data = json.loads(payload)
        
        if msg.topic == "security/events":
            print(f"📨 ESP Event [{msg.topic}]: {data}")

        # Update global status memory for the web dashboard to read
        for k in last_status:
            if k in data:
                last_status[k] = data[k]
                if k == "gunshot" and data[k] == 1: 
                    gunshot_timestamp = current_time

        # Sensor-specific logic and logging
        if msg.topic == "security/events":
            
            # 1. MOTION
            if data.get('motion') == 1:
                if (current_time - sensor_alert_history['motion']) > SENSOR_COOLDOWN:
                    msg_text = f"🏃 *MOTION DETECTED* 🏃\n\n⏱️ *Time:* {datetime.now().strftime('%H:%M:%S')}\n📍 *Unit:* Field Cam 01"
                    Thread(target=send_telegram_alert, args=(msg_text,)).start()
                    log_event_to_db("motion") # Save to DB
                    sensor_alert_history['motion'] = current_time

            # 2. TILT
            tilt_val = data.get('tilt', 0.0)
            if tilt_val > 30: # If camera falls off tree
                if (current_time - sensor_alert_history['tilt']) > SENSOR_COOLDOWN:
                    msg_text = f"⚠️ *DEVICE TILT WARNING* ⚠️\n\n📉 *Angle:* {tilt_val}°\n📍 *Unit:* Field Cam 01\nCheck mounting immediately."
                    Thread(target=send_telegram_alert, args=(msg_text,)).start()
                    log_event_to_db("tilt", tilt_val) # Save to DB with angle
                    sensor_alert_history['tilt'] = current_time

            # 3. GUNSHOT
            if data.get('gunshot') == 1: 
                if (current_time - sensor_alert_history['gunshot']) > 1:
                    msg_text = f"🔥 *GUNSHOT DETECTED* 🔥\n\n⏱️ *Time:* {datetime.now().strftime('%H:%M:%S')}\n📍 *Unit:* Field Cam 01\n*IMMEDIATE ACTION REQUIRED*"
                    Thread(target=send_telegram_alert, args=(msg_text,)).start()
                    log_event_to_db("gunshot") # Save to DB
                    sensor_alert_history['gunshot'] = current_time
            # 4. MANUAL WAKE ACKNOWLEDGMENT
            if data.get('manual_wake') == 1:
                log_event_to_db("system", value=1.0) # Logging as a system event
                print("✅ Received ACK: Camera manually awakened by user.")

    except Exception as e:
        print(f"❌ MQTT Error: {e}")

# --- FLASK ROUTES ---

def handle_amb82_video(video_file, sample_fps=1, min_conf=0.5, country='IND', rotate_video=False):
    """Helper function to save and process uploaded videos."""
    filename = secure_filename(f"amb82_{int(time.time())}.mp4")
    file_path = os.path.join(VIDEO_DIR, filename)
    video_file.save(file_path)

    new_video = VideoRecord(filename=filename, filepath=file_path)
    db.session.add(new_video)
    db.session.commit()
    print(f"💾 Video saved to DB: {filename}")

    global video_processor
    if video_processor is None:
        video_processor = BatchVideoProcessor(model_manager)

    try:
        # 1. Get the raw, unfiltered detections from the AI
        raw_detections = video_processor.process_video_batched(
            file_path, batch_size=2, sample_fps=sample_fps, 
            min_confidence=min_conf, country=country, rotate_video=rotate_video
        )
        
        # ==========================================
        # 🚀 THE "SMART FILTER" PIPELINE
        # ==========================================
        CONFIDENCE_THRESHOLD = 0.55
        TIME_GAP_THRESHOLD = 5

        clean_detections = []
        last_seen_dict = {}

        # Ensure they are in chronological order
        raw_detections.sort(key=lambda x: x['timestamp'])

        for d in raw_detections:
            # Rule 1: Confidence Check
            if d['confidence'] < CONFIDENCE_THRESHOLD and d['species'] != "ARMED HUMAN": 
                continue
            
            # Rule 2: Spam Prevention (5-second gap per species)
            last_time = last_seen_dict.get(d['species'], -999)
            if (d['timestamp'] - last_time) > TIME_GAP_THRESHOLD:
                clean_detections.append(d)
                last_seen_dict[d['species']] = d['timestamp']

        # Rule 3: Empty Video Purge (Only save if clean_detections has items)
        if clean_detections:
            for d in clean_detections:
                res = DetectionResult(
                    video_id=new_video.id, species=d['species'], confidence=d['confidence'],
                    timestamp_in_video=d['timestamp'], image_url=d.get('image_url')
                )
                db.session.add(res)
                
        new_video.processed = True
        db.session.commit()
        
        # Send ONLY the clean data to the UI
        return {
            "success": True, 
            "video_id": new_video.id, 
            "count": len(clean_detections), 
            "results": clean_detections
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False, "error": str(e)}

@app.route('/')
def index(): return render_template('index.html')

@app.route('/sensor')
def sen(): return render_template('sensor.html')

@app.route('/field_unit')
def amb82_analysis(): return render_template('amb82_dashboard.html')
@app.route('/analytics')
def analytics(): return render_template('analytics.html')

@app.route('/test')
def simple_test(): return render_template('simple_test.html')

@app.route('/api/detect', methods=['POST'])
def detect():
    """Handles single image uploads from the web interface."""
    try:
        data = request.json
        img_data = data.get('image')
        if not img_data: return jsonify(success=False, error="No image data"), 400
        
        if "," in img_data: _, encoded = img_data.split(",", 1)
        else: encoded = img_data
        try: binary = base64.b64decode(encoded)
        except Exception: return jsonify(success=False, error="Invalid image encoding"), 400
        
        filename = f"upload_{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(UPLOAD_DIR, filename)
        with open(filepath, "wb") as f: f.write(binary)

        # Initialize models if they somehow fell asleep
        if not model_manager.initialized:
            print("⚠️ Model wasn't loaded! Attempting to load now...")
            model_manager.initialize()

        if model_manager.model:
            result = model_manager.predict(filepath)
            
            # 1. Read the image into OpenCV BEFORE deleting the temp file
            img = cv2.imread(filepath)
            if os.path.exists(filepath): os.remove(filepath)
            
            if result:
                predictions = result.get('predictions', {})
                if not predictions: 
                    return jsonify(success=True, species="None", scientific_name="N/A", confidence=0.0)
                
                if isinstance(predictions, list): pred_data = predictions[0]
                elif isinstance(predictions, dict): pred_data = next(iter(predictions.values()))
                else: return jsonify(success=False, error="Unknown prediction format"), 500
                
                class_data = pred_data.get("classifications", {})
                if not class_data: 
                    return jsonify(success=True, species="None", scientific_name="N/A", confidence=0.0)

                top_class = class_data.get("classes", ["Unknown"])[0]
                top_score = class_data.get("scores", [0])[0]

                if ";" in top_class:
                    parts = [p.strip() for p in top_class.split(";") if p.strip()]
                    species = parts[-1].title()
                    scientific = parts[-2] if len(parts) >= 2 else species
                else:
                    species = top_class.title()
                    scientific = top_class

              # ==========================================
                # 🚀 NEW: WEAPON DETECTION FOR SINGLE IMAGES
                # ==========================================
                
                # Debug Print 1: See what SpeciesNet actually returned
                print(f"\n📸 UI UPLOAD SCANNED: GPU thinks it is a '{species}' (Conf: {top_score:.2f})")
                
                processed_b64 = None
                
                # Make the check more forgiving
                is_human = "human" in species.lower() or "person" in species.lower()

                if is_human and img is not None:
                    print("👤 Human detected! Routing image to Intel i5 CPU for Weapon Scan...")
                    
                    # Resize for CPU speed
                    h, w = img.shape[:2]
                    if w > 640:
                        scale = 640 / w
                        img = cv2.resize(img, (640, int(h * scale)))
                    
                    try:
                        # Run Weapon Model on Intel i5
                        weapon_res = model_manager.weapon_model.infer(img,confidence=WEAPON_CONFIDENCE_THRESHOLD)
                        
                        # Debug Print 2: See how many weapons the CPU found
                        print(f"🎯 Weapon scan complete. Found {len(weapon_res[0].predictions)} threats.")
                        
                        if len(weapon_res[0].predictions) > 0:
                            print("🚨 LETHAL WEAPON DETECTED in UI upload! Drawing boxes...")
                            species = "ARMED HUMAN"
                            scientific = "Lethal Threat Detected"
                            top_score = float(weapon_res[0].predictions[0].confidence)
                            
                            # Draw Red Bounding Boxes
                            for pred in weapon_res[0].predictions:
                                x1 = int(pred.x - (pred.width / 2))
                                y1 = int(pred.y - (pred.height / 2))
                                x2 = int(pred.x + (pred.width / 2))
                                y2 = int(pred.y + (pred.height / 2))
                                label = f"{pred.class_name} {pred.confidence:.2f}"
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(img, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            
                            # Convert the drawn image back to base64
                            _, buffer = cv2.imencode('.jpg', img)
                            processed_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
                            
                    except Exception as e:
                        print(f"❌ CPU Weapon Model Error: {e}")
                        
                elif is_human and img is None:
                    print("❌ ERROR: SpeciesNet saw a human, but OpenCV failed to read the image file.")

                return jsonify({
                    "success": True, 
                    "species": species, 
                    "scientific_name": scientific, 
                    "confidence": float(top_score),
                    "processed_image": processed_b64 
                })
            else:
                return jsonify(success=False, error="Inference failed."), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(success=False, error=str(e)), 500

@app.route('/status')
def status():
    """Endpoint for web dashboard to poll live sensor data."""
    t = time.time()
    return jsonify({
        **last_status,
        "esp_online": (t - last_seen) < 62, # Heartbeat check
        "gunshot": 1 if (t - gunshot_timestamp) < 5 else 0,
        "model_loaded": model_manager.model is not None
    })

@app.route('/api/video/upload', methods=['POST'])
def upload_video():
    """Endpoint where the IoT Camera or UI posts videos."""
    if 'video' not in request.files: return jsonify(success=False), 400
    video_file = request.files['video']
    if video_file.filename == '': return jsonify(success=False), 400
    
    country = request.form.get('country', 'IND')
    response_mode = request.args.get('mode', 'simple')
    
    # Check if the request came from the UI to disable rotation
    source = request.form.get('source', 'amb82')
    should_rotate = True if source == 'amb82' else False
    
    full_data = handle_amb82_video(video_file, country=country, rotate_video=should_rotate)
    
    if response_mode == 'simple': return jsonify({"success": True, "status": "Ack", "id": full_data.get('video_id')}), 200
    else: return jsonify(full_data), 200

@app.route('/api/history')
def get_history():
    """Endpoint to fetch past detections for the dashboard."""
    videos = VideoRecord.query.order_by(VideoRecord.upload_time.desc()).all()
    output = []

    for v in videos:
        # Since we did the "Empty Video Purge" during upload, 
        # any video with 0 detections in the DB can just be skipped
        if not v.detections: 
            continue
            
        # Sort the already-clean detections chronologically
        sorted_detections = sorted(v.detections, key=lambda x: x.timestamp_in_video)
        
        # Format the data for the UI
        clean_detections = [{
            "species": d.species, 
            "confidence": d.confidence,
            "time": d.timestamp_in_video, 
            "image_url": d.image_url
        } for d in sorted_detections]

        output.append({
            "id": v.id, 
            "filename": v.filename, 
            "time": v.upload_time, 
            "detections": clean_detections
        })
            
    return jsonify(output)
@app.route('/api/sensor_history')
def get_sensor_history():
    """Endpoint to fetch past sensor events for the dashboard log."""
    try:
        # Get the 50 most recent events from the database
        events = SensorEvent.query.order_by(SensorEvent.timestamp.desc()).limit(50).all()
        output = []
        for e in events:
            output.append({
                "id": e.id,
                "type": e.event_type,
                "value": e.value,
                "timestamp": e.timestamp.isoformat() + "Z" # Format as standard UTC string for JS
            })
        return jsonify({"success": True, "events": output})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
# Static file servers for videos and images
@app.route('/uploads/videos/<path:filename>')
def serve_video(filename): return send_from_directory(VIDEO_DIR, filename)

@app.route('/static/detections/<path:filename>')
def serve_detections(filename): return send_from_directory(DETECTIONS_DIR, filename)

# command sending for esp
@app.route('/api/command', methods=['POST'])
def send_command():
    """Endpoint for the UI to send commands to the ESP32 via MQTT."""
    try:
        data = request.json
        command = data.get('command')
        if command:
            # Publish to the exact topic the ESP32 is subscribed to
            mqtt_client.publish("security/command", command)
            return jsonify({"success": True, "message": f"Sent {command} to ESP32"})
        return jsonify({"success": False, "error": "No command provided"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ==========================================
# 6. MAIN SERVER EXECUTION
# ==========================================
if __name__ == '__main__':
    print("\n🚀 STARTING WILDLIFE SERVER (LOCAL ONLY) 🚀\n")
    
    # 1. Boot up the heavy AI models before opening the server
    model_manager.initialize()

    # 2. Start MQTT Client to listen to ESP32 sensors in the background
    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message
    
    try:
        mqtt_client.connect("127.0.0.1", 1883, 60)
        mqtt_client.loop_start() 
    except Exception as e:
        print(f"⚠️ Could not connect to MQTT Broker. Is Mosquitto running? Error: {e}")

    print(f"\n{'='*60}")
    print(f"🌐 LOCAL URL: http://127.0.0.1:5000")
   # print(f"🌐 TEST URL:  http://127.0.0.1:5000/test")
    print(f"{'='*60}\n")
    
    # 3. Start the Flask web server
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)
