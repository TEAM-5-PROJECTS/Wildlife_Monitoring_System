
# 🐾 Wildllife Monitoring System
A fully local, AI-powered wildlife monitoring and IoT security system built for Windows. This project combines **Google's SpeciesNet**  with a real-time **MQTT sensor dashboard** to monitor remote field cameras .

## ✨ Features
* **AI Image & Video Analysis:** Fast, local inference using SpeciesNet to identify wildlife in photos and MP4 videos.
* **GPU Accelerated:** Optimized for local NVIDIA RTX hardware via PyTorch and CUDA.
* **IoT Sensor Dashboard:** Real-time monitoring of field units (Motion, Tilt, Gunshot detection) via an MQTT broker.
* **Instant Telegram Alerts:** Sends instant push notifications with confidence scores and timestamps when target species or security events are detected.

---

## 🛠️ 1. Prerequisites
Before installing, ensure your Windows machine has the following:
* **Python 3.9, 3.10, or 3.11:** [Download Python](https://www.python.org/downloads/) (Make sure to check "Add Python to PATH" during installation).
* **NVIDIA GPU Drivers:** Required for hardware acceleration.
* **Eclipse Mosquitto:** [Download Mosquitto for Windows](https://mosquitto.org/download/). This is the MQTT broker that allows your ESP32 to talk to the Python server. Install it and ensure the "Mosquitto Broker" service is running in Windows Services.

---

## 🚀 2. Project Setup

### Create the Project Structure
Open your terminal (PowerShell) and create your project folder. Run this command inside your new folder to instantly build the required directory structure:
```powershell
mkdir templates, static\detections, uploads\videos, temp_inference\frames, speciesnet_model

```

### Set Up the Virtual Environment

Keep your dependencies isolated by creating a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\activate

```

*(Note: You must run `.\venv\Scripts\activate` every time you open a new terminal to work on this project).*

### Create `requirements.txt`

Create a file named `requirements.txt` in the root folder and paste the following:

```text
# --- GPU Configuration ---
--extra-index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
torch==2.5.1+cu118
torchvision==0.20.1+cu118

# --- Web Server & Database ---
Flask==3.1.2
flask-cors==6.0.2
Flask-SQLAlchemy==3.1.1
Werkzeug==3.1.5

# --- Image & Video Processing ---
opencv-python==4.11.0.86
pillow==12.0.0

# --- AI Models ---
speciesnet==5.0.3

# --- IoT & Notifications ---
paho-mqtt==2.1.0
python-dotenv==1.2.1
requests==2.32.5

```

Install everything by running:

```powershell
pip install -r requirements.txt

```

---

## 🧠 3. Install the AI Model (Crucial Step)

The SpeciesNet AI model is a ~200MB file that must be downloaded manually.

1. Go to Kaggle: [SpeciesNet v4.0.2a PyTorch Model](https://www.kaggle.com/api/v1/models/google/speciesnet/pyTorch/v4.0.2a/1/download)
2. Download the `archive.tar.gz` file and extract it.
3. Open the extracted folder. **Move all of its contents** directly into your project's `speciesnet_model/` folder.
* ⚠️ **Rule:** The file `info.json` MUST be located exactly at `WildlifeProject\speciesnet_model\info.json`. Do not leave it nested inside an "archive" subfolder.



---

## 📱 4. Configure Telegram Alerts

Create a hidden environment file to securely store your API keys.

1. Create a file named exactly **`.env`** in your main project folder.
2. Add your Telegram Bot credentials to the file:

```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

```

---

## 📂 5. Project Directory Structure

Before running the server, verify your project folder looks exactly like this:

```text
WildlifeProject/
│
├── app.py                      # Main backend server & AI logic
├── requirements.txt            # Python dependencies list
├── .env                        # Hidden file for Telegram API keys
├── species_data.db             # SQLite database for detection history
│
├── speciesnet_model/           # The downloaded Kaggle AI model
│   ├── always_crop_... .pt     # PyTorch model weights
│   └── info.json               # Model configuration
│
├── templates/                  # Frontend HTML UI
│   ├── index.html
│   ├── simple_test.html
│   ├── sensor.html
│   └── amb82_dashboard.html
│
├── static/
│   └── detections/             # Where AI-cropped animal images are saved
│
├── uploads/
│   └── videos/                 # Temporary storage for video processing
│
├── temp_inference/
│   └── frames/                 # Temporary frame extraction folder
│
└── venv/                       # Virtual environment (Do not share this folder)

```

---

## 🏃‍♂️ 6. Running the Server

1. Ensure your Mosquitto MQTT broker is running in the background.
2. Open your terminal, activate your virtual environment, and run:
```powershell
python app.py

```


3. **Wait for Warmup:** The first time you launch, it will take 1-3 minutes to load the PyTorch model into your GPU's VRAM. Wait until you see `🌐 LOCAL URL: http://127.0.0.1:5000` in the console.
4. Open your web browser and navigate to:
* **Dashboard:** `http://127.0.0.1:5000/field_unit`
* **Simple Testing UI:** `http://127.0.0.1:5000/test`



---

## 🔧 Troubleshooting

* **Server is running, but the browser says "Site cannot be reached" (HTTP 404/Refused):**
Windows Defender Firewall is likely blocking Python. Open Windows Search -> "Windows Defender Firewall with Advanced Security" -> "Inbound Rules". Delete all rules named `python.exe`. Restart the server and check BOTH "Private" and "Public" networks on the security popup.
* **Model Load Error: `No such file or directory 'info.json'`:**
Your Kaggle model files are extracted into a subfolder. Move them out of the `archive` folder directly into `speciesnet_model`.* **MQTT Connection Error:**
Ensure Mosquitto is installed and the Windows Service is actively running. The script attempts to connect to `127.0.0.1` on port `1883`.


```
