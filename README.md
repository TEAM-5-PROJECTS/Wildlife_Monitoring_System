
# Intelligent Wildlife Monitoring System

This project integrates Google’s SpeciesNet for wildlife classification with a  weapon detection model (ONNX Runtime) and a real-time MQTT sensor dashboard to monitor remote field cameras, detect threats, and trigger instant Telegram alerts.

## Hardware Setup

<table align="center">
  <tr>
    <td align="center">
      <img src="assets/main.jpeg" width="300" alt="Field Unit Setup"/><br>
      <em>1. Main unit with sensors and camera</em>
    </td>
    <td align="center">
      <img src="assets/auxnode.jpeg" width="350" alt="Sensor Node"/><br>
      <em>2. Auxiliary unit with sensors only</em>
    </td>
  </tr>
</table>

## Hardware Components
### Main Field Unit
- ESP32 Development Board
- AMB82 Mini Camera Module
- PIR Motion Sensor
- HLK-LD105 Microwave Motion Sensor
- MPU6050 Accelerometer/Gyroscope
- I2S Digital Microphone(INMP441)
- 2S Li-ion Battery Pack
- Battery Voltage Divider (47kΩ + 22kΩ resistors)
- Status LEDs
- Push Buttons (Calibration + Wi-Fi Config)

### Auxiliary Sensor Node
- ESP32 Development Board
- PIR Motion Sensor
- RCWL Microwave Motion Sensor
- I2S Digital Microphone
- DHT11 Temperature & Humidity Sensor
- 2S Li-ion Battery Pack
- Battery Voltage Divider (47kΩ + 22kΩ resistors)
- Status LEDs
- Wi-Fi Config Button

### Camera Unit
- AMB82 Mini AI Camera Board
- MicroSD Card
- Wi-Fi Connectivity
- Wake Trigger Interface
<h3>Main Field Unit (ESP32)</h3>
<table>
  <thead>
    <tr>
      <th>Component</th>
      <th>GPIO Pin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>PIR Motion Sensor</td>
      <td>GPIO 27</td>
    </tr>
    <tr>
      <td>HLK-LD105 Motion Sensor</td>
      <td>GPIO 17</td>
    </tr>
    <tr>
      <td>MPU6050 (I2C SDA/SCL)</td>
      <td>Default I2C Pins</td>
    </tr>
    <tr>
      <td>I2S Microphone WS</td>
      <td>GPIO 15</td>
    </tr>
    <tr>
      <td>I2S Microphone SD</td>
      <td>GPIO 33</td>
    </tr>
    <tr>
      <td>I2S Microphone SCK</td>
      <td>GPIO 14</td>
    </tr>
    <tr>
      <td>Battery Voltage Monitor</td>
      <td>GPIO 34</td>
    </tr>
    <tr>
      <td>Status LED</td>
      <td>GPIO 2</td>
    </tr>
    <tr>
      <td>Wi-Fi Connected LED</td>
      <td>GPIO 19</td>
    </tr>
    <tr>
      <td>Wi-Fi Disconnected LED</td>
      <td>GPIO 32</td>
    </tr>
    <tr>
      <td>Wake Trigger Output</td>
      <td>GPIO 18</td>
    </tr>
    <tr>
      <td>Calibration Button</td>
      <td>GPIO 4</td>
    </tr>
    <tr>
      <td>Wi-Fi Config Button</td>
      <td>GPIO 5</td>
    </tr>
    <tr>
      <td>Control Output</td>
      <td>GPIO 26</td>
    </tr>
  </tbody>
</table>

<hr>

<h3>Auxiliary Sensor Node (ESP32)</h3>
<table>
  <thead>
    <tr>
      <th>Component</th>
      <th>GPIO Pin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>PIR Motion Sensor</td>
      <td>GPIO 27</td>
    </tr>
    <tr>
      <td>RCWL Motion Sensor</td>
      <td>GPIO 25</td>
    </tr>
    <tr>
      <td>DHT11 Sensor</td>
      <td>GPIO 13</td>
    </tr>
    <tr>
      <td>I2S Microphone WS</td>
      <td>GPIO 15</td>
    </tr>
    <tr>
      <td>I2S Microphone SD</td>
      <td>GPIO 33</td>
    </tr>
    <tr>
      <td>I2S Microphone SCK</td>
      <td>GPIO 14</td>
    </tr>
    <tr>
      <td>Battery Voltage Monitor</td>
      <td>GPIO 34</td>
    </tr>
    <tr>
      <td>Wi-Fi Status LED</td>
      <td>GPIO 2</td>
    </tr>
    <tr>
      <td>Gunshot Detection LED</td>
      <td>GPIO 4</td>
    </tr>
    <tr>
      <td>Wake Trigger Output</td>
      <td>GPIO 18</td>
    </tr>
    <tr>
      <td>Wi-Fi Config Button</td>
      <td>GPIO 5</td>
    </tr>
  </tbody>
</table>

<hr>

<h3>AMB82 Mini Camera Unit</h3>
<table>
  <thead>
    <tr>
      <th>Component</th>
      <th>Connection</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Wake Trigger Input</td>
      <td>GPIO 21</td>
    </tr>
    <tr>
      <td>MicroSD Card</td>
      <td>Onboard</td>
    </tr>
    <tr>
      <td>Wi-Fi</td>
      <td>Internal</td>
    </tr>
    <tr>
      <td>Camera + Audio</td>
      <td>Onboard</td>
    </tr>
  </tbody>
</table>

## Software Interface

<table align="center">
  <tr>
    <td align="center">
      <img src="assets/mainweb.png" width="350" alt="Software Interface 1"/><br>
      <em>Main menue</em>
    </td>
    <td align="center">
      <img src="assets/sensor.png" width="350" alt="Software Interface 2"/><br>
      <em>Sensor  dashboard</em>
    </td>
    <td align="center">
      <img src="assets/detection1.png" width="350" alt="Software Interface 3"/><br>
      <em>Detections </em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/detection2.png" width="350" alt="Software Interface 4"/><br>
      <em>Detections</em>
    </td>
    <td align="center">
      <img src="assets/analytics.png" width="350" alt="Software Interface 5"/><br>
      <em>Analytics</em>
    </td>
    <td align="center">
      <img src="assets/settings.png" width="350" alt="Software Interface 6"/><br>
      <em>Custom Settings </em>
    </td>
  </tr>
</table>

## Features
* **AI Image & Video Analysis:** Real-time local inference using  SpeciesNet for wildlife classification and weapon detection for identifying armed threats.
* **Hybrid GPU–CPU Architecture:** Wildlife classification runs on NVIDIA RTX GPUs via PyTorch (CUDA), while weapon detection operates on the CPU using ONNX Runtime for efficient asymmetric processing.
* **IoT Sensor Dashboard:** Real-time monitoring of remote field units (Motion, Tilt, Gunshot, and System Status) through an MQTT broker with live device health tracking.
* **Telegram Alerts:** Automated push notifications with species name, confidence score, timestamps, and threat classification , including cooldown logic to prevent alert spam.

---

## 1. Prerequisites
Before installing, ensure your Windows machine has the following:
* **Python 3.10.11:** [Download Python](https://www.python.org/downloads/) (Make sure to check "Add Python to PATH" during installation).
* **NVIDIA GPU Drivers:** Required for hardware acceleration.
* **Eclipse Mosquitto:** [Download Mosquitto for Windows](https://mosquitto.org/download/). This is the MQTT broker that allows your ESP32 to talk to the Python server. Install it and ensure the "Mosquitto Broker" service is running in Windows Services.
---
*(Note: You must add  "listener 1883 0.0.0.0" "allow_anonymous true" at the end of mosquitto.conf file).*

## 2. Project Setup

### Set Up the Virtual Environment

Keep your dependencies isolated by creating a virtual environment:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\activate

```

*(Note: You must run `.\venv\Scripts\activate` every time you open a new terminal to work on this project).*

### Install `CUDA  11.8  and requirements.txt `

NVIDIA GPU (CUDA 11.8)
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

```
Install everything by running:

```powershell
pip install -r requirements.txt
```

---

## 3. Install the AI Model (Crucial Step)

The SpeciesNet AI model is a ~200MB file that must be downloaded.

1. Go to Kaggle: [SpeciesNet v4.0.2a PyTorch Model](https://www.kaggle.com/api/v1/models/google/speciesnet/pyTorch/v4.0.2a/1/download)
2. Download the `archive.tar.gz` file and extract it.
3. Open the extracted folder. **Move all of its contents** directly into your project's `speciesnet_model/` folder.
* ⚠️ **Rule:** The file `info.json` MUST be located exactly at `WildlifeProject\speciesnet_model\info.json`. Do not leave it nested inside an "archive" subfolder.



---

## 4. Environment Variable Setup (.env Configuration)

Create a hidden environment file to securely store your API keys.

1. Create a file named exactly **`.env`** in your main project folder.
2. Add your credentials to the file:

```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
ROBOFLOW_API_KEY=your_api_key
```

---

## 5. Project Directory Structure

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
│   └── info.json 
|              # Model configuration
├── weapon_model.onnx           # Weapon Detection Model(yolov26)
│   
│   
│
├── templates/                  # Frontend HTML UI
│   ├── index.html
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
└── venv/                       # Virtual environment 

```

---

## 6. Running the Server

1. Ensure your Mosquitto MQTT broker is running in the background at port:1883 and host:127.0.0.1
2. Open your terminal, activate your virtual environment, and run:
```powershell
python app.py

```


3. **Wait for Warmup:** The first time you launch, it will take 1-3 minutes to load the PyTorch model into your GPU's VRAM. Wait until you see `🌐 LOCAL URL: http://127.0.0.1:5000` in the console.
4. Open your web browser and navigate to:
* **Dashboard:** `http://127.0.0.1:5000/field_unit`



---
## Project Demo

🎥 **Software Demo Video:** [Watch  Demo](https://youtu.be/htrTLkHLNSU)<br>
🎥 **Hardware Demo Video:** [Watch  Demo](https://youtube.com/shorts/Bd568EzYMTM?feature=share)

## Troubleshooting

* **Server is running, but the browser says "Site cannot be reached" (HTTP 404/Refused):**
Windows Defender Firewall is likely blocking Python. Restart the server and check BOTH "Private" and "Public" networks on the Windows defender .
* **Model Load Error: `No such file or directory 'info.json'`:**
Your Kaggle model files are extracted into a subfolder. Move them out of the `archive` folder directly into `speciesnet_model`.* **MQTT Connection Error:**
Ensure Mosquitto is installed and the Windows Service is actively running. The script attempts to connect to `127.0.0.1` on port `1883`.


```
