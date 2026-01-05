# ESP32 IoT Smart Security System

A real-time IoT-based smart security system built on ESP32 that detects motion, tampering (tilt), and gunshot-like acoustic events using digital signal processing (DSP).  
The system sends instant alerts and periodic health heartbeats to a cloud server over secure HTTPS.

## Project Overview

This project implements a multi-sensor security node designed for surveillance and perimeter protection use cases.  
It combines audio signal analysis, motion detection, and orientation monitoring to reliably identify suspicious events while minimizing false positives.

### Key Capabilities
- Gunshot detection using FFT and Zero Crossing Rate analysis
- Motion detection using PIR and microwave radar sensors
- Tamper detection via accelerometer-based tilt monitoring
- Secure cloud communication using HTTPS REST APIs
- System heartbeat for health monitoring (RSSI, memory, uptime)
- Dual-core task separation for real-time performance

## System Architecture

**Core 0 (ESP32)**  
Dedicated to high-speed audio sampling and gunshot detection using I2S and FFT.

**Core 1 (ESP32)**  
Handles motion sensing, tilt calculation, WiFi connectivity, event reporting, and heartbeat transmission.

## Gunshot Detection Logic

Gunshot events are identified using a multi-stage DSP pipeline:

1. **Amplitude Trigger**  
   Detects impulsive high-energy sound events.

2. **FFT (Frequency Domain Analysis)**  
   Separates low-frequency and high-frequency energy bands and computes energy ratios to distinguish gunshots from ambient noise.

3. **Zero Crossing Rate (ZCR)**  
   Measures signal impulsiveness to filter non-ballistic sounds.

4. **Adaptive Thresholding**  
   Applies strict or standard thresholds based on event duration to reduce false positives such as thunder.

## Hardware Components

| Component | Description |
|---------|------------|
| ESP32 | Dual-core microcontroller |
| I2S MEMS Microphone | High-speed audio capture |
| MPU6050 | Accelerometer and gyroscope for tilt detection |
| PIR Sensor | Passive infrared motion detection |
| RCWL-0516 | Microwave radar motion sensor |
| LEDs | System and event status indicators |
| Push Buttons | Calibration and WiFi configuration |
## Software Stack

- Programming Language: Embedded C++ (Arduino Framework)
- RTOS: ESP32 FreeRTOS
- Signal Processing: FFT, Zero Crossing Rate
- Backend: Flask (Cloud hosted)

### Libraries Used
- ArduinoFFT
- Adafruit MPU6050
- WiFiManager
- Preferences
- ESP32 I2S Driver

## Cloud Communication

The device sends structured JSON payloads to a backend server.

### Event Payload
```json
{
  "motion": 1,
  "tilt": 32.4,
  "gunshot": 1,
  "ratio": 4.82,
  "zcr": 96
}
