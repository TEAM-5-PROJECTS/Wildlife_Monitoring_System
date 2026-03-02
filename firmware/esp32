#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <math.h>
#include <WiFi.h>
#include <WiFiManager.h>
#include <PubSubClient.h> 
#include <Preferences.h>
#include <driver/i2s.h>
#include <arduinoFFT.h>

// ==========================================
// PIN DEFINITIONS
// ==========================================
const int pirPin = 27;
const int rcwlPin = 17;
const int ledPin = 2;          
const int wifi_on = 19;
const int wifi_off = 32;
const int wake_up = 18;
const int buttonPin = 4;       
const int wificonfig_button = 5;

// --- NEW CONTROL PIN ---
const int controlPin = 26; 

// I2S Microphone
#define I2S_WS 15
#define I2S_SD 33
#define I2S_SCK 14
#define I2S_PORT I2S_NUM_0

// ==========================================
// SYSTEM & HEARTBEAT CONFIG
// ==========================================
#define HEARTBEAT_INTERVAL 60000   // 60 seconds
static unsigned long lastHeartbeat = 0;

// ==========================================
// AUDIO & FFT CONFIG (STEREO ADAPTIVE)
// ==========================================

#define SOFTWARE_GAIN_FACTOR 0.8
#define TRIGGER_AMP_THRESHOLD 5000
#define I2S_SAMPLE_RATE 16000
#define SAMPLES_PER_CHUNK 256

#define MIN_CONSECUTIVE_CHUNKS 8
#define MAX_CONSECUTIVE_CHUNKS 45    // INCREASED from 35 to allow for room echo
#define MAX_EVENT_DURATION 600

#define RATIO_STANDARD 2.0
#define ZCR_STANDARD 40              // LOWERED slightly for safety
#define RATIO_STRICT 4.0             // LOWERED from 13.0 to match your 5.06 ratio
#define ZCR_STRICT 60                // LOWERED from 100 to match your 69 ZCR
#define MAX_LOW_ENERGY_THRESHOLD 5000000 // INCREASED so loud bass doesn't block it

// ==========================================
// GLOBALS & OBJECTS
// ==========================================
Adafruit_MPU6050 mpu;
Preferences preferences;
ArduinoFFT<double> FFT = ArduinoFFT<double>();

// MQTT Client Setup
WiFiClient espClient;
PubSubClient mqttClient(espClient);
const int mqttPort = 1883;
const char* topicEvents = "security/events";       
const char* topicHeartbeat = "security/heartbeat"; 
const char* topicCommand = "security/command"; // --- NEW TOPIC ---

// Audio Buffers
double vReal[SAMPLES_PER_CHUNK];
double vImag[SAMPLES_PER_CHUNK];
int16_t peak_chunk_buffer[SAMPLES_PER_CHUNK];

// State Variables (Motion/Tilt)
float refAccelX, refAccelY, refAccelZ;
float refMag;
float last_angle = 0.0;
float current_angle = 0.0;
int motion_status = 0;
int last_motion = 0;
int tilt_status = 0;

// State Variables (Gunshot - Shared between Cores)
volatile bool gunshotDetected = false;
volatile double gunshotRatio = 0.0;
volatile int gunshotZCR = 0;
bool gunshotDataSent = false;

// Buttons
int lastButtonState = HIGH;
int lastWifiButtonState = HIGH;

// Timing
unsigned long previousMillis = 0;
const long interval = 1000; 
unsigned long lastMQTTReconnectAttempt = 0; 

// --- MOTION HOLD TIMER VARIABLES ---
const unsigned long MOTION_HOLD_TIME_MS = 5000; 
unsigned long lastRawMotionTime = 0;            
bool serverMotionState = false;                 
// -----------------------------------

// --- NEW COOLDOWN VARIABLES ---
const unsigned long EVENT_COOLDOWN_MS = 30000; 
unsigned long lastEventSendTime = 0;     

// Server
char serverUrlBuffer[100];
String serverUrl; 

// Task Handle for Audio
TaskHandle_t AudioTaskHandle;

// --- MANUAL WAKE TRACKERS ---
unsigned long manualWakeTimer = 0;
bool isManualWakeActive = false;

// ==========================================
// FUNCTION PROTOTYPES
// ==========================================
void sendData(bool isGunshotEvent);
void calibrate();
void i2sInit();
void sendHeartbeat();   
void handleHeartbeat(); 
void reconnectMQTT(); 
void mqttCallback(char* topic, byte* payload, unsigned int length); // NEW

// ==========================================
// CORE 0: STEREO AUDIO PROCESSING TASK 
// ==========================================
void AudioProcessingTask(void * parameter) {
  enum State { IDLE, TRIGGERED };
  State currentState = IDLE;
  unsigned long eventStartTime = 0;
  int consecutiveLoudChunks = 0;
  int16_t peak_amplitude_of_event = 0;

  int32_t samples32[SAMPLES_PER_CHUNK * 2]; 
  int16_t samplesLeft[SAMPLES_PER_CHUNK];
  int16_t samplesRight[SAMPLES_PER_CHUNK];
  size_t bytes_read;

  for(;;) {
    i2s_read(I2S_PORT, samples32, sizeof(samples32), &bytes_read, portMAX_DELAY);
    
    if (bytes_read > 0) {
      
      // --- 1. CALCULATE DC OFFSET (MEAN) ---
      int32_t meanL = 0;
      int32_t meanR = 0;
      for (int i = 0; i < (SAMPLES_PER_CHUNK * 2); i += 2) {
        meanL += (samples32[i] >> 16);
        meanR += (samples32[i+1] >> 16);
      }
      meanL /= SAMPLES_PER_CHUNK;
      meanR /= SAMPLES_PER_CHUNK;

      // --- 2. REMOVE OFFSET, APPLY GAIN, FIND PEAKS ---
      int16_t current_peak_L = 0;
      int16_t current_peak_R = 0;
      int sampleIdx = 0;

      for (int i = 0; i < (SAMPLES_PER_CHUNK * 2); i += 2) {
        // Process Left Channel
        int32_t sL = (samples32[i] >> 16) - meanL; // Subtract DC offset
        sL *= SOFTWARE_GAIN_FACTOR;
        if (sL > 32767) sL = 32767;
        if (sL < -32768) sL = -32768;
        samplesLeft[sampleIdx] = (int16_t)sL;
        if (abs(samplesLeft[sampleIdx]) > current_peak_L) current_peak_L = abs(samplesLeft[sampleIdx]);

        // Process Right Channel
        int32_t sR = (samples32[i+1] >> 16) - meanR; // Subtract DC offset
        sR *= SOFTWARE_GAIN_FACTOR;
        if (sR > 32767) sR = 32767;
        if (sR < -32768) sR = -32768;
        samplesRight[sampleIdx] = (int16_t)sR;
        if (abs(samplesRight[sampleIdx]) > current_peak_R) current_peak_R = abs(samplesRight[sampleIdx]);

        sampleIdx++;
      }

      int16_t current_peak = max(current_peak_L, current_peak_R);
      int16_t* loudest_channel_buffer = (current_peak_L > current_peak_R) ? samplesLeft : samplesRight;

      switch (currentState) {
        case IDLE:
          if (current_peak > TRIGGER_AMP_THRESHOLD) {
            currentState = TRIGGERED;
            eventStartTime = millis();
            consecutiveLoudChunks = 1;
            peak_amplitude_of_event = current_peak;
            
            memcpy(peak_chunk_buffer, loudest_channel_buffer, sizeof(samplesLeft));
            Serial.printf("Triggered (Amp: %d)... ", current_peak);
          }
          break;

        case TRIGGERED:
          if (current_peak > (TRIGGER_AMP_THRESHOLD / 2)) {
            consecutiveLoudChunks++;
            if (current_peak > peak_amplitude_of_event) {
              peak_amplitude_of_event = current_peak;
              memcpy(peak_chunk_buffer, loudest_channel_buffer, sizeof(samplesLeft));
            }
          }

          if (millis() - eventStartTime > MAX_EVENT_DURATION) {
            if (consecutiveLoudChunks >= MIN_CONSECUTIVE_CHUNKS && consecutiveLoudChunks <= MAX_CONSECUTIVE_CHUNKS) {
              
              for (int i = 0; i < SAMPLES_PER_CHUNK; i++) {
                vReal[i] = peak_chunk_buffer[i];
                vImag[i] = 0;
              }
              FFT.windowing(vReal, SAMPLES_PER_CHUNK, FFT_WIN_TYP_HAMMING, FFT_FORWARD);
              FFT.compute(vReal, vImag, SAMPLES_PER_CHUNK, FFT_FORWARD);
              FFT.complexToMagnitude(vReal, vImag, SAMPLES_PER_CHUNK);

              double lowEnergy = 0;
              double highEnergy = 0;

              for (int i = 2; i < SAMPLES_PER_CHUNK / 2; i++) {
                double freq = i * 62.5;
                if (freq < 1000) lowEnergy += vReal[i];
                if (freq > 2500) highEnergy += vReal[i];
              }
              if (lowEnergy == 0) lowEnergy = 1; 

              double ratio = highEnergy / lowEnergy;
              
              int peak_zcr = 0;
              int16_t p = 0;
              for(int k=0; k<SAMPLES_PER_CHUNK; k++){
                 if ((peak_chunk_buffer[k] > 0 && p <= 0) || (peak_chunk_buffer[k] < 0 && p >= 0)) peak_zcr++;
                 p = peak_chunk_buffer[k];
              }

              Serial.printf("Done. Dur:%d | Ratio:%.2f | ZCR:%d ", consecutiveLoudChunks, ratio, peak_zcr);

              bool isGunshot = false;
              bool passBass = (lowEnergy < MAX_LOW_ENERGY_THRESHOLD);

              if (consecutiveLoudChunks <= 25) {
                 if (ratio > RATIO_STANDARD && peak_zcr > ZCR_STANDARD && passBass) {
                    isGunshot = true;
                    Serial.print("[Standard Pass]");
                 }
              } else {
                 if ((ratio > RATIO_STRICT || peak_zcr > ZCR_STRICT) && passBass) {
                    isGunshot = true;
                    Serial.print("[Strict Pass]");
                 } else {
                    Serial.print("[Strict Fail: Likely Thunder]");
                 }
              }

              if (isGunshot) {
                digitalWrite(wake_up, HIGH); 
                digitalWrite(ledPin, HIGH);
                
                gunshotRatio = ratio;
                gunshotZCR = peak_zcr;
                gunshotDetected = true; 
                
                Serial.println(" -> >>> STEREO GUNSHOT CONFIRMED <<<");
              } else {
                 Serial.println(" -> REJECTED");
              }

            } else {
              Serial.printf("Done. REJECTED: Duration (%d) out of bounds.\n", consecutiveLoudChunks);
            }
            
            currentState = IDLE;
          }
          break;
      }
    }
  }
}

// ==========================================
// HELPER: MQTT Callback (Receives Messages)
// ==========================================
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  
  Serial.print("MQTT Message arrived on topic: ");
  Serial.print(topic);
  Serial.print(". Message: ");
  Serial.println(message);

  // Check if the message is on our command topic
  if (String(topic) == topicCommand) {
    message.toUpperCase(); // Make it case-insensitive
    
    // Turn PIN 23 HIGH if message is ON, 1, or HIGH
    if (message == "ON" || message == "1" || message == "HIGH") {
      digitalWrite(controlPin, HIGH);
      Serial.println("Action: Pin 23 turned HIGH");
    } 
    // Turn PIN 23 LOW if message is OFF, 0, or LOW
    else if (message == "OFF" || message == "0" || message == "LOW") {
      digitalWrite(controlPin, LOW);
      Serial.println("Action: Pin 23 turned LOW");
    }
    // --- NON-BLOCKING WAKE COMMAND ---
    else if (message == "WAKE") {
      Serial.println("Action: Manual Wake Pulse Started");
      isManualWakeActive = true;
      manualWakeTimer = millis(); // Start the stopwatch
      digitalWrite(wake_up, HIGH);
      // NEW: Send a response back to the server!
      mqttClient.publish(topicEvents, "{\"manual_wake\":1}");
    }
  }
}

// ==========================================
// HELPER: Reconnect MQTT (Non-Blocking)
// ==========================================
void reconnectMQTT() {
  if (WiFi.status() != WL_CONNECTED) return;
  
  if (!mqttClient.connected()) {
    if (millis() - lastMQTTReconnectAttempt > 5000) {
      lastMQTTReconnectAttempt = millis();
      Serial.print("Attempting MQTT connection...");
      
      String clientId = "ESP32Security-";
      clientId += String(random(0xffff), HEX);
      
      if (mqttClient.connect(clientId.c_str())) {
        Serial.println("connected");
        // --- SUBSCRIBE TO COMMANDS AFTER CONNECTING ---
        mqttClient.subscribe(topicCommand);
        Serial.println("Subscribed to command topic");
      } else {
        Serial.print("failed, rc=");
        Serial.print(mqttClient.state());
        Serial.println(" try again in 5 seconds");
      }
    }
  }
}

// ==========================================
// HELPER: Send Data (Events) -> MQTT
// ==========================================
void sendData(bool isGunshotEvent) {
  // 1. LED check: ONLY cares about WiFi status
  if (WiFi.status() == WL_CONNECTED) {
     digitalWrite(wifi_on, HIGH);
     digitalWrite(wifi_off, LOW); // Green ON, Red OFF
  } else {
     digitalWrite(wifi_on, LOW);
     digitalWrite(wifi_off, HIGH); // Green OFF, Red ON
     Serial.println("WiFi disconnected. Skipping send.");
     return; // Stop running the function right here
  }

  // 2. Data check: ONLY try to send if MQTT is connected
  if (mqttClient.connected()) {
     char payload[128];
     int gsFlag = isGunshotEvent ? 1 : 0;
     double r = isGunshotEvent ? gunshotRatio : 0.0;
     int z = isGunshotEvent ? gunshotZCR : 0;

     snprintf(payload, sizeof(payload), 
              "{\"motion\":%d,\"tilt\":%.2f,\"gunshot\":%d,\"ratio\":%.2f,\"zcr\":%d}", 
              motion_status, current_angle, gsFlag, r, z);
     
     bool success = mqttClient.publish(topicEvents, payload);
     
     if (success) {
        Serial.println("Sent Event to server via MQTT");
     } else {
        Serial.println("MQTT Publish Failed");
     }
  } else {
     Serial.println("MQTT disconnected. Skipping send.");
  }
  
  if(isGunshotEvent) digitalWrite(ledPin, LOW); 
}

// ==========================================
// HELPER: Send Heartbeat -> MQTT
// ==========================================
void sendHeartbeat() {
    if (WiFi.status() != WL_CONNECTED || !mqttClient.connected()) return;

    uint32_t freeHeap = ESP.getFreeHeap();
    uint32_t minHeap  = ESP.getMinFreeHeap();
    int8_t rssi       = WiFi.RSSI();
    float temperature = temperatureRead(); 

    char payload[180];
    snprintf(payload, sizeof(payload),
        "{"
        "\"alive\":1,"
        "\"uptime\":%lu,"
        "\"free_heap\":%u,"
        "\"min_heap\":%u,"
        "\"temp\":%.1f,"
        "\"rssi\":%d"
        "}",
        millis(), freeHeap, minHeap, temperature, rssi
    );

    mqttClient.publish(topicHeartbeat, payload);
}

void handleHeartbeat() {
    if (millis() - lastHeartbeat >= HEARTBEAT_INTERVAL) {
        lastHeartbeat = millis();
        sendHeartbeat();
    }
}

// ==========================================
// HELPER: Calibration
// ==========================================
void calibrate() {
  digitalWrite(wifi_on, 1);
  digitalWrite(wifi_off, 1);
  const int samples = 50;
  float sumX = 0, sumY = 0, sumZ = 0;
  sensors_event_t a, g, temp;

  Serial.println("Calibrating... keep sensor steady");
  delay(1000); 

  for (int i = 0; i < samples; i++) {
    mpu.getEvent(&a, &g, &temp);
    sumX += a.acceleration.x;
    sumY += a.acceleration.y;
    sumZ += a.acceleration.z;
    delay(20);
  }

  refAccelX = sumX / samples;
  refAccelY = sumY / samples;
  refAccelZ = sumZ / samples;
  refMag = sqrt(refAccelX*refAccelX + refAccelY*refAccelY + refAccelZ*refAccelZ);

  Serial.println("New reference saved!");
  if (WiFi.status() == WL_CONNECTED) {
     digitalWrite(wifi_on, HIGH);
     digitalWrite(wifi_off, LOW);
  } else {
     digitalWrite(wifi_on, LOW);
     digitalWrite(wifi_off, HIGH);
  }
}

// ==========================================
// SETUP
// ==========================================
void setup() {
  Serial.begin(115200);

  // Pin Modes
  pinMode(pirPin, INPUT);
  pinMode(rcwlPin, INPUT);
  pinMode(ledPin, OUTPUT);
  pinMode(wifi_on, OUTPUT);
  pinMode(wifi_off, OUTPUT);
  pinMode(wake_up, OUTPUT);
  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(wificonfig_button, INPUT_PULLUP);
  
  // --- Initialize the new control pin ---
  pinMode(controlPin, OUTPUT);
  digitalWrite(controlPin, LOW); // Start with it turned off

  digitalWrite(ledPin, LOW);

  // --- I2S Setup ---
  i2sInit();

  // --- MPU6050 Setup ---
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) { delay(10); }
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_16_G);
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  delay(100);
  
  calibrate(); 

  // --- Preferences & WiFi ---
  preferences.begin("my-app", false);
  
  String storedUrl = preferences.getString("server_url", "192.168.1.100"); 
  
  storedUrl.toCharArray(serverUrlBuffer, 100);
  serverUrl = storedUrl;
  Serial.print("Loaded MQTT Broker IP: "); Serial.println(serverUrl);

  WiFiManager wifiManager;
  WiFiManagerParameter custom_server_url("server", "MQTT Broker IP", serverUrlBuffer, 100);
  wifiManager.addParameter(&custom_server_url);

  if (!wifiManager.autoConnect("ESP32-Security")) {
    Serial.println("Failed to connect. Restarting...");
    Serial.flush();
    delay(500);
    digitalWrite(wifi_off, 1);
    digitalWrite(wifi_on, 0);
    ESP.restart(); // SW_RESET triggered here if WiFi fails!
  } else {
    Serial.println("Connected to WiFi");
    digitalWrite(wifi_on, 1);
    digitalWrite(wifi_off, 0);
  }

  String newUrl = custom_server_url.getValue();
  if (newUrl != storedUrl) {
    preferences.putString("server_url", newUrl);
    serverUrl = newUrl;
  }

  // --- Initialize MQTT ---
  mqttClient.setServer(serverUrlBuffer, mqttPort);
  mqttClient.setCallback(mqttCallback); // --- REGISTER THE CALLBACK HERE ---

  // --- Start Audio Task AFTER everything else is ready ---
  xTaskCreatePinnedToCore(
    AudioProcessingTask,   
    "AudioTask",           
    10000,                 
    NULL,                  
    1,                     
    &AudioTaskHandle,      
    0                      
  );
  Serial.println("Audio Task Started on Core 0");
}

// ==========================================
// LOOP (Core 1)
// ==========================================
void loop() {
  // ---- 0. MAINTAIN MQTT CONNECTION ----
  if (!mqttClient.connected()) {
    reconnectMQTT();
  } else {
    mqttClient.loop(); // This line is what actually checks for incoming messages!
  }
  // --- NEW: CHECK IF PULSE IS DONE (100ms) ---
  if (isManualWakeActive && (millis() - manualWakeTimer >= 100)) {
      isManualWakeActive = false;
      digitalWrite(wake_up, LOW); // Turn it off after 100ms
  }
  // ---- 1. SYSTEM HEARTBEAT ----
  handleHeartbeat();

  // ---- 2. GUNSHOT EVENT (High Priority) ----
  if (gunshotDetected && !gunshotDataSent) {
      sendData(true); 
      gunshotDataSent = true; 
  }

  // ---- 3. FAST LOOP (Buttons) ----
  int readingWifi = digitalRead(wificonfig_button);
  if (readingWifi == LOW && lastWifiButtonState == HIGH) {
      delay(50); 
      if(digitalRead(wificonfig_button) == LOW) {
        Serial.println("Starting WiFi config...");
        digitalWrite(wifi_on, 0);
        digitalWrite(wifi_off, 1);
        
        WiFiManager wifiManager;
        wifiManager.setBreakAfterConfig(true);
        serverUrl.toCharArray(serverUrlBuffer, 100);
        WiFiManagerParameter custom_server_url("server", "MQTT Broker IP", serverUrlBuffer, 100);
        wifiManager.addParameter(&custom_server_url);
        wifiManager.startConfigPortal("ESP32-Security");
        
        String newUrl = custom_server_url.getValue();
        preferences.putString("server_url", newUrl);
        preferences.end();
        ESP.restart();
      }
  }
  lastWifiButtonState = readingWifi;

  int readingCal = digitalRead(buttonPin);
  if (readingCal == LOW && lastButtonState == HIGH) {
     delay(50);
     if(digitalRead(buttonPin) == LOW) calibrate();
  }
  lastButtonState = readingCal;

  // ---- 4. SLOW LOOP (Sensors 1Hz) ----
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis; 
  // --- REAL-TIME LED STATUS CHECK (WIFI ONLY) ---
    if (WiFi.status() == WL_CONNECTED) {
        digitalWrite(wifi_on, HIGH);
        digitalWrite(wifi_off, LOW);
    } else {
        digitalWrite(wifi_on, LOW);
        digitalWrite(wifi_off, HIGH);
    }
    // ----------------------------------------------
    int motion1 = digitalRead(pirPin);
    int motion2 = digitalRead(rcwlPin);
    
    Serial.print("pir: "); Serial.println(motion1);
    Serial.print("rcwl: "); Serial.println(motion2);
    
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    float curX = a.acceleration.x;
    float curY = a.acceleration.y;
    float curZ = a.acceleration.z;
    float dot = curX * refAccelX + curY * refAccelY + curZ * refAccelZ;
    float magCur = sqrt(curX*curX + curY*curY + curZ*curZ);
    
    current_angle = 0.0;
    if (magCur * refMag != 0) {
      float cosine = dot / (magCur * refMag);
      if (cosine > 1.0) cosine = 1.0;
      if (cosine < -1.0) cosine = -1.0;
      current_angle = acos(cosine) * 180.0 / PI;
    }

    Serial.print("Tilt Angle: "); Serial.println(current_angle);

    tilt_status = 0;
    if(last_angle > 30 && current_angle < 30) tilt_status = 1;
    else if(current_angle > 30 && abs(current_angle - last_angle) > 2.0) tilt_status = 1;

    last_angle = current_angle;

    // --- OCCUPANCY HOLD TIMER LOGIC ---
    int raw_motion = (motion1 == HIGH && motion2 == HIGH);
    bool trigger_send = false;

    if (tilt_status == 1) {
        trigger_send = true;
        Serial.println("Tilt detected!");
    }

    if (raw_motion == 1) {
        lastRawMotionTime = currentMillis; 
        if (serverMotionState == false) {
            serverMotionState = true;
            trigger_send = true;
            Serial.println("Motion started. Alerting server.");
        }
    } else {
        if (serverMotionState == true && (currentMillis - lastRawMotionTime >= MOTION_HOLD_TIME_MS)) {
            serverMotionState = false; 
            trigger_send = true;
            Serial.println("Motion definitely stopped. Alerting server.");
        }
    }

    motion_status = serverMotionState ? 1 : 0; 
    bool anyEventActive = (motion_status == 1) || (gunshotDetected == true);

    // Only let sensors control the pin if a manual pulse isn't happening
    if (!isManualWakeActive) {
        digitalWrite(wake_up, anyEventActive ? HIGH : LOW);
    }

    if (gunshotDetected) {
       gunshotDetected = false; 
       gunshotDataSent = false;
    }

    if (trigger_send) {
        sendData(false);
    }
  } 

  // --- ADDED TO PREVENT WATCHDOG CRASHES ---
  delay(10); 
}

// ==========================================
// I2S INIT
// ==========================================
void i2sInit() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = I2S_SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,
    .dma_buf_len = SAMPLES_PER_CHUNK,
    .use_apll = false
  };
  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = -1,
    .data_in_num = I2S_SD
  };
  i2s_set_pin(I2S_PORT, &pin_config);
}
