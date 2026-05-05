#include <Wire.h>
#include <math.h>
#include <WiFi.h>
#include <WiFiManager.h>
#include <PubSubClient.h> 
#include <Preferences.h>
#include <driver/i2s.h>
#include <arduinoFFT.h>
#include <Ticker.h> 
#include <DHT.h>

// --- DHT11 CONFIG ---
#define DHTPIN 13       
#define DHTTYPE DHT11 
DHT dht(DHTPIN, DHTTYPE);

// ==========================================
// PIN DEFINITIONS
// ==========================================
const int pirPin = 27;
const int rcwlPin = 25;
const int wake_up = 18;
const int wificonfig_button = 5;

// --- LED PINS ---
const int wifiLedPin = 2;      
const int gunshotLedPin = 4;   

// I2S Microphone (Mono)
#define I2S_WS 15
#define I2S_SD 33
#define I2S_SCK 14
#define I2S_PORT I2S_NUM_0

// ==========================================
// BATTERY CONFIG (2S Li-ion)
// ==========================================
#define BATT_PIN 34          // ADC1 pin for Wi-Fi compatibility
#define R1_VAL 47000.0       // 47k Ohm
#define R2_VAL 22000.0       // 22k Ohm
#define V_REF 3.3            // ESP32 standard reference voltage
#define BATT_CALIBRATION 1.141 // Calibration applied from earlier test

// ==========================================
// SYSTEM & HEARTBEAT CONFIG
// ==========================================
#define HEARTBEAT_INTERVAL 60000   // 60 seconds
static unsigned long lastHeartbeat = 0;

// ==========================================
// AUDIO & FFT CONFIG (MONO ADAPTIVE)
// ==========================================
#define SOFTWARE_GAIN_FACTOR 0.8
#define TRIGGER_AMP_THRESHOLD 4000
#define I2S_SAMPLE_RATE 16000
#define SAMPLES_PER_CHUNK 256

#define MIN_CONSECUTIVE_CHUNKS 8
#define MAX_CONSECUTIVE_CHUNKS 45    
#define MAX_EVENT_DURATION 600

// --- NEW BOUNDING BOX THRESHOLDS ---
#define RATIO_MIN 3.5        // Minimum high-to-low energy ratio
#define RATIO_MAX 14.0       // NEW: Maximum ratio (rejects pure high-frequency snaps/claps)
#define ZCR_MIN 55           // Minimum zero crossings (rejects pure bass thuds)
#define ZCR_MAX 73           // LOWERED: Maximum zero crossings (real gunshots maxed out at 72)
#define MAX_LOW_ENERGY_THRESHOLD 5000000
// ==========================================
// GLOBALS & OBJECTS
// ==========================================
Preferences preferences;
ArduinoFFT<double> FFT = ArduinoFFT<double>();
Ticker ticker; 

// MQTT Client Setup
WiFiClient espClient;
PubSubClient mqttClient(espClient);
const int mqttPort = 1883;
const char* topicEvents = "security/node1/events";       
const char* topicHeartbeat = "security/node1/heartbeat";

// Audio Buffers
double vReal[SAMPLES_PER_CHUNK];
double vImag[SAMPLES_PER_CHUNK];
int16_t peak_chunk_buffer[SAMPLES_PER_CHUNK];

// State Variables (Motion)
int motion_status = 0;

// State Variables (Gunshot - Shared between Cores)
volatile bool gunshotDetected = false;
volatile double gunshotRatio = 0.0;
volatile int gunshotZCR = 0;
bool gunshotDataSent = false;

// Buttons
int lastWifiButtonState = HIGH;

// Timing
unsigned long previousMillis = 0;
const long interval = 1000; 
unsigned long lastMQTTReconnectAttempt = 0; 

// --- MOTION HOLD TIMER VARIABLES ---
const unsigned long MOTION_HOLD_TIME_MS = 5000; 
unsigned long lastRawMotionTime = 0;            
bool serverMotionState = false;                 

// Server
char serverUrlBuffer[100];
String serverUrl; 

// Task Handle for Audio
TaskHandle_t AudioTaskHandle;

// ==========================================
// FUNCTION PROTOTYPES
// ==========================================
void sendData(bool isGunshotEvent);
void i2sInit();
void sendHeartbeat();   
void handleHeartbeat(); 
void reconnectMQTT(); 

// ==========================================
// HELPER: TICKER CALLBACKS FOR WIFI LED
// ==========================================
void tickLed() {
  int state = digitalRead(wifiLedPin);
  digitalWrite(wifiLedPin, !state);
}

void configModeCallback(WiFiManager *myWiFiManager) {
  Serial.println("Entered config mode");
  Serial.println(WiFi.softAPIP());
  ticker.attach(0.2, tickLed); 
}

// ==========================================
// CORE 0: HYBRID STEREO-TO-MONO AUDIO TASK 
// ==========================================
void AudioProcessingTask(void * parameter) {
  enum State { IDLE, TRIGGERED };
  State currentState = IDLE;
  unsigned long eventStartTime = 0;
  int consecutiveLoudChunks = 0;
  int16_t peak_amplitude_of_event = 0;

  // FIX 1: Doubled buffer size for Stereo data
  int32_t samples32[SAMPLES_PER_CHUNK * 2]; 
  int16_t samplesMono[SAMPLES_PER_CHUNK];
  size_t bytes_read;

  for(;;) {
    i2s_read(I2S_PORT, samples32, sizeof(samples32), &bytes_read, portMAX_DELAY);
    
    if (bytes_read > 0) {
      
      int32_t mean = 0;

      // FIX 2: Convert Stereo to Mono and apply `>> 14` volume boost
      for (int i = 0; i < SAMPLES_PER_CHUNK; i++) {
        int32_t left = samples32[i * 2] >> 14;
        int32_t right = samples32[i * 2 + 1] >> 14;

        int16_t mixedMono = (int16_t)(left + right);
        
        samplesMono[i] = mixedMono;
        mean += mixedMono;
      }
      mean /= SAMPLES_PER_CHUNK;

      int16_t current_peak = 0;

      for (int i = 0; i < SAMPLES_PER_CHUNK; i++) {
        int32_t s = samplesMono[i] - mean; 
        s *= SOFTWARE_GAIN_FACTOR;
        
        if (s > 32767) s = 32767;
        if (s < -32768) s = -32768;
        
        samplesMono[i] = (int16_t)s;
        if (abs(samplesMono[i]) > current_peak) current_peak = abs(samplesMono[i]);
      }

      switch (currentState) {
        case IDLE:
          if (current_peak > TRIGGER_AMP_THRESHOLD) {
            currentState = TRIGGERED;
            eventStartTime = millis();
            consecutiveLoudChunks = 1;
            peak_amplitude_of_event = current_peak;
            
            memcpy(peak_chunk_buffer, samplesMono, sizeof(samplesMono));
            Serial.printf("\n[AUDIO] Triggered (Amp: %d)... ", current_peak);
          }
          break;

        case TRIGGERED:
          if (current_peak > (TRIGGER_AMP_THRESHOLD / 2)) {
            consecutiveLoudChunks++;
            if (current_peak > peak_amplitude_of_event) {
              peak_amplitude_of_event = current_peak;
              memcpy(peak_chunk_buffer, samplesMono, sizeof(samplesMono));
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

// 1. Check if ZCR falls perfectly inside the Gunshot Envelope (55 to 73)
              bool passZCR = (peak_zcr >= ZCR_MIN && peak_zcr <= ZCR_MAX);
              
              // 2. Check if the energy ratio implies a sharp crack, but NOT a pure snap
              bool passRatio = (ratio >= RATIO_MIN && ratio <= RATIO_MAX);

              // 3. ALL conditions must be met (AND logic, no ORs)
              if (passZCR && passRatio && passBass) {
                  isGunshot = true;
                  Serial.print("[Gunshot Profile Matched]");
              } else {
                  // Optional: Print exactly why it failed for easier debugging
                  Serial.print("[Rejected: ");
                  if (!passZCR) Serial.print("ZCR out of bounds ");
                  if (!passRatio) Serial.print("Ratio out of bounds ");
                  if (!passBass) Serial.print("Failed Bass Test ");
                  Serial.print("]");
              }

              if (isGunshot) {
                digitalWrite(wake_up, HIGH); 
                digitalWrite(gunshotLedPin, HIGH);
                
                gunshotRatio = ratio;
                gunshotZCR = peak_zcr;
                gunshotDetected = true; 
                
                Serial.println(" -> >>> MONO GUNSHOT CONFIRMED <<<");
              } else {
                 Serial.println("");
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
// HELPER: Reconnect MQTT
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
  if (WiFi.status() != WL_CONNECTED) return;

  if (mqttClient.connected()) {
     char payload[200];
     int gsFlag = isGunshotEvent ? 1 : 0;
     double r = isGunshotEvent ? gunshotRatio : 0.0;
     int z = isGunshotEvent ? gunshotZCR : 0;

     snprintf(payload, sizeof(payload), 
              "{\"source\":\"node1\",\"motion\":%d,\"gunshot\":%d,\"ratio\":%.2f,\"zcr\":%d}", 
              motion_status, gsFlag, r, z);
     
     if(mqttClient.publish(topicEvents, payload)) {
         Serial.println("[MQTT] Event data published successfully.");
     } else {
         Serial.println("[MQTT] Event data publish FAILED.");
     }
  }
  
  if(isGunshotEvent) digitalWrite(gunshotLedPin, LOW); 
}

// ==========================================
// HELPER: Send Heartbeat -> MQTT
// ==========================================
void sendHeartbeat() {
    if (WiFi.status() != WL_CONNECTED || !mqttClient.connected()) return;

    uint32_t freeHeap = ESP.getFreeHeap();
    uint32_t minHeap  = ESP.getMinFreeHeap();
    int8_t rssi       = WiFi.RSSI();
    
    // 1. Read Internal CPU Temperature
    float cpu_temp = temperatureRead(); 

    // 2. Read from DHT11 Environment Sensor
    float dht_temp = dht.readTemperature(); 
    float humidity = dht.readHumidity();

    if (isnan(dht_temp)) dht_temp = 0.0;
    if (isnan(humidity)) humidity = 0.0;

    // 3. Read and Calculate Smoothed Battery Voltage
    long totalAdc = 0;
    const int numSamples = 20;
    for (int i = 0; i < numSamples; i++) {
      totalAdc += analogRead(BATT_PIN);
      delay(2); 
    }
    float avgAdc = (float)totalAdc / numSamples;
    
    float pinVoltage = (avgAdc / 4095.0) * V_REF;
    float batteryVoltage = pinVoltage * ((R1_VAL + R2_VAL) / R2_VAL) * BATT_CALIBRATION;
    
    // 4. Calculate Battery Percentage
    float batteryPercentage = ((batteryVoltage - 6.0) / (8.4 - 6.0)) * 100.0;
    if (batteryPercentage > 100.0) batteryPercentage = 100.0;
    if (batteryPercentage < 0.0) batteryPercentage = 0.0;

    // 5. Build and Publish Payload
    char payload[400]; 
    snprintf(payload, sizeof(payload),
        "{"
        "\"source\":\"node1\","
        "\"alive\":1,"
        "\"uptime\":%lu,"
        "\"free_heap\":%u,"
        "\"min_heap\":%u,"
        "\"temp\":%.1f,"
        "\"dht_temp\":%.1f,"
        "\"humidity\":%.1f,"
        "\"rssi\":%d,"
        "\"batt_v\":%.2f,"
        "\"batt_pct\":%.0f"
        "}",
        millis(), freeHeap, minHeap, cpu_temp, dht_temp, humidity, rssi, batteryVoltage, batteryPercentage
    );

    if(mqttClient.publish(topicHeartbeat, payload)) {
        Serial.printf("[MQTT] Heartbeat published | Temp: %.1fC | Hum: %.1f%% | Batt: %.2fV (%.0f%%)\n", 
                      dht_temp, humidity, batteryVoltage, batteryPercentage);
    }
}

void handleHeartbeat() {
    if (millis() - lastHeartbeat >= HEARTBEAT_INTERVAL) {
        lastHeartbeat = millis();
        sendHeartbeat();
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
  pinMode(gunshotLedPin, OUTPUT);
  pinMode(wifiLedPin, OUTPUT);
  pinMode(wake_up, OUTPUT);
  pinMode(wificonfig_button, INPUT_PULLUP);
  pinMode(BATT_PIN, INPUT); // Initialize Battery ADC Pin
  
  digitalWrite(gunshotLedPin, LOW);
  digitalWrite(wifiLedPin, LOW);
  digitalWrite(wake_up, LOW);
  dht.begin();
  
  // --- I2S Setup ---
  i2sInit();

  // --- Preferences & WiFi ---
  preferences.begin("my-app", false);
  String storedUrl = preferences.getString("server_url", "192.168.1.100"); 
  
  storedUrl.toCharArray(serverUrlBuffer, 100);
  serverUrl = storedUrl;

  WiFiManager wifiManager;
  wifiManager.setAPCallback(configModeCallback);
  
  WiFiManagerParameter custom_server_url("server", "MQTT Broker IP", serverUrlBuffer, 100);
  wifiManager.addParameter(&custom_server_url);

  if (!wifiManager.autoConnect("ESP32-Security")) {
    Serial.println("Failed to connect. Restarting...");
    delay(500);
    ESP.restart(); 
  } else {
    Serial.println("Connected to WiFi");
    ticker.detach(); 
    digitalWrite(wifiLedPin, HIGH); 
  }

  String newUrl = custom_server_url.getValue();
  if (newUrl != storedUrl) {
    preferences.putString("server_url", newUrl);
    serverUrl = newUrl;
  }

  // --- Initialize MQTT ---
  mqttClient.setServer(serverUrlBuffer, mqttPort);

  // --- Start Audio Task ---
  xTaskCreatePinnedToCore(
    AudioProcessingTask,   
    "AudioTask",           
    10000,                 
    NULL,                  
    1,                     
    &AudioTaskHandle,      
    0                      
  );
}

// ==========================================
// LOOP (Core 1)
// ==========================================
void loop() {
  // --- Real-time WiFi LED Status Update ---
  if (WiFi.status() == WL_CONNECTED) {
      digitalWrite(wifiLedPin, HIGH); 
  } else {
      digitalWrite(wifiLedPin, LOW);  
  }

  // ---- 0. MAINTAIN MQTT CONNECTION ----
  if (!mqttClient.connected()) {
    reconnectMQTT();
  } else {
    mqttClient.loop(); 
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
        WiFiManager wifiManager;
        wifiManager.setBreakAfterConfig(true);
        wifiManager.setAPCallback(configModeCallback); 

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

  // ---- 4. SLOW LOOP (Sensors 1Hz) ----
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis; 

    int motion1 = digitalRead(pirPin);
    int motion2 = digitalRead(rcwlPin);
    
    // --> ADDED THIS BACK SO YOU CAN SEE IT IS ALIVE <--
    Serial.printf("Sensors | PIR: %d | RCWL: %d | Status: %d\n", motion1, motion2, motion_status);
    
    // --- OCCUPANCY HOLD TIMER LOGIC ---
    int raw_motion = (motion1 == HIGH && motion2 == HIGH);
    bool trigger_send = false;

    if (raw_motion == 1) {
        lastRawMotionTime = currentMillis; 
        if (serverMotionState == false) {
            serverMotionState = true;
            trigger_send = true;
            Serial.println("[MOTION] Occupancy detected! Alerting server.");
        }
    } else {
        if (serverMotionState == true && (currentMillis - lastRawMotionTime >= MOTION_HOLD_TIME_MS)) {
            serverMotionState = false; 
            trigger_send = true;
            Serial.println("[MOTION] Occupancy cleared. Alerting server.");
        }
    }

    motion_status = serverMotionState ? 1 : 0; 
    bool anyEventActive = (motion_status == 1) || (gunshotDetected == true);

    digitalWrite(wake_up, anyEventActive ? HIGH : LOW);

    if (gunshotDetected) {
       gunshotDetected = false; 
       gunshotDataSent = false;
    }

    if (trigger_send) {
        sendData(false);
    }
  } 

  delay(10); 
}

// ==========================================
// I2S INIT (FIX 3: STEREO + FLAG CONFIG)
// ==========================================
void i2sInit() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = I2S_SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT, // Now listening in Stereo
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,     // High priority hardware interrupt
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
