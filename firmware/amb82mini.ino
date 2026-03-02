#include <WiFi.h>
// #include <WiFiSSLClient.h> // Removed for HTTP
#include "StreamIO.h"
#include "VideoStream.h"
#include "AudioStream.h"
#include "AudioEncoder.h"
#include "MP4Recording.h"
#include "AmebaFatFS.h"
#include "sys_api.h" 
#include "PowerMode.h" // Preserved Deep Sleep

#define CHANNEL 0
#define FILENAME "TestRecordingAV.mp4"
#define CONFIG_FILENAME "wifi_conf.txt"
#define RECORDING_DURATION 10 

// ================= DEEP SLEEP CONFIG =================
#define WAKEUP_SOURCE 1   // 1: wake up by AON GPIO
#define WAKEUP_PIN    21  // Pin 21 (High)
#define RETENTION     0

// ================= LED DEFINITIONS =================
#define LED_RECORDING  LED_G  
#define LED_WIFI       LED_B  

// ================= GLOBALS =================
String config_ssid = "";
String config_pass = "";
String config_host = "";
int server_port = 5000; // Changed to 5000 for local HTTP (Flask default)

VideoSetting configV(VIDEO_HD, 15, VIDEO_H264, 0);
AudioSetting configA(0); 

Audio audio;
AAC aac;
MP4Recording mp4;
StreamIO audioStreamer(1, 1);
StreamIO avMixStreamer(2, 1);
AmebaFatFS fs;

// WiFiSSLClient client; // Removed
WiFiClient client;       // Added Standard HTTP Client
WiFiServer server(80); 

// ================= HELPER: LOAD/SAVE CONFIG =================

void saveConfig(String s, String p, String h) {
    Serial.println("Saving config to SD...");
    File file = fs.open(CONFIG_FILENAME);
    if (file) {
        file.println(s);
        file.println(p);
        file.println(h);
        file.close();
        Serial.println("Config Saved!");
    } else {
        Serial.println("ERROR: Could not save config.");
    }
}

bool loadConfig() {
    Serial.println("Loading config from SD...");
    if (!fs.exists(CONFIG_FILENAME)) {
        Serial.println("Config file not found.");
        return false;
    }

    File file = fs.open(CONFIG_FILENAME);
    if (file) {
        config_ssid = file.readStringUntil('\n');
        config_pass = file.readStringUntil('\n');
        config_host = file.readStringUntil('\n');
        
        config_ssid.trim();
        config_pass.trim();
        config_host.trim();

        // SAFETY: Remove http:// or https:// if user typed it in the file
        config_host.replace("http://", "");
        config_host.replace("https://", "");
        
        file.close();
        
        Serial.println("Loaded Config:");
        Serial.println("SSID: " + config_ssid);
        Serial.println("PASS: " + config_pass);
        Serial.println("HOST: " + config_host);
        
        if(config_ssid.length() == 0 || config_host.length() == 0) return false;
        return true;
    }
    return false;
}

// ================= CONFIG PORTAL (AP MODE) =================

void startConfigPortal() {
    Serial.println("\n!!! STARTING CONFIG PORTAL !!!");
    digitalWrite(LED_WIFI, HIGH); 

    Serial.println("1. Connect to WiFi: 'Ameba-Cam-Setup'");
    Serial.println("2. Open Browser: http://192.168.1.1");

    WiFi.apbegin("Ameba-Cam-Setup", "12345678", "1", 0); 
    server.begin();

    while (true) {
        WiFiClient client = server.available();
        if (client) {
            String currentLine = "";
            String request = "";
            while (client.connected()) {
                if (client.available()) {
                    char c = client.read();
                    request += c;
                    if (c == '\n') {
                        if (currentLine.length() == 0) {
                            if (request.indexOf("GET /save?") >= 0) {
                                Serial.println("Received Config Save Request");
                                int idx_ssid = request.indexOf("ssid=");
                                int idx_pass = request.indexOf("&pass=");
                                int idx_host = request.indexOf("&host=");
                                int idx_end  = request.indexOf(" HTTP");

                                if (idx_ssid > 0 && idx_host > 0) {
                                    String new_ssid = request.substring(idx_ssid + 5, idx_pass);
                                    String new_pass = request.substring(idx_pass + 6, idx_host);
                                    String new_host = request.substring(idx_host + 6, idx_end);
                                    
                                    // Basic URL Decode
                                    new_ssid.replace("%20", " ");

                                    saveConfig(new_ssid, new_pass, new_host);

                                    client.println("HTTP/1.1 200 OK");
                                    client.println("Content-Type: text/html");
                                    client.println();
                                    client.println("<h1>Saved! Rebooting...</h1>");
                                    delay(1000);
                                    client.stop();
                                    
                                    Serial.println("Rebooting...");
                                    NVIC_SystemReset();
                                }
                            } else {
                                client.println("HTTP/1.1 200 OK");
                                client.println("Content-Type: text/html");
                                client.println();
                                client.println("<!DOCTYPE HTML><html>");
                                client.println("<h1>Ameba Cam Setup</h1>");
                                client.println("<form action='/save' method='GET'>");
                                client.println("SSID: <input type='text' name='ssid' value='" + config_ssid + "'><br><br>");
                                client.println("Pass: <input type='text' name='pass' value='" + config_pass + "'><br><br>");
                                client.println("Host IP: <input type='text' name='host' value='" + config_host + "'><br><br>");
                                client.println("<input type='submit' value='Save & Reboot'>");
                                client.println("</form></html>");
                            }
                            break;
                        } else {
                            currentLine = "";
                        }
                    } else if (c != '\r') {
                        currentLine += c;
                    }
                }
            }
            client.stop();
        }
    }
}

// ================= CORE FUNCTIONS =================

void stopAllStreaming() {
    Serial.println("--- Stopping All Streaming ---");
    digitalWrite(LED_RECORDING, LOW);

    mp4.end();
    delay(500);
    avMixStreamer.pause();
    audioStreamer.pause();
    delay(200);
    audio.end();
    delay(200);
    Camera.channelEnd(CHANNEL);
    delay(500);
    Serial.println("All streaming stopped.");
}

void connectToWiFi() {
    Serial.println("\n--- Connecting to WiFi: " + config_ssid + " ---");
    
    char ssid_buf[50];
    char pass_buf[50];
    config_ssid.toCharArray(ssid_buf, 50);
    config_pass.toCharArray(pass_buf, 50);

    WiFi.begin(ssid_buf, pass_buf);
    
    int retries = 0;
    while (WiFi.status() != WL_CONNECTED) {
        digitalWrite(LED_WIFI, HIGH);
        delay(250);
        digitalWrite(LED_WIFI, LOW);
        delay(250);

        Serial.print(".");
        retries++;
        if(retries > 30) { 
            Serial.println("\nFailed to connect to WiFi.");
            startConfigPortal(); 
            return;
        }
    }
    digitalWrite(LED_WIFI, LOW);
    Serial.println("\nWiFi Connected!");
}

void uploadVideo() {
    if (WiFi.status() != WL_CONNECTED) return;

    File file = fs.open(FILENAME);
    if (!file) {
        Serial.println("ERROR: File does not exist!");
        return;
    }

    uint32_t fileSize = file.size();
    if (fileSize < 1000) {
        Serial.println("ERROR: File too small.");
        file.close();
        return;
    }
    file.close();

    Serial.println("--- Starting Upload to: " + config_host + ":" + String(server_port) + " ---");

    char host_buf[100];
    config_host.toCharArray(host_buf, 100);

    client.setTimeout(20000);
    
    // Connect via Standard HTTP
    if (client.connect(host_buf, server_port)) {
        Serial.println("Connected to Server!");
    } else {
        Serial.println("Connection failed! Check IP and Port.");
        return;
    }

    file = fs.open(FILENAME);
    
    String boundary = "---AmebaBoundary";
    String head = "--" + boundary + "\r\nContent-Disposition: form-data; name=\"source\"\r\n\r\namb82\r\n";
    head += "--" + boundary + "\r\nContent-Disposition: form-data; name=\"video\"; filename=\"" + FILENAME + "\"\r\nContent-Type: video/mp4\r\n\r\n";
    String tail = "\r\n--" + boundary + "--\r\n";
    
    uint32_t totalLen = head.length() + fileSize + tail.length();

    client.println("POST /api/video/upload HTTP/1.1");
    client.println("Host: " + config_host + ":" + String(server_port));
    client.println("Content-Type: multipart/form-data; boundary=" + boundary);
    client.print("Content-Length: "); client.println(totalLen);
    client.println("Connection: close");
    client.println(); 

    client.print(head);

    uint8_t buf[1024];
    size_t bytesSent = 0;
    Serial.println("Uploading...");

    // Blink Logic Variables
    unsigned long previousMillis = 0;
    const long blinkInterval = 200; 
    bool ledState = LOW;

    while (file.available()) {
        size_t len = file.read(buf, sizeof(buf));
        client.write(buf, len);
        bytesSent += len;
        
        if (bytesSent % 102400 == 0) Serial.print("."); 

        // Non-blocking Blink Logic
        unsigned long currentMillis = millis();
        if (currentMillis - previousMillis >= blinkInterval) {
            previousMillis = currentMillis;
            ledState = !ledState;
            digitalWrite(LED_RECORDING, ledState);
        }

        // --- THE FIX: Let the Wi-Fi driver catch its breath ---
        delay(2); 
        // ------------------------------------------------------
    }
    file.close();
    Serial.println("\nFile Sent.");
    client.print(tail);
    client.flush();
    
    // Ensure LED turns off after upload
    digitalWrite(LED_RECORDING, LOW);
    
    client.stop();
    Serial.println("Upload Complete.");
}
// ================= MAIN SETUP =================

void setup() {
    Serial.begin(115200);
    
    // 1. Initialize Deep Sleep Configuration
    PowerMode.begin(DEEPSLEEP_MODE, WAKEUP_SOURCE, RETENTION, WAKEUP_PIN);
    
    pinMode(LED_RECORDING, OUTPUT);
    pinMode(LED_WIFI, OUTPUT);
    digitalWrite(LED_RECORDING, LOW);
    digitalWrite(LED_WIFI, LOW);
    
    if (!fs.begin()) {
        Serial.println("SD Mount Failed!");
        while(1);
    }

    if (!loadConfig()) {
        Serial.println("No Config Found. Starting Config Portal...");
        startConfigPortal(); 
    }

    Serial.println("--- Starting Recording Sequence ---");
    digitalWrite(LED_RECORDING, HIGH);

    Camera.configVideoChannel(CHANNEL, configV);
    Camera.videoInit();
    audio.configAudio(configA);
    audio.begin();
    aac.configAudio(configA);
    aac.begin();
    
    mp4.configVideo(configV);
    mp4.configAudio(configA, CODEC_AAC);
    mp4.setRecordingDuration(RECORDING_DURATION);
    mp4.setRecordingFileCount(1);
    mp4.setRecordingFileName("TestRecordingAV");
    
    audioStreamer.registerInput(audio);
    audioStreamer.registerOutput(aac);
    audioStreamer.begin();
    
    avMixStreamer.registerInput1(Camera.getStream(CHANNEL));
    avMixStreamer.registerInput2(aac);
    avMixStreamer.registerOutput(mp4);
    avMixStreamer.begin();

    Camera.channelBegin(CHANNEL);
    Serial.println("Recording...");
    mp4.begin();
    
    delay((RECORDING_DURATION + 5) * 1000);
    
    stopAllStreaming(); 

    connectToWiFi(); 
    uploadVideo();

    // ================= ENTER DEEP SLEEP =================
    Serial.println("Task complete. Entering Deep Sleep. Wake up via Pin 21 High.");
    delay(1000); 
    PowerMode.start(); 
    // ====================================================
}

void loop() {
    // Execution will never reach here as board resets on wakeup
}
