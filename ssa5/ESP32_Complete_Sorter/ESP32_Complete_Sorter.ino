/*
 * ESP32-CAM + PCA9685 Screw Sorter
 * ===========================================
 */

#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"
#include <Wire.h>

// ==================== PINS ====================
// Camera pins (standard ESP32-CAM)
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// PCA9685 I2C pins
#define PCA_SDA_PIN       15
#define PCA_SCL_PIN       14
#define PCA9685_ADDR      0x40

// PCA9685 registers
#define MODE1       0x00
#define PRESCALE    0xFE
#define LED0_ON_L   0x06

// ==================== CONFIG ====================
const char* WIFI_SSID = "VM4914218";
const char* WIFI_PASS = "Hk9ygmqs6dvd";

// Servo positions (pulse width in microseconds)
int servo_positions[4] = {1500, 1500, 1500, 1500}; // Base, Shoulder, Elbow, Gripper

// ==================== PCA9685 FUNCTIONS ====================

void pca_write8(uint8_t reg, uint8_t val) {
    Wire.beginTransmission(PCA9685_ADDR);
    Wire.write(reg);
    Wire.write(val);
    Wire.endTransmission();
}

uint8_t pca_read8(uint8_t reg) {
    Wire.beginTransmission(PCA9685_ADDR);
    Wire.write(reg);
    Wire.endTransmission(false);
    Wire.requestFrom(PCA9685_ADDR, (uint8_t)1);
    return Wire.available() ? Wire.read() : 0xFF;
}

void pca_setPWM(uint8_t channel, uint16_t on, uint16_t off) {
    uint8_t reg = LED0_ON_L + 4 * channel;
    Wire.beginTransmission(PCA9685_ADDR);
    Wire.write(reg);
    Wire.write(on & 0xFF);
    Wire.write((on >> 8) & 0xFF);
    Wire.write(off & 0xFF);
    Wire.write((off >> 8) & 0xFF);
    Wire.endTransmission();
}

void pca_setFreq(float freq_hz) {
    float prescaleval = 25000000.0 / 4096.0 / freq_hz - 1.0;
    uint8_t prescale = (uint8_t)(prescaleval + 0.5);
    
    uint8_t oldmode = pca_read8(MODE1);
    pca_write8(MODE1, (oldmode & 0x7F) | 0x10); // Sleep
    pca_write8(PRESCALE, prescale);
    pca_write8(MODE1, oldmode);
    delay(5);
    pca_write8(MODE1, oldmode | 0xA1); // Restart + auto-increment
}

void pca_setServo(uint8_t channel, int microseconds) {
    // Convert microseconds to PCA9685 ticks (50Hz = 20ms period)
    uint16_t ticks = (microseconds * 4096L) / 20000L;
    pca_setPWM(channel, 0, ticks);
    servo_positions[channel] = microseconds;
}

// ==================== CAMERA ====================

bool init_camera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    
    // HIGH RES, LOW FPS for YOLO
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size = FRAMESIZE_XGA;  // 1024x768 - good for YOLO
    config.jpeg_quality = 10;            // High quality
    config.fb_count = 1;                 // Single buffer = lower FPS
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_LATEST;
    
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed: 0x%x\n", err);
        return false;
    }
    
    // Configure for quality
    sensor_t * s = esp_camera_sensor_get();
    if (s) {
        s->set_brightness(s, 0);
        s->set_contrast(s, 0);
        s->set_saturation(s, 0);
        s->set_whitebal(s, 1);
        s->set_awb_gain(s, 1);
        s->set_exposure_ctrl(s, 1);
        s->set_aec2(s, 0);
        s->set_gain_ctrl(s, 1);
        s->set_lenc(s, 1);
    }
    
    Serial.println("Camera OK");
    return true;
}

// ==================== HTTP HANDLERS ====================

static esp_err_t index_handler(httpd_req_t *req) {
    const char html[] = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Screw Sorter Control</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial; background: #667eea; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 10px; padding: 20px; }
        h1 { color: #667eea; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
        #stream { width: 100%; border-radius: 5px; background: #000; }
        .controls { background: #f5f5f5; padding: 15px; border-radius: 5px; margin-top: 10px; }
        .slider-group { margin: 15px 0; }
        .slider-group label { display: block; margin-bottom: 5px; font-weight: bold; }
        input[type="range"] { width: 100%; }
        .value { display: inline-block; min-width: 50px; text-align: right; }
        button { padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; background: #667eea; color: white; cursor: pointer; font-size: 14px; }
        button:hover { background: #5568d3; }
        .status { padding: 10px; background: #e8f5e9; border-left: 4px solid #4caf50; margin-bottom: 15px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔩 Screw Sorter Control Panel</h1>
        
        <div class="status">
            <strong>Status:</strong> <span id="status">Ready</span> | 
            <strong>Last detection:</strong> <span id="detection">None</span>
        </div>
        
        <div class="grid">
            <div>
                <img id="stream" src="/stream">
                <div style="text-align:center; margin-top:10px;">
                    <button onclick="capture()">📸 Capture for YOLO</button>
                    <button onclick="homeArm()">🏠 Home Arm</button>
                </div>
            </div>
            
            <div>
                <div class="controls">
                    <h3>Servo Control</h3>
                    
                    <div class="slider-group">
                        <label>Base (CH0): <span class="value" id="val0">1500</span> μs</label>
                        <input type="range" min="600" max="2400" value="1500" 
                               oninput="updateServo(0, this.value)">
                    </div>
                    
                    <div class="slider-group">
                        <label>Shoulder (CH1): <span class="value" id="val1">1500</span> μs</label>
                        <input type="range" min="600" max="2400" value="1500" 
                               oninput="updateServo(1, this.value)">
                    </div>
                    
                    <div class="slider-group">
                        <label>Elbow (CH2): <span class="value" id="val2">1500</span> μs</label>
                        <input type="range" min="600" max="2400" value="1500" 
                               oninput="updateServo(2, this.value)">
                    </div>
                    
                    <div class="slider-group">
                        <label>Gripper (CH3): <span class="value" id="val3">1500</span> μs</label>
                        <input type="range" min="600" max="2400" value="1500" 
                               oninput="updateServo(3, this.value)">
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function updateServo(channel, value) {
            document.getElementById('val' + channel).textContent = value;
            fetch('/api/servo?ch=' + channel + '&val=' + value);
        }
        
        function capture() {
            document.getElementById('status').textContent = 'Capturing...';
            fetch('/api/capture').then(() => {
                document.getElementById('status').textContent = 'Ready';
            });
        }
        
        function homeArm() {
            fetch('/api/home').then(() => {
                // Reset sliders
                for (let i = 0; i < 4; i++) {
                    document.getElementById('val' + i).textContent = '1500';
                    document.querySelectorAll('input[type="range"]')[i].value = 1500;
                }
            });
        }
        
        // Poll for detection status
        setInterval(async () => {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                if (data.detection) {
                    document.getElementById('detection').textContent = data.detection;
                }
            } catch (e) {}
        }, 1000);
    </script>
</body>
</html>
)rawliteral";
    
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, html, HTTPD_RESP_USE_STRLEN);
}

// Stream handler - LOW FPS, HIGH RES
static esp_err_t stream_handler(httpd_req_t *req) {
    camera_fb_t * fb = NULL;
    esp_err_t res = ESP_OK;
    
    res = httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=frame");
    if (res != ESP_OK) return res;
    
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    
    while (true) {
        fb = esp_camera_fb_get();
        if (!fb) { res = ESP_FAIL; break; }
        
        if (res == ESP_OK) {
            res = httpd_resp_send_chunk(req, "\r\n--frame\r\nContent-Type: image/jpeg\r\n\r\n", 37);
            if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
        }
        
        esp_camera_fb_return(fb);
        if (res != ESP_OK) break;
        
        // LOW FPS - delay between frames
        delay(500);  // ~2 FPS
    }
    
    return res;
}

// Capture single high-res image for YOLO
static esp_err_t capture_handler(httpd_req_t *req) {
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    
    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    
    return res;
}

// Servo control API
static esp_err_t servo_handler(httpd_req_t *req) {
    char query[64];
    int channel = -1, value = -1;
    
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char ch_str[8], val_str[8];
        if (httpd_query_key_value(query, "ch", ch_str, sizeof(ch_str)) == ESP_OK) {
            channel = atoi(ch_str);
        }
        if (httpd_query_key_value(query, "val", val_str, sizeof(val_str)) == ESP_OK) {
            value = atoi(val_str);
        }
    }
    
    if (channel >= 0 && channel < 4 && value >= 600 && value <= 2400) {
        pca_setServo(channel, value);
        Serial.printf("Servo CH%d: %d us\n", channel, value);
    }
    
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, "{\"ok\":1}", HTTPD_RESP_USE_STRLEN);
}

// Home position
static esp_err_t home_handler(httpd_req_t *req) {
    for (int i = 0; i < 4; i++) {
        pca_setServo(i, 1500);  // Center position
    }
    Serial.println("Arm homed");
    
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(req, "{\"ok\":1}", HTTPD_RESP_USE_STRLEN);
}

// Status API
static esp_err_t status_handler(httpd_req_t *req) {
    char json[256];
    snprintf(json, sizeof(json),
        "{\"servos\":[%d,%d,%d,%d],\"detection\":\"None\"}",
        servo_positions[0], servo_positions[1], servo_positions[2], servo_positions[3]);
    
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, json, HTTPD_RESP_USE_STRLEN);
}

// ==================== SETUP ====================

void start_webserver() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    httpd_handle_t server = NULL;
    
    if (httpd_start(&server, &config) == ESP_OK) {
        httpd_uri_t uri_index = {"/", HTTP_GET, index_handler, NULL};
        httpd_register_uri_handler(server, &uri_index);
        
        httpd_uri_t uri_stream = {"/stream", HTTP_GET, stream_handler, NULL};
        httpd_register_uri_handler(server, &uri_stream);
        
        httpd_uri_t uri_capture = {"/api/capture", HTTP_GET, capture_handler, NULL};
        httpd_register_uri_handler(server, &uri_capture);
        
        httpd_uri_t uri_servo = {"/api/servo", HTTP_GET, servo_handler, NULL};
        httpd_register_uri_handler(server, &uri_servo);
        
        httpd_uri_t uri_home = {"/api/home", HTTP_GET, home_handler, NULL};
        httpd_register_uri_handler(server, &uri_home);
        
        httpd_uri_t uri_status = {"/api/status", HTTP_GET, status_handler, NULL};
        httpd_register_uri_handler(server, &uri_status);
        
        Serial.println("Webserver started");
    }
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n=== ESP32-CAM + PCA9685 Screw Sorter ===\n");
    
    // Init PCA9685
    Wire.begin(PCA_SDA_PIN, PCA_SCL_PIN);
    pca_write8(MODE1, 0x00);
    delay(10);
    pca_setFreq(50.0);
    
    // Home servos
    for (int i = 0; i < 4; i++) {
        pca_setServo(i, 1500);
    }
    Serial.println("PCA9685 OK");
    
    // WiFi
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi OK");
    Serial.println("IP: " + WiFi.localIP().toString());
    
    // Camera
    if (!init_camera()) {
        Serial.println("Camera FAILED!");
        while(1) delay(1000);
    }
    
    // Webserver
    start_webserver();
    
    Serial.println("\n=== READY ===");
    Serial.println("Open: http://" + WiFi.localIP().toString());
}

void loop() {
    delay(1000);
}
