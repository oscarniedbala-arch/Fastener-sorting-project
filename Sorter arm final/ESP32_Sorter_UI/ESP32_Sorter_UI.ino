/*
 * ESP32-CAM Enhanced for YOLO Integration
 * =========================================
 * 
 * Features:
 * - Clean web UI with live detection overlay
 * - JSON API for Python integration
 * - Trigger capture endpoint
 * - Status indicators
 * - Optimized for robotic arm control
 */

#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"
#include "camera_pins.h"

// ==================== CONFIGURATION ====================
static const char* WIFI_SSID = "00000";
static const char* WIFI_PASS = "00000";

// Camera settings
framesize_t g_framesize = FRAMESIZE_UXGA;  // 1600x1200 for better detection
int g_jpeg_quality = 12;

// Frame buffer
uint8_t* g_frame_buffer = nullptr;
size_t g_frame_len = 0;
size_t g_frame_capacity = 0;
SemaphoreHandle_t g_frame_mutex;

// Detection trigger
volatile bool g_trigger_capture = false;

// ==================== WEB UI ====================
const char HTML_UI[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Screw Sorter - Live View</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #5f72bd 0%, #9b23ea 100%);
            color: white;
            padding: 30px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        h1 {
            font-size: 32px;
            font-weight: 700;
        }
        
        .status {
            display: flex;
            gap: 20px;
            align-items: center;
        }
        
        .status-badge {
            padding: 8px 16px;
            border-radius: 20px;
            background: rgba(255,255,255,0.2);
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .status-indicator {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #4ade80;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            padding: 40px;
        }
        
        .video-panel {
            position: relative;
        }
        
        .video-container {
            position: relative;
            background: #1f2937;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        #camera-feed {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .detection-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }
        
        .controls {
            margin-top: 20px;
            display: flex;
            gap: 15px;
        }
        
        button {
            flex: 1;
            padding: 15px 30px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn-danger {
            background: #ef4444;
            color: white;
        }
        
        .btn-success {
            background: #10b981;
            color: white;
        }
        
        .info-panel {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .card {
            background: #f9fafb;
            padding: 25px;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
        }
        
        .card h3 {
            font-size: 18px;
            margin-bottom: 15px;
            color: #1f2937;
        }
        
        .detection-info {
            display: grid;
            gap: 12px;
        }
        
        .detection-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px;
            background: white;
            border-radius: 8px;
            border-left: 4px solid;
        }
        
        .detection-item.m3 { border-color: #ef4444; }
        .detection-item.m4 { border-color: #f59e0b; }
        .detection-item.m5 { border-color: #10b981; }
        .detection-item.m6 { border-color: #3b82f6; }
        
        .gauge-label {
            font-weight: 600;
            font-size: 16px;
        }
        
        .confidence {
            font-size: 14px;
            color: #6b7280;
        }
        
        .stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
        .stat-box {
            text-align: center;
            padding: 15px;
            background: white;
            border-radius: 8px;
        }
        
        .stat-value {
            font-size: 28px;
            font-weight: 700;
            color: #667eea;
        }
        
        .stat-label {
            font-size: 12px;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 5px;
        }
        
        .log {
            max-height: 200px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            color: #374151;
            padding: 15px;
            background: white;
            border-radius: 8px;
        }
        
        .log-entry {
            padding: 4px 0;
            border-bottom: 1px solid #f3f4f6;
        }
        
        .log-entry:last-child {
            border-bottom: none;
        }
        
        .timestamp {
            color: #9ca3af;
            margin-right: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔩 Screw Sorter - Live View</h1>
            <div class="status">
                <div class="status-badge">
                    <div class="status-indicator"></div>
                    <span id="status-text">Ready</span>
                </div>
                <div class="status-badge">
                    <span id="fps">0 FPS</span>
                </div>
            </div>
        </header>
        
        <div class="main-content">
            <div class="video-panel">
                <div class="video-container">
                    <img id="camera-feed" src="/stream" alt="Camera Feed">
                    <canvas id="detection-overlay" class="detection-overlay"></canvas>
                </div>
                
                <div class="controls">
                    <button class="btn-primary" onclick="triggerCapture()">📸 Capture & Detect</button>
                    <button class="btn-success" onclick="startSorting()">▶ Start Sorting</button>
                    <button class="btn-danger" onclick="stopSorting()">⏸ Stop</button>
                </div>
            </div>
            
            <div class="info-panel">
                <div class="card">
                    <h3>Current Detection</h3>
                    <div id="detection-info" class="detection-info">
                        <div style="text-align: center; padding: 20px; color: #9ca3af;">
                            No detections yet
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Statistics</h3>
                    <div class="stats">
                        <div class="stat-box">
                            <div class="stat-value" id="total-sorted">0</div>
                            <div class="stat-label">Total Sorted</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value" id="current-batch">0</div>
                            <div class="stat-label">Current Batch</div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Activity Log</h3>
                    <div id="activity-log" class="log">
                        <div class="log-entry">
                            <span class="timestamp">00:00:00</span>
                            System initialized
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let sortingActive = false;
        let totalSorted = 0;
        let currentBatch = 0;
        
        // Update camera feed
        const feed = document.getElementById('camera-feed');
        feed.onload = () => {
            // Update canvas size to match image
            const canvas = document.getElementById('detection-overlay');
            canvas.width = feed.naturalWidth;
            canvas.height = feed.naturalHeight;
        };
        
        // Trigger manual capture
        async function triggerCapture() {
            updateStatus('Capturing...');
            addLog('Manual capture triggered');
            
            try {
                const response = await fetch('/api/trigger', { method: 'POST' });
                const data = await response.json();
                updateStatus('Processing...');
                
                // Image will be processed by Python
                setTimeout(() => updateStatus('Ready'), 2000);
            } catch (e) {
                updateStatus('Error');
                addLog('Capture failed: ' + e.message);
            }
        }
        
        // Start automatic sorting
        function startSorting() {
            sortingActive = true;
            updateStatus('Sorting Active');
            addLog('Automatic sorting started');
            document.getElementById('current-batch').textContent = '0';
        }
        
        // Stop sorting
        function stopSorting() {
            sortingActive = false;
            updateStatus('Ready');
            addLog('Sorting stopped');
        }
        
        // Update status display
        function updateStatus(text) {
            document.getElementById('status-text').textContent = text;
        }
        
        // Add log entry
        function addLog(message) {
            const log = document.getElementById('activity-log');
            const now = new Date();
            const time = now.toTimeString().split(' ')[0];
            
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerHTML = `<span class="timestamp">${time}</span>${message}`;
            
            log.insertBefore(entry, log.firstChild);
            
            // Keep only last 20 entries
            while (log.children.length > 20) {
                log.removeChild(log.lastChild);
            }
        }
        
        // Update detection display (called from Python via WebSocket or polling)
        function updateDetection(detections) {
            const container = document.getElementById('detection-info');
            container.innerHTML = '';
            
            if (detections.length === 0) {
                container.innerHTML = '<div style="text-align: center; padding: 20px; color: #9ca3af;">No screws detected</div>';
                return;
            }
            
            detections.forEach(det => {
                const item = document.createElement('div');
                item.className = `detection-item ${det.class.toLowerCase()}`;
                item.innerHTML = `
                    <span class="gauge-label">${det.class}</span>
                    <span class="confidence">${(det.confidence * 100).toFixed(1)}%</span>
                `;
                container.appendChild(item);
            });
            
            // Draw on canvas
            drawDetections(detections);
        }
        
        // Draw bounding boxes on canvas
        function drawDetections(detections) {
            const canvas = document.getElementById('detection-overlay');
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            const colors = {
                'M3': '#ef4444',
                'M4': '#f59e0b',
                'M5': '#10b981',
                'M6': '#3b82f6'
            };
            
            detections.forEach(det => {
                const color = colors[det.class] || '#fff';
                ctx.strokeStyle = color;
                ctx.lineWidth = 3;
                ctx.strokeRect(det.x, det.y, det.w, det.h);
                
                // Label background
                ctx.fillStyle = color;
                ctx.fillRect(det.x, det.y - 25, 100, 25);
                
                // Label text
                ctx.fillStyle = 'white';
                ctx.font = 'bold 16px sans-serif';
                ctx.fillText(`${det.class} ${(det.confidence * 100).toFixed(0)}%`, det.x + 5, det.y - 5);
            });
        }
        
        // Poll for status updates
        setInterval(async () => {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                if (data.sorting_active !== sortingActive) {
                    sortingActive = data.sorting_active;
                    updateStatus(sortingActive ? 'Sorting Active' : 'Ready');
                }
                
                if (data.total_sorted !== totalSorted) {
                    totalSorted = data.total_sorted;
                    document.getElementById('total-sorted').textContent = totalSorted;
                }
            } catch (e) {
                // Ignore polling errors
            }
        }, 1000);
    </script>
</body>
</html>
)rawliteral";

// ==================== HTTP HANDLERS ====================

static esp_err_t handle_root(httpd_req_t *req) {
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, HTML_UI, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t handle_stream(httpd_req_t *req) {
    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache, no-store, must-revalidate");
    
    xSemaphoreTake(g_frame_mutex, portMAX_DELAY);
    
    if (g_frame_buffer && g_frame_len > 0) {
        httpd_resp_send(req, (const char*)g_frame_buffer, g_frame_len);
    } else {
        httpd_resp_send(req, "", 0);
    }
    
    xSemaphoreGive(g_frame_mutex);
    
    return ESP_OK;
}

static esp_err_t handle_api_trigger(httpd_req_t *req) {
    g_trigger_capture = true;
    
    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, "{\"status\":\"triggered\"}", HTTPD_RESP_USE_STRLEN);
    
    return ESP_OK;
}

static esp_err_t handle_api_status(httpd_req_t *req) {
    char json[256];
    snprintf(json, sizeof(json), 
        "{\"ip\":\"%s\",\"rssi\":%d,\"free_heap\":%u}",
        WiFi.localIP().toString().c_str(),
        WiFi.RSSI(),
        ESP.getFreeHeap()
    );
    
    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, json, HTTPD_RESP_USE_STRLEN);
    
    return ESP_OK;
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
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size = g_framesize;
    config.jpeg_quality = g_jpeg_quality;
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    
    return esp_camera_init(&config) == ESP_OK;
}

void capture_task(void* param) {
    while (true) {
        camera_fb_t* fb = esp_camera_fb_get();
        
        if (fb && fb->format == PIXFORMAT_JPEG) {
            xSemaphoreTake(g_frame_mutex, portMAX_DELAY);
            
            if (fb->len > g_frame_capacity) {
                if (g_frame_buffer) free(g_frame_buffer);
                g_frame_buffer = (uint8_t*)malloc(fb->len);
                g_frame_capacity = fb->len;
            }
            
            if (g_frame_buffer) {
                memcpy(g_frame_buffer, fb->buf, fb->len);
                g_frame_len = fb->len;
            }
            
            xSemaphoreGive(g_frame_mutex);
        }
        
        if (fb) esp_camera_fb_return(fb);
        
        vTaskDelay(pdMS_TO_TICKS(100));  // 10 FPS
    }
}

// ==================== SETUP ====================

void setup() {
    Serial.begin(115200);
    
    g_frame_mutex = xSemaphoreCreateMutex();
    
    // WiFi
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    
    Serial.println("\nWiFi connected");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
    
    // Camera
    if (!init_camera()) {
        Serial.println("Camera init failed");
        return;
    }
    
    // HTTP server
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    httpd_handle_t server = nullptr;
    
    if (httpd_start(&server, &config) == ESP_OK) {
        httpd_uri_t uri_root = {"/", HTTP_GET, handle_root, nullptr};
        httpd_uri_t uri_stream = {"/stream", HTTP_GET, handle_stream, nullptr};
        httpd_uri_t uri_trigger = {"/api/trigger", HTTP_POST, handle_api_trigger, nullptr};
        httpd_uri_t uri_status = {"/api/status", HTTP_GET, handle_api_status, nullptr};
        
        httpd_register_uri_handler(server, &uri_root);
        httpd_register_uri_handler(server, &uri_stream);
        httpd_register_uri_handler(server, &uri_trigger);
        httpd_register_uri_handler(server, &uri_status);
    }
    
    // Start capture task
    xTaskCreatePinnedToCore(capture_task, "capture", 8192, nullptr, 1, nullptr, 1);
    
    Serial.println("Web UI ready!");
    Serial.print("Open: http://");
    Serial.println(WiFi.localIP());
}

void loop() {
    delay(1000);
}
