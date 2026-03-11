#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

// -------------------- WiFi --------------------
static const char* WIFI_SSID = "******";
static const char* WIFI_PASS = ""******";"

// -------------------- AI Thinker ESP32-CAM pin map --------------------
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

#define FLASH_LED_GPIO     4

// -------------------- HTTP servers --------------------
static httpd_handle_t camera_httpd = NULL;
static httpd_handle_t stream_httpd = NULL;

// -------------------- LED --------------------
static bool led_state = false;

// -------------------- Camera mutex to prevent crashes --------------------
static SemaphoreHandle_t g_cam_mutex = nullptr;

// -------------------- Stream constants --------------------
static const char* STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=frame";
static const char* STREAM_BOUNDARY = "\r\n--frame\r\n";
static const char* STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// -------------------- Profiles --------------------
// Smooth stream
static framesize_t STREAM_FRAMESIZE   = FRAMESIZE_QVGA; // 320x240
static int         STREAM_QUALITY     = 18;             // more compressed -> faster
static int         STREAM_DELAY_MS    = 60;             // pacing

// HQ stream (stable HQ): CIF at ~2 FPS
static framesize_t HQSTREAM_FRAMESIZE = FRAMESIZE_CIF;  // 352x288 (more stable than VGA)
static int         HQSTREAM_QUALITY   = 12;
static int         HQSTREAM_DELAY_MS  = 500;            // ~2 fps

// Capture (HQ still)
static framesize_t CAPTURE_FRAMESIZE  = FRAMESIZE_VGA;  // 640x480
static int         CAPTURE_QUALITY    = 12;

static volatile int g_stream_delay_ms = 60;

// -------------------- Helpers --------------------
static void apply_camera(framesize_t fs, int quality) {
  sensor_t* s = esp_camera_sensor_get();
  if (!s) return;
  s->set_framesize(s, fs);
  s->set_quality(s, quality);
}

static esp_err_t send_text(httpd_req_t *req, const char* txt) {
  httpd_resp_set_type(req, "text/plain");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, txt, HTTPD_RESP_USE_STRLEN);
}

// -------------------- GUI --------------------
static esp_err_t index_handler(httpd_req_t *req) {
  const char* html =
    "<!doctype html><html><head><meta name='viewport' content='width=device-width,initial-scale=1'>"
    "<title>ESP32-CAM</title>"
    "<style>"
    "body{font-family:Arial,Helvetica,sans-serif;margin:16px;max-width:860px}"
    ".row{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0}"
    "button,a.btn{display:inline-block;padding:10px 14px;border:1px solid #333;background:#f2f2f2;"
    "border-radius:10px;text-decoration:none;color:#000;cursor:pointer}"
    ".card{border:1px solid #ddd;border-radius:14px;padding:14px;margin:12px 0}"
    ".small{color:#444;font-size:13px;line-height:1.4}"
    "code{background:#f6f6f6;padding:2px 6px;border-radius:6px}"
    "</style></head><body>"
    "<h2>ESP32-CAM Control</h2>"

    "<div class='card'>"
    "<h3>View</h3>"
    "<div class='row'>"
    "<a class='btn' href='/capture' target='_blank'>Capture JPEG (HQ)</a>"
    "<a class='btn' id='streamlink' href='#' target='_blank'>Open Stream (port 81)</a>"
    "</div>"
    "<div class='small'>Stream URL: <code>http://DEVICE_IP:81/stream</code></div>"
    "</div>"

    "<div class='card'>"
    "<h3>Camera Profiles</h3>"
    "<div class='row'>"
    "<button onclick=\"fetch('/profile?mode=stream').then(r=>r.text()).then(setMsg)\">Stream (smooth)</button>"
    "<button onclick=\"fetch('/profile?mode=hq2').then(r=>r.text()).then(setMsg)\">HQ Stream (~2 FPS)</button>"
    "<button onclick=\"fetch('/profile?mode=capture').then(r=>r.text()).then(setMsg)\">Capture profile</button>"
    "</div>"
    "<div class='small'>HQ Stream uses CIF to avoid buffer overflows. Capture uses VGA.</div>"
    "<div class='small'>Status: <span id='msg'>-</span></div>"
    "</div>"

    "<div class='card'>"
    "<h3>LED</h3>"
    "<div class='row'>"
    "<button onclick=\"fetch('/led?state=on').then(r=>r.text()).then(setLed)\">LED ON</button>"
    "<button onclick=\"fetch('/led?state=off').then(r=>r.text()).then(setLed)\">LED OFF</button>"
    "<button onclick=\"fetch('/led?state=toggle').then(r=>r.text()).then(setLed)\">TOGGLE</button>"
    "</div>"
    "<div class='small'>State: <span id='ledstate'>UNKNOWN</span></div>"
    "</div>"

    "<script>"
    "function setLed(t){document.getElementById('ledstate').innerText=t;}"
    "function setMsg(t){document.getElementById('msg').innerText=t;}"
    "window.addEventListener('load', ()=>{"
    "  const ip = window.location.hostname;"
    "  document.getElementById('streamlink').href = 'http://' + ip + ':81/stream';"
    "});"
    "</script>"
    "</body></html>";

  httpd_resp_set_type(req, "text/html");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_send(req, html, HTTPD_RESP_USE_STRLEN);
  return ESP_OK;
}

// -------------------- Profile handler --------------------
// /profile?mode=stream | hq2 | capture
static esp_err_t profile_handler(httpd_req_t *req) {
  size_t buf_len = httpd_req_get_url_query_len(req) + 1;
  if (buf_len <= 1) return send_text(req, "Missing mode");

  char* buf = (char*)malloc(buf_len);
  if (!buf) return send_text(req, "No mem");

  if (httpd_req_get_url_query_str(req, buf, buf_len) != ESP_OK) {
    free(buf);
    return send_text(req, "Bad query");
  }

  char mode[16] = {0};
  if (httpd_query_key_value(buf, "mode", mode, sizeof(mode)) != ESP_OK) {
    free(buf);
    return send_text(req, "Missing mode=");
  }
  free(buf);

  // If streaming is open, profile switches still work, but mutex will protect capture/stream access.
  if (!strcmp(mode, "stream")) {
    apply_camera(STREAM_FRAMESIZE, STREAM_QUALITY);
    g_stream_delay_ms = STREAM_DELAY_MS;
    return send_text(req, "OK: Stream profile applied");
  } else if (!strcmp(mode, "hq2")) {
    apply_camera(HQSTREAM_FRAMESIZE, HQSTREAM_QUALITY);
    g_stream_delay_ms = HQSTREAM_DELAY_MS;
    return send_text(req, "OK: HQ stream (~2 FPS) profile applied");
  } else if (!strcmp(mode, "capture")) {
    apply_camera(CAPTURE_FRAMESIZE, CAPTURE_QUALITY);
    return send_text(req, "OK: Capture profile applied");
  }

  return send_text(req, "Unknown mode");
}

// -------------------- Capture handler (mutex protected) --------------------
static esp_err_t capture_handler(httpd_req_t *req) {
  if (!g_cam_mutex || xSemaphoreTake(g_cam_mutex, pdMS_TO_TICKS(1500)) != pdTRUE) {
    // 503 without relying on HTTPD_503 macro
    httpd_resp_send_err(req, (httpd_err_code_t)503, "Camera busy");
    return ESP_FAIL;
  }

  apply_camera(CAPTURE_FRAMESIZE, CAPTURE_QUALITY);

  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    xSemaphoreGive(g_cam_mutex);
    httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Camera capture failed");
    return ESP_FAIL;
  }

  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  esp_err_t res = httpd_resp_send(req, (const char*)fb->buf, fb->len);
  esp_camera_fb_return(fb);

  // Revert to safe smooth streaming defaults
  apply_camera(STREAM_FRAMESIZE, STREAM_QUALITY);
  g_stream_delay_ms = STREAM_DELAY_MS;

  xSemaphoreGive(g_cam_mutex);
  return res;
}

// -------------------- Stream handler (mutex + disconnect-safe) --------------------
static esp_err_t stream_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  while (true) {
    if (!g_cam_mutex || xSemaphoreTake(g_cam_mutex, pdMS_TO_TICKS(1500)) != pdTRUE) {
      vTaskDelay(pdMS_TO_TICKS(50));
      continue;
    }

    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) {
      xSemaphoreGive(g_cam_mutex);
      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }

    esp_err_t res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
    if (res != ESP_OK) {
      esp_camera_fb_return(fb);
      xSemaphoreGive(g_cam_mutex);
      break;
    }

    char part_buf[64];
    int hlen = snprintf(part_buf, sizeof(part_buf), STREAM_PART, fb->len);
    res = httpd_resp_send_chunk(req, part_buf, hlen);
    if (res != ESP_OK) {
      esp_camera_fb_return(fb);
      xSemaphoreGive(g_cam_mutex);
      break;
    }

    res = httpd_resp_send_chunk(req, (const char*)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    xSemaphoreGive(g_cam_mutex);

    if (res != ESP_OK) break;

    int d = g_stream_delay_ms;
    if (d < 0) d = 0;
    vTaskDelay(pdMS_TO_TICKS(d));
  }

  httpd_resp_send_chunk(req, NULL, 0);
  return ESP_OK;
}

// -------------------- Control handler --------------------
static esp_err_t control_handler(httpd_req_t *req) {
  size_t buf_len = httpd_req_get_url_query_len(req) + 1;
  if (buf_len <= 1) {
    httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Missing query");
    return ESP_FAIL;
  }

  char* buf = (char*)malloc(buf_len);
  if (!buf) {
    httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "No mem");
    return ESP_FAIL;
  }

  if (httpd_req_get_url_query_str(req, buf, buf_len) != ESP_OK) {
    free(buf);
    httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Bad query");
    return ESP_FAIL;
  }

  char var[32] = {0};
  char val[16] = {0};

  if (httpd_query_key_value(buf, "var", var, sizeof(var)) != ESP_OK ||
      httpd_query_key_value(buf, "val", val, sizeof(val)) != ESP_OK) {
    free(buf);
    httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Need var and val");
    return ESP_FAIL;
  }
  free(buf);

  sensor_t * s = esp_camera_sensor_get();
  int value = atoi(val);

  int r = 0;
  if (!strcmp(var, "framesize"))         r = s->set_framesize(s, (framesize_t)value);
  else if (!strcmp(var, "quality"))     r = s->set_quality(s, value);
  else if (!strcmp(var, "brightness"))  r = s->set_brightness(s, value);
  else if (!strcmp(var, "contrast"))    r = s->set_contrast(s, value);
  else if (!strcmp(var, "saturation"))  r = s->set_saturation(s, value);
  else if (!strcmp(var, "gainceiling")) r = s->set_gainceiling(s, (gainceiling_t)value);
  else {
    httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Unknown var");
    return ESP_FAIL;
  }

  if (r != 0) {
    httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Setting failed");
    return ESP_FAIL;
  }

  return send_text(req, "OK");
}

// -------------------- LED handler --------------------
static esp_err_t led_handler(httpd_req_t *req) {
  bool has_query = (httpd_req_get_url_query_len(req) > 0);

  if (has_query) {
    size_t len = httpd_req_get_url_query_len(req) + 1;
    char* buf = (char*)malloc(len);
    if (!buf) {
      httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "No mem");
      return ESP_FAIL;
    }

    if (httpd_req_get_url_query_str(req, buf, len) == ESP_OK) {
      char state[16] = {0};
      if (httpd_query_key_value(buf, "state", state, sizeof(state)) == ESP_OK) {
        if (!strcmp(state, "on")) led_state = true;
        else if (!strcmp(state, "off")) led_state = false;
        else if (!strcmp(state, "toggle")) led_state = !led_state;
      } else {
        led_state = !led_state;
      }
    }
    free(buf);
  } else {
    led_state = !led_state;
  }

  digitalWrite(FLASH_LED_GPIO, led_state ? HIGH : LOW);
  return send_text(req, led_state ? "ON" : "OFF");
}

// -------------------- Start servers --------------------
static void start_camera_server() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.ctrl_port   = 32768;
  config.lru_purge_enable = true;
  config.max_open_sockets = 2;

  httpd_uri_t index_uri   = { .uri="/",        .method=HTTP_GET, .handler=index_handler,   .user_ctx=NULL };
  httpd_uri_t capture_uri = { .uri="/capture", .method=HTTP_GET, .handler=capture_handler, .user_ctx=NULL };
  httpd_uri_t control_uri = { .uri="/control", .method=HTTP_GET, .handler=control_handler, .user_ctx=NULL };
  httpd_uri_t led_uri     = { .uri="/led",     .method=HTTP_GET, .handler=led_handler,     .user_ctx=NULL };
  httpd_uri_t profile_uri = { .uri="/profile", .method=HTTP_GET, .handler=profile_handler, .user_ctx=NULL };

  esp_err_t err = httpd_start(&camera_httpd, &config);
  if (err == ESP_OK) {
    httpd_register_uri_handler(camera_httpd, &index_uri);
    httpd_register_uri_handler(camera_httpd, &capture_uri);
    httpd_register_uri_handler(camera_httpd, &control_uri);
    httpd_register_uri_handler(camera_httpd, &led_uri);
    httpd_register_uri_handler(camera_httpd, &profile_uri);
    Serial.println("HTTP server started on port 80");
  } else {
    Serial.printf("Failed to start HTTP server on port 80: %d\n", (int)err);
  }

  httpd_config_t stream_config = HTTPD_DEFAULT_CONFIG();
  stream_config.server_port = 81;
  stream_config.ctrl_port   = 32769;
  stream_config.lru_purge_enable = true;
  stream_config.max_open_sockets = 1; // enforce 1 stream client

  httpd_uri_t stream_uri = { .uri="/stream", .method=HTTP_GET, .handler=stream_handler, .user_ctx=NULL };

  err = httpd_start(&stream_httpd, &stream_config);
  if (err == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    Serial.println("Stream server started on port 81");
  } else {
    Serial.printf("Failed to start stream server on port 81: %d\n", (int)err);
  }
}

void setup() {
  Serial.begin(115200);
  delay(200);

  pinMode(FLASH_LED_GPIO, OUTPUT);
  digitalWrite(FLASH_LED_GPIO, LOW);

  g_cam_mutex = xSemaphoreCreateMutex();
  if (!g_cam_mutex) {
    Serial.println("ERROR: Failed to create camera mutex");
  }

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Start in smooth stream profile
  config.frame_size   = STREAM_FRAMESIZE;
  config.jpeg_quality = STREAM_QUALITY;

  // Conservative settings without relying on PSRAM menu
  config.fb_count  = 1;
  config.grab_mode = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed 0x%x\n", err);
    while (true) delay(1000);
  }

  sensor_t * s = esp_camera_sensor_get();
  s->set_saturation(s, -1);
  s->set_contrast(s, 1);

  g_stream_delay_ms = STREAM_DELAY_MS;

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("WiFi connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());

  start_camera_server();

  Serial.println("Endpoints:");
  Serial.println("  http://<IP>/");
  Serial.println("  http://<IP>/profile?mode=stream|hq2|capture");
  Serial.println("  http://<IP>/capture");
  Serial.println("  http://<IP>/led?state=on|off|toggle");
  Serial.println("  http://<IP>:81/stream");
}

void loop() {
  delay(1000);
}
