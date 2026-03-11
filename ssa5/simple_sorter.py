#!/usr/bin/env python3
"""
SIMPLE SCREW SORTER - PYTHON CONTROL
"""

import cv2
import numpy as np
from ultralytics import YOLO
import requests
import time
from flask import Flask, render_template_string, jsonify, Response
import threading

# ==================== CONFIG ====================
ESP32_IP = "192.168.0.202"  # Change to your ESP32 IP
YOLO_MODEL = "best.pt"       # Your trained model

# Coordinate grid for bins (x, y in pixels from center)
# These are pixel offsets - you'll calibrate these
BIN_POSITIONS = {
    'M3': (100, 80),    # Top right
    'M4': (100, 0),     # Middle right
    'M5': (100, -80),   # Bottom right
    'M6': (50, -100)    # Bottom center
}

# Servo positions for each bin (microseconds)
# You'll need to calibrate these by testing
SERVO_PRESETS = {
    'M3': [1700, 1300, 1600, 800],  # Base, Shoulder, Elbow, Gripper
    'M4': [1500, 1300, 1600, 800],
    'M5': [1300, 1300, 1600, 800],
    'M6': [1200, 1300, 1600, 800],
    'home': [1500, 1500, 1500, 1500],
    'pickup': [1500, 1200, 1700, 1500]  # Position to grab screw
}

# ==================== YOLO DETECTOR ====================

class SimpleDetector:
    def __init__(self, model_path):
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        print("Model loaded!")
        
        self.latest_image = None
        self.latest_detections = []
    
    def detect(self, image, conf=0.6):
        """Run YOLO detection."""
        results = self.model.predict(image, conf=conf, verbose=False)
        
        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                gauge = ['M3', 'M4', 'M5', 'M6'][cls]
                
                detections.append({
                    'gauge': gauge,
                    'confidence': confidence,
                    'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                    'center': (cx, cy)
                })
        
        detections.sort(key=lambda d: d['confidence'], reverse=True)
        return detections
    
    def draw_detections(self, image, detections):
        """Draw boxes and labels."""
        colors = {
            'M3': (0, 0, 255),
            'M4': (0, 165, 255),
            'M5': (0, 255, 0),
            'M6': (255, 0, 0)
        }
        
        vis = image.copy()
        
        for det in detections:
            x, y, w, h = det['bbox']
            color = colors[det['gauge']]
            
            cv2.rectangle(vis, (x, y), (x+w, y+h), color, 3)
            
            label = f"{det['gauge']} {det['confidence']:.0%}"
            cv2.putText(vis, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw center point
            cx, cy = det['center']
            cv2.circle(vis, (cx, cy), 5, color, -1)
        
        return vis

# ==================== ESP32 INTERFACE ====================

class ESP32:
    def __init__(self, ip):
        self.base_url = f"http://{ip}"
        print(f"Connecting to ESP32 at {ip}...")
        
        # Test connection
        try:
            r = requests.get(f"{self.base_url}/api/status", timeout=2)
            print("✓ ESP32 connected")
        except:
            print("✗ Cannot connect to ESP32!")
            print(f"   Make sure ESP32 is at {ip}")
            exit(1)
    
    def capture_image(self):
        """Get high-res image for YOLO."""
        try:
            r = requests.get(f"{self.base_url}/api/capture", timeout=5)
            if r.status_code == 200:
                nparr = np.frombuffer(r.content, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return image
        except Exception as e:
            print(f"Capture failed: {e}")
        return None
    
    def set_servo(self, channel, value):
        """Set servo position."""
        try:
            requests.get(f"{self.base_url}/api/servo?ch={channel}&val={value}", timeout=1)
        except:
            pass
    
    def move_to_preset(self, preset_name):
        """Move arm to preset position."""
        if preset_name in SERVO_PRESETS:
            positions = SERVO_PRESETS[preset_name]
            print(f"Moving to {preset_name}: {positions}")
            for ch, val in enumerate(positions):
                self.set_servo(ch, val)
                time.sleep(0.05)
            time.sleep(1)  # Wait for movement
    
    def home(self):
        """Home the arm."""
        self.move_to_preset('home')

# ==================== MAIN SORTER ====================

class SimpleSorter:
    def __init__(self):
        print("\n=== SIMPLE SCREW SORTER ===\n")
        
        self.esp32 = ESP32(ESP32_IP)
        self.detector = SimpleDetector(YOLO_MODEL)
        
        self.stats = {'M3': 0, 'M4': 0, 'M5': 0, 'M6': 0, 'total': 0}
        
        print("\n✓ System ready!\n")
    
    def process_one(self):
        """Detect and sort one screw - SIMPLE version."""
        print("\n" + "="*50)
        print("PROCESSING SCREW...")
        print("="*50)
        
        # Capture
        print("[1/4] Capturing...")
        image = self.esp32.capture_image()
        if image is None:
            print("✗ Failed")
            return False
        
        print(f"✓ Got {image.shape[1]}x{image.shape[0]} image")
        self.detector.latest_image = image
        
        # Detect
        print("[2/4] Running YOLO...")
        detections = self.detector.detect(image)
        self.detector.latest_detections = detections
        
        if not detections:
            print("✗ No screws found")
            return False
        
        det = detections[0]
        print(f"✓ Found {det['gauge']} ({det['confidence']:.0%})")
        
        # Pick
        print("[3/4] Moving arm to pickup...")
        self.esp32.move_to_preset('pickup')
        
        # Close gripper
        self.esp32.set_servo(3, 800)  # Close
        time.sleep(0.5)
        
        # Sort
        gauge = det['gauge']
        print(f"[4/4] Sorting to {gauge} bin...")
        self.esp32.move_to_preset(gauge)
        
        # Open gripper
        self.esp32.set_servo(3, 1500)  # Open
        time.sleep(0.5)
        
        # Home
        self.esp32.home()
        
        # Update stats
        self.stats[gauge] += 1
        self.stats['total'] += 1
        
        print(f"\n✅ SUCCESS! Sorted {gauge} (Total: {self.stats['total']})")
        print("="*50)
        
        return True
    
    def run_auto(self, delay=2):
        """Continuous sorting."""
        print("\n🚀 AUTO MODE")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.process_one()
                time.sleep(delay)
        except KeyboardInterrupt:
            print("\n\n⏸ Stopped")
            self.show_stats()
    
    def show_stats(self):
        """Show statistics."""
        print("\n" + "="*50)
        print("STATISTICS")
        print("="*50)
        print(f"Total: {self.stats['total']}")
        print(f"  M3: {self.stats['M3']}")
        print(f"  M4: {self.stats['M4']}")
        print(f"  M5: {self.stats['M5']}")
        print(f"  M6: {self.stats['M6']}")
        print("="*50)

# ==================== WEB UI ====================

app = Flask(__name__)
sorter = None

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Screw Sorter - Python Control</title>
    <style>
        body { font-family: Arial; background: #667eea; margin: 0; padding: 20px; }
        .container { max-width: 1000px; margin: 0 auto; background: white; 
                     padding: 30px; border-radius: 10px; }
        h1 { color: #667eea; }
        .controls { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; 
                    margin: 20px 0; }
        button { padding: 15px; border: none; border-radius: 5px; font-size: 16px; 
                 font-weight: bold; cursor: pointer; background: #667eea; color: white; }
        button:hover { background: #5568d3; }
        #camera { width: 100%; border-radius: 5px; margin: 20px 0; }
        .stats { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }
        .stat { text-align: center; padding: 15px; background: #f5f5f5; border-radius: 5px; }
        .stat-value { font-size: 32px; font-weight: bold; color: #667eea; }
        .stat-label { font-size: 12px; color: #666; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔩 Screw Sorter - Python Control</h1>
        
        <div class="controls">
            <button onclick="processOne()">📸 Process One Screw</button>
            <button onclick="startAuto()">▶ Start Auto Mode</button>
            <button onclick="stopAuto()">⏸ Stop Auto</button>
        </div>
        
        <img id="camera" src="/video">
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value" id="total">0</div>
                <div class="stat-label">Total</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="m3">0</div>
                <div class="stat-label">M3</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="m4">0</div>
                <div class="stat-label">M4</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="m5">0</div>
                <div class="stat-label">M5</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="m6">0</div>
                <div class="stat-label">M6</div>
            </div>
        </div>
    </div>
    
    <script>
        let autoMode = false;
        
        async function processOne() {
            await fetch('/api/process', { method: 'POST' });
            updateStats();
        }
        
        async function startAuto() {
            autoMode = true;
            while (autoMode) {
                await processOne();
                await new Promise(r => setTimeout(r, 2000));
            }
        }
        
        function stopAuto() {
            autoMode = false;
        }
        
        async function updateStats() {
            const r = await fetch('/api/stats');
            const stats = await r.json();
            document.getElementById('total').textContent = stats.total;
            document.getElementById('m3').textContent = stats.M3;
            document.getElementById('m4').textContent = stats.M4;
            document.getElementById('m5').textContent = stats.M5;
            document.getElementById('m6').textContent = stats.M6;
        }
        
        setInterval(updateStats, 1000);
    </script>
</body>
</html>
    ''')

@app.route('/video')
def video():
    """Stream with YOLO detections."""
    def generate():
        while True:
            if sorter.detector.latest_image is not None:
                img = sorter.detector.draw_detections(
                    sorter.detector.latest_image,
                    sorter.detector.latest_detections
                )
                _, buffer = cv2.imencode('.jpg', img)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + 
                       buffer.tobytes() + b'\r\n')
            time.sleep(0.5)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/process', methods=['POST'])
def api_process():
    """Process one screw."""
    threading.Thread(target=sorter.process_one, daemon=True).start()
    return jsonify({'ok': 1})

@app.route('/api/stats')
def api_stats():
    """Get statistics."""
    return jsonify(sorter.stats)

# ==================== MAIN ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" SIMPLE SCREW SORTER - QUICK SETUP")
    print("="*60 + "\n")
    
    sorter = SimpleSorter()
    
    print("\nCHOOSE MODE:")
    print("1. Manual - Process one screw at a time")
    print("2. Auto - Continuous sorting")
    print("3. Web UI - Control via browser")
    print()
    
    choice = input("Enter 1, 2, or 3: ").strip()
    
    if choice == '1':
        # Manual mode
        while True:
            input("\nPress Enter to process a screw (Ctrl+C to quit)...")
            sorter.process_one()
    
    elif choice == '2':
        # Auto mode
        sorter.run_auto()
    
    elif choice == '3':
        # Web UI
        print("\n🌐 Starting web UI...")
        print("Open: http://localhost:5000")
        print()
        app.run(host='0.0.0.0', port=5000, debug=False)
    
    else:
        print("Invalid choice!")
