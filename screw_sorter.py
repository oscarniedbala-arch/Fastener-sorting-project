#!/usr/bin/env python3
"""
Automated Screw Sorting System
===============================

Complete pipeline:
1. ESP32-CAM captures image
2. YOLO detects and classifies screws
3. EEZYbotARM MK2 picks and sorts screws
4. Web UI shows live progress

Usage:
    python screw_sorter.py --model best.pt --esp32 192.168.0.202 --arm COM3
"""

import cv2
import numpy as np
from ultralytics import YOLO
import requests
import serial
import time
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import math
import threading
from flask import Flask, render_template_string, jsonify, Response
import queue


# ==================== CONFIGURATION ====================

@dataclass
class ArmConfig:
    """EEZYbotARM MK2 configuration."""
    # Physical dimensions (mm)
    base_height: float = 60.0      # Height from base to shoulder
    shoulder_length: float = 135.0  # Upper arm length
    elbow_length: float = 147.0     # Forearm length
    gripper_length: float = 85.0    # Gripper offset
    
    # Servo limits (degrees)
    base_min: float = 0.0
    base_max: float = 180.0
    shoulder_min: float = 20.0
    shoulder_max: float = 160.0
    elbow_min: float = 0.0
    elbow_max: float = 180.0
    gripper_min: float = 20.0      # Open
    gripper_max: float = 90.0      # Closed
    
    # Home position (degrees)
    home_base: float = 90.0
    home_shoulder: float = 90.0
    home_elbow: float = 90.0
    home_gripper: float = 20.0


@dataclass
class SortingZones:
    """Sorting bin positions in 3D space (mm)."""
    # Relative to arm base center
    M3_position: Tuple[float, float, float] = (150.0, 100.0, 10.0)   # (x, y, z)
    M4_position: Tuple[float, float, float] = (150.0, 0.0, 10.0)
    M5_position: Tuple[float, float, float] = (150.0, -100.0, 10.0)
    M6_position: Tuple[float, float, float] = (100.0, -150.0, 10.0)
    
    # Pickup area (center of camera view)
    pickup_area: Tuple[float, float, float] = (200.0, 0.0, 0.0)


@dataclass
class Detection:
    """YOLO detection result."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center_px: Tuple[int, int]
    center_mm: Optional[Tuple[float, float]] = None


# ==================== EEZYBOT ARM CONTROLLER ====================

class EEZYbotARM:
    """EEZYbotARM MK2 controller with inverse kinematics."""
    
    def __init__(self, port: str, config: ArmConfig = None):
        self.config = config or ArmConfig()
        self.serial = serial.Serial(port, 115200, timeout=1)
        time.sleep(2)  # Wait for Arduino reset
        
        # Current position
        self.current_angles = [
            self.config.home_base,
            self.config.home_shoulder,
            self.config.home_elbow,
            self.config.home_gripper
        ]
        
        print(f"✓ EEZYbotARM connected on {port}")
        self.home()
    
    def send_command(self, base: float, shoulder: float, elbow: float, gripper: float, speed: int = 50):
        """Send servo positions to Arduino."""
        # Clamp values
        base = np.clip(base, self.config.base_min, self.config.base_max)
        shoulder = np.clip(shoulder, self.config.shoulder_min, self.config.shoulder_max)
        elbow = np.clip(elbow, self.config.elbow_min, self.config.elbow_max)
        gripper = np.clip(gripper, self.config.gripper_min, self.config.gripper_max)
        
        # Send to Arduino: "MOVE,base,shoulder,elbow,gripper,speed\n"
        cmd = f"MOVE,{int(base)},{int(shoulder)},{int(elbow)},{int(gripper)},{speed}\n"
        self.serial.write(cmd.encode())
        
        # Update current position
        self.current_angles = [base, shoulder, elbow, gripper]
        
        # Wait for move to complete
        time.sleep(0.05 * speed)  # Approximate timing
    
    def inverse_kinematics(self, x: float, y: float, z: float) -> Optional[Tuple[float, float, float]]:
        """
        Calculate joint angles for target position.
        
        Args:
            x, y, z: Target position in mm (relative to base)
        
        Returns:
            (base_angle, shoulder_angle, elbow_angle) or None if unreachable
        """
        # Base angle (rotation around vertical axis)
        base_angle = math.degrees(math.atan2(y, x))
        
        # Distance in horizontal plane
        r = math.sqrt(x**2 + y**2)
        
        # Adjust z for base height
        z_adj = z - self.config.base_height
        
        # Distance from shoulder to target
        d = math.sqrt(r**2 + z_adj**2)
        
        # Check if reachable
        L1 = self.config.shoulder_length
        L2 = self.config.elbow_length + self.config.gripper_length
        
        if d > (L1 + L2) or d < abs(L1 - L2):
            print(f"⚠️ Target ({x}, {y}, {z}) unreachable (d={d:.1f}mm)")
            return None
        
        # Elbow angle (using law of cosines)
        cos_elbow = (L1**2 + L2**2 - d**2) / (2 * L1 * L2)
        cos_elbow = np.clip(cos_elbow, -1.0, 1.0)
        elbow_angle = 180 - math.degrees(math.acos(cos_elbow))
        
        # Shoulder angle
        alpha = math.atan2(z_adj, r)
        beta = math.acos((L1**2 + d**2 - L2**2) / (2 * L1 * d))
        shoulder_angle = 90 - math.degrees(alpha + beta)
        
        return (base_angle, shoulder_angle, elbow_angle)
    
    def move_to(self, x: float, y: float, z: float, speed: int = 50):
        """Move arm to target position using IK."""
        angles = self.inverse_kinematics(x, y, z)
        
        if angles is None:
            print(f"❌ Cannot reach ({x}, {y}, {z})")
            return False
        
        base, shoulder, elbow = angles
        gripper = self.current_angles[3]  # Keep current gripper state
        
        print(f"Moving to ({x:.1f}, {y:.1f}, {z:.1f}) -> angles: ({base:.1f}°, {shoulder:.1f}°, {elbow:.1f}°)")
        
        self.send_command(base, shoulder, elbow, gripper, speed)
        return True
    
    def home(self):
        """Move to home position."""
        print("Moving to home position")
        self.send_command(
            self.config.home_base,
            self.config.home_shoulder,
            self.config.home_elbow,
            self.config.home_gripper,
            speed=30
        )
        time.sleep(1)
    
    def open_gripper(self):
        """Open gripper."""
        angles = self.current_angles.copy()
        angles[3] = self.config.gripper_min
        self.send_command(*angles, speed=30)
        time.sleep(0.5)
    
    def close_gripper(self):
        """Close gripper."""
        angles = self.current_angles.copy()
        angles[3] = self.config.gripper_max
        self.send_command(*angles, speed=30)
        time.sleep(0.5)
    
    def pick_and_place(self, pickup: Tuple[float, float, float], 
                       dropoff: Tuple[float, float, float],
                       approach_height: float = 50.0):
        """
        Pick object from pickup location and place at dropoff.
        
        Args:
            pickup: (x, y, z) pickup position
            dropoff: (x, y, z) dropoff position
            approach_height: Height to approach from above
        """
        px, py, pz = pickup
        dx, dy, dz = dropoff
        
        print(f"🤖 Pick from ({px:.0f}, {py:.0f}, {pz:.0f}) → Place at ({dx:.0f}, {dy:.0f}, {dz:.0f})")
        
        # Open gripper
        self.open_gripper()
        
        # Move to approach position above pickup
        if not self.move_to(px, py, pz + approach_height, speed=40):
            return False
        
        # Lower to pickup
        if not self.move_to(px, py, pz, speed=20):
            return False
        
        # Close gripper
        self.close_gripper()
        
        # Lift
        self.move_to(px, py, pz + approach_height, speed=30)
        
        # Move to approach position above dropoff
        if not self.move_to(dx, dy, dz + approach_height, speed=40):
            return False
        
        # Lower to dropoff
        if not self.move_to(dx, dy, dz, speed=20):
            return False
        
        # Open gripper
        self.open_gripper()
        
        # Lift
        self.move_to(dx, dy, dz + approach_height, speed=30)
        
        # Return home
        self.home()
        
        return True
    
    def close(self):
        """Close serial connection."""
        self.home()
        time.sleep(1)
        self.serial.close()


# ==================== ESP32 INTERFACE ====================

class ESP32Camera:
    """Interface to ESP32-CAM."""
    
    def __init__(self, ip: str):
        self.ip = ip
        self.base_url = f"http://{ip}"
        
        # Test connection
        try:
            response = requests.get(f"{self.base_url}/api/status", timeout=2)
            print(f"✓ ESP32 connected at {ip}")
        except:
            print(f"⚠️ Cannot connect to ESP32 at {ip}")
    
    def capture_image(self) -> Optional[np.ndarray]:
        """Capture image from ESP32."""
        try:
            response = requests.get(f"{self.base_url}/stream", timeout=5)
            
            if response.status_code == 200:
                # Decode JPEG
                nparr = np.frombuffer(response.content, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return image
            
        except Exception as e:
            print(f"❌ Capture failed: {e}")
        
        return None
    
    def trigger_capture(self):
        """Trigger capture (for synchronization)."""
        try:
            requests.post(f"{self.base_url}/api/trigger", timeout=2)
        except:
            pass


# ==================== YOLO DETECTOR ====================

class ScrewDetector:
    """YOLO-based screw detector."""
    
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        print(f"✓ YOLO model loaded: {model_path}")
        
        # Camera calibration (pixels per mm at working distance)
        # You'll need to calibrate this based on your setup
        self.px_per_mm = 2.5  # Adjust based on your camera height
    
    def detect(self, image: np.ndarray, conf_threshold: float = 0.6) -> List[Detection]:
        """
        Detect screws in image.
        
        Returns:
            List of Detection objects
        """
        results = self.model.predict(image, conf=conf_threshold, verbose=False)
        
        detections = []
        
        for r in results:
            boxes = r.boxes
            
            for box in boxes:
                # Extract info
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Calculate center
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                w = int(x2 - x1)
                h = int(y2 - y1)
                
                # Get class name
                class_name = ['M3', 'M4', 'M5', 'M6'][cls]
                
                det = Detection(
                    class_name=class_name,
                    confidence=conf,
                    bbox=(int(x1), int(y1), w, h),
                    center_px=(cx, cy)
                )
                
                detections.append(det)
        
        # Sort by confidence
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        return detections
    
    def pixel_to_mm(self, px: Tuple[int, int], image_shape: Tuple[int, int]) -> Tuple[float, float]:
        """
        Convert pixel coordinates to mm (relative to image center).
        
        Args:
            px: (x, y) in pixels
            image_shape: (height, width)
        
        Returns:
            (x_mm, y_mm) relative to center
        """
        h, w = image_shape
        center_x, center_y = w // 2, h // 2
        
        # Offset from center in pixels
        dx_px = px[0] - center_x
        dy_px = px[1] - center_y
        
        # Convert to mm
        x_mm = dx_px / self.px_per_mm
        y_mm = -dy_px / self.px_per_mm  # Invert y (image y increases downward)
        
        return (x_mm, y_mm)
    
    def annotate_image(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detections on image."""
        vis = image.copy()
        
        colors = {
            'M3': (0, 0, 255),
            'M4': (0, 165, 255),
            'M5': (0, 255, 0),
            'M6': (255, 0, 0)
        }
        
        for det in detections:
            x, y, w, h = det.bbox
            color = colors.get(det.class_name, (255, 255, 255))
            
            # Box
            cv2.rectangle(vis, (x, y), (x+w, y+h), color, 3)
            
            # Label
            label = f"{det.class_name} {det.confidence:.0%}"
            cv2.putText(vis, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, color, 2)
            
            # Center point
            cx, cy = det.center_px
            cv2.circle(vis, (cx, cy), 5, color, -1)
        
        return vis


# ==================== SORTING SYSTEM ====================

class ScrewSortingSystem:
    """Complete automated screw sorting system."""
    
    def __init__(self, 
                 model_path: str,
                 esp32_ip: str,
                 arm_port: str,
                 zones: SortingZones = None):
        
        self.detector = ScrewDetector(model_path)
        self.camera = ESP32Camera(esp32_ip)
        self.arm = EEZYbotARM(arm_port)
        self.zones = zones or SortingZones()
        
        # Statistics
        self.stats = {
            'total_sorted': 0,
            'M3': 0,
            'M4': 0,
            'M5': 0,
            'M6': 0,
            'errors': 0
        }
        
        # Web UI queue
        self.ui_queue = queue.Queue(maxsize=10)
        
        print("\n" + "="*60)
        print("✓ SCREW SORTING SYSTEM READY")
        print("="*60)
    
    def process_single_screw(self, verify: bool = False) -> bool:
        """
        Detect and sort one screw.
        
        Args:
            verify: Move closer and re-detect before sorting
        
        Returns:
            True if successful, False otherwise
        """
        # Step 1: Capture image
        print("\n📸 Capturing image...")
        image = self.camera.capture_image()
        
        if image is None:
            print("❌ Capture failed")
            return False
        
        # Step 2: Detect screws
        print("🔍 Detecting screws...")
        detections = self.detector.detect(image, conf_threshold=0.6)
        
        if not detections:
            print("⚠️ No screws detected")
            return False
        
        # Take highest confidence detection
        detection = detections[0]
        
        print(f"✓ Detected: {detection.class_name} ({detection.confidence:.0%})")
        
        # Convert pixel position to mm
        x_mm, y_mm = self.detector.pixel_to_mm(detection.center_px, image.shape[:2])
        
        # Calculate pickup position (relative to arm base)
        # Assuming camera is mounted above pickup area
        pickup_x = self.zones.pickup_area[0] + x_mm
        pickup_y = self.zones.pickup_area[1] + y_mm
        pickup_z = 5.0  # Just above platform
        
        # Optional: Move closer for verification
        if verify:
            print("🔍 Moving closer for verification...")
            # Move camera/arm to get closer view
            # TODO: Implement close-up verification if needed
            pass
        
        # Step 3: Pick and place
        gauge = detection.class_name
        
        if gauge == 'M3':
            dropoff = self.zones.M3_position
        elif gauge == 'M4':
            dropoff = self.zones.M4_position
        elif gauge == 'M5':
            dropoff = self.zones.M5_position
        elif gauge == 'M6':
            dropoff = self.zones.M6_position
        else:
            print(f"❌ Unknown gauge: {gauge}")
            return False
        
        success = self.arm.pick_and_place(
            (pickup_x, pickup_y, pickup_z),
            dropoff
        )
        
        if success:
            self.stats['total_sorted'] += 1
            self.stats[gauge] += 1
            print(f"✅ {gauge} screw sorted! Total: {self.stats['total_sorted']}")
            
            # Send to web UI
            self.ui_queue.put({
                'type': 'detection',
                'gauge': gauge,
                'confidence': detection.confidence,
                'stats': self.stats.copy(),
                'image': self.detector.annotate_image(image, [detection])
            })
            
            return True
        else:
            self.stats['errors'] += 1
            print(f"❌ Pick and place failed")
            return False
    
    def run_continuous(self, delay: float = 2.0):
        """
        Run continuous sorting loop.
        
        Args:
            delay: Delay between cycles (seconds)
        """
        print("\n🚀 Starting continuous sorting...")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                success = self.process_single_screw(verify=False)
                
                if not success:
                    print(f"⏸ Waiting {delay}s...")
                
                time.sleep(delay)
                
        except KeyboardInterrupt:
            print("\n\n⏸ Sorting stopped by user")
            self.print_statistics()
    
    def print_statistics(self):
        """Print sorting statistics."""
        print("\n" + "="*60)
        print("SORTING STATISTICS")
        print("="*60)
        print(f"Total sorted: {self.stats['total_sorted']}")
        print(f"  M3: {self.stats['M3']}")
        print(f"  M4: {self.stats['M4']}")
        print(f"  M5: {self.stats['M5']}")
        print(f"  M6: {self.stats['M6']}")
        print(f"Errors: {self.stats['errors']}")
        print("="*60)
    
    def shutdown(self):
        """Shutdown system gracefully."""
        print("\n🛑 Shutting down...")
        self.arm.close()
        print("✓ Shutdown complete")


# ==================== WEB UI ====================

app = Flask(__name__)
sorter = None  # Will be set in main()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Screw Sorter Control Panel</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
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
            max-width: 1600px;
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
        
        .status-badge {
            padding: 10px 20px;
            border-radius: 20px;
            background: rgba(255,255,255,0.2);
            font-size: 16px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #4ade80;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            padding: 40px;
        }
        
        .video-panel {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .video-container {
            position: relative;
            background: #1f2937;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        #live-feed {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .controls {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }
        
        button {
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
        
        .btn-success {
            background: #10b981;
            color: white;
        }
        
        .btn-danger {
            background: #ef4444;
            color: white;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
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
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }
        
        .stat-box {
            text-align: center;
            padding: 20px;
            background: white;
            border-radius: 8px;
            border-left: 4px solid;
        }
        
        .stat-box.total { border-color: #667eea; }
        .stat-box.m3 { border-color: #ef4444; }
        .stat-box.m4 { border-color: #f59e0b; }
        .stat-box.m5 { border-color: #10b981; }
        .stat-box.m6 { border-color: #3b82f6; }
        
        .stat-value {
            font-size: 36px;
            font-weight: 700;
            color: #1f2937;
        }
        
        .stat-label {
            font-size: 14px;
            color: #6b7280;
            margin-top: 5px;
        }
        
        .current-detection {
            padding: 20px;
            background: white;
            border-radius: 8px;
            text-align: center;
        }
        
        .detection-gauge {
            font-size: 48px;
            font-weight: 700;
            margin-bottom: 10px;
        }
        
        .detection-confidence {
            font-size: 18px;
            color: #6b7280;
        }
        
        .log {
            max-height: 300px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            padding: 15px;
            background: white;
            border-radius: 8px;
        }
        
        .log-entry {
            padding: 5px 0;
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
            <h1>🤖 Automated Screw Sorter</h1>
            <div class="status-badge">
                <div class="status-indicator"></div>
                <span id="status">Ready</span>
            </div>
        </header>
        
        <div class="main-grid">
            <div class="video-panel">
                <div class="video-container">
                    <img id="live-feed" src="/video_feed" alt="Live Feed">
                </div>
                
                <div class="controls">
                    <button class="btn-primary" onclick="captureOne()">📸 Process One</button>
                    <button class="btn-success" onclick="startAuto()">▶ Start Auto</button>
                    <button class="btn-danger" onclick="stopAuto()">⏸ Stop</button>
                </div>
            </div>
            
            <div class="info-panel">
                <div class="card">
                    <h3>Current Detection</h3>
                    <div class="current-detection">
                        <div class="detection-gauge" id="current-gauge">--</div>
                        <div class="detection-confidence" id="current-conf">Waiting...</div>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Statistics</h3>
                    <div class="stats-grid">
                        <div class="stat-box total">
                            <div class="stat-value" id="stat-total">0</div>
                            <div class="stat-label">Total</div>
                        </div>
                        <div class="stat-box m3">
                            <div class="stat-value" id="stat-m3">0</div>
                            <div class="stat-label">M3</div>
                        </div>
                        <div class="stat-box m4">
                            <div class="stat-value" id="stat-m4">0</div>
                            <div class="stat-label">M4</div>
                        </div>
                        <div class="stat-box m5">
                            <div class="stat-value" id="stat-m5">0</div>
                            <div class="stat-label">M5</div>
                        </div>
                        <div class="stat-box m6">
                            <div class="stat-value" id="stat-m6">0</div>
                            <div class="stat-label">M6</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value" id="stat-errors">0</div>
                            <div class="stat-label">Errors</div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Activity Log</h3>
                    <div id="log" class="log"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function addLog(message) {
            const log = document.getElementById('log');
            const time = new Date().toLocaleTimeString();
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerHTML = `<span class="timestamp">${time}</span>${message}`;
            log.insertBefore(entry, log.firstChild);
            
            while (log.children.length > 50) {
                log.removeChild(log.lastChild);
            }
        }
        
        async function captureOne() {
            document.getElementById('status').textContent = 'Processing...';
            addLog('Manual capture triggered');
            
            try {
                const response = await fetch('/api/process_one', { method: 'POST' });
                const data = await response.json();
                
                if (data.success) {
                    addLog(`✓ Sorted ${data.gauge} screw`);
                } else {
                    addLog('⚠️ No screw found');
                }
            } catch (e) {
                addLog('❌ Error: ' + e.message);
            }
            
            document.getElementById('status').textContent = 'Ready';
        }
        
        async function startAuto() {
            document.getElementById('status').textContent = 'Auto Mode Active';
            addLog('Auto-sorting started');
            
            await fetch('/api/start_auto', { method: 'POST' });
        }
        
        async function stopAuto() {
            document.getElementById('status').textContent = 'Ready';
            addLog('Auto-sorting stopped');
            
            await fetch('/api/stop_auto', { method: 'POST' });
        }
        
        // Poll for updates
        setInterval(async () => {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                // Update stats
                document.getElementById('stat-total').textContent = data.stats.total_sorted;
                document.getElementById('stat-m3').textContent = data.stats.M3;
                document.getElementById('stat-m4').textContent = data.stats.M4;
                document.getElementById('stat-m5').textContent = data.stats.M5;
                document.getElementById('stat-m6').textContent = data.stats.M6;
                document.getElementById('stat-errors').textContent = data.stats.errors;
                
                // Update current detection
                if (data.current_detection) {
                    document.getElementById('current-gauge').textContent = data.current_detection.gauge;
                    document.getElementById('current-conf').textContent = 
                        `${(data.current_detection.confidence * 100).toFixed(1)}% confidence`;
                }
            } catch (e) {}
        }, 1000);
        
        addLog('System initialized');
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    """Stream video feed."""
    def generate():
        while True:
            image = sorter.camera.capture_image()
            
            if image is not None:
                # Detect and annotate
                detections = sorter.detector.detect(image, conf_threshold=0.6)
                annotated = sorter.detector.annotate_image(image, detections)
                
                # Encode as JPEG
                _, buffer = cv2.imencode('.jpg', annotated)
                frame = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            time.sleep(0.1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    """Get current status."""
    data = {
        'stats': sorter.stats,
        'current_detection': None
    }
    
    try:
        ui_data = sorter.ui_queue.get_nowait()
        if ui_data['type'] == 'detection':
            data['current_detection'] = {
                'gauge': ui_data['gauge'],
                'confidence': ui_data['confidence']
            }
    except queue.Empty:
        pass
    
    return jsonify(data)

@app.route('/api/process_one', methods=['POST'])
def api_process_one():
    """Process single screw."""
    success = sorter.process_single_screw()
    return jsonify({'success': success})

auto_sorting = False

@app.route('/api/start_auto', methods=['POST'])
def api_start_auto():
    """Start auto-sorting."""
    global auto_sorting
    auto_sorting = True
    
    def auto_sort_loop():
        while auto_sorting:
            sorter.process_single_screw()
            time.sleep(2)
    
    threading.Thread(target=auto_sort_loop, daemon=True).start()
    return jsonify({'status': 'started'})

@app.route('/api/stop_auto', methods=['POST'])
def api_stop_auto():
    """Stop auto-sorting."""
    global auto_sorting
    auto_sorting = False
    return jsonify({'status': 'stopped'})


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description="Automated Screw Sorting System")
    
    parser.add_argument("--model", required=True, help="Path to YOLO model (best.pt)")
    parser.add_argument("--esp32", required=True, help="ESP32 IP address")
    parser.add_argument("--arm", required=True, help="Arduino serial port (e.g., COM3 or /dev/ttyUSB0)")
    parser.add_argument("--manual", action="store_true", help="Manual mode (no auto-sorting)")
    parser.add_argument("--web-ui", action="store_true", help="Start web UI")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration routine")
    
    args = parser.parse_args()
    
    # Create system
    global sorter
    sorter = ScrewSortingSystem(
        model_path=args.model,
        esp32_ip=args.esp32,
        arm_port=args.arm
    )
    
    try:
        if args.calibrate:
            print("\n=== CALIBRATION MODE ===\n")
            print("Testing arm positions...")
            
            # Test each zone
            for gauge in ['M3', 'M4', 'M5', 'M6']:
                input(f"\nPress Enter to test {gauge} position...")
                
                if gauge == 'M3':
                    pos = sorter.zones.M3_position
                elif gauge == 'M4':
                    pos = sorter.zones.M4_position
                elif gauge == 'M5':
                    pos = sorter.zones.M5_position
                else:
                    pos = sorter.zones.M6_position
                
                sorter.arm.move_to(*pos)
                print(f"✓ Moved to {gauge} position: {pos}")
            
            sorter.arm.home()
            print("\n✓ Calibration complete")
        
        elif args.web_ui:
            print("\n🌐 Starting web UI at http://localhost:5000")
            app.run(host='0.0.0.0', port=5000, debug=False)
        
        elif args.manual:
            print("\n=== MANUAL MODE ===\n")
            print("Commands:")
            print("  1 - Process one screw")
            print("  s - Show statistics")
            print("  q - Quit")
            
            while True:
                cmd = input("\n> ").strip().lower()
                
                if cmd == '1':
                    sorter.process_single_screw()
                elif cmd == 's':
                    sorter.print_statistics()
                elif cmd == 'q':
                    break
        
        else:
            # Auto mode
            sorter.run_continuous()
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    
    finally:
        sorter.shutdown()


if __name__ == "__main__":
    main()
