import sys
import cv2
import serial
import time
import numpy as np                          # ← added for grabbing math
from ultralytics import YOLO
try:
    import psutil
except Exception:
    psutil = None
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

# Serial
ser = serial.Serial('COM3',115200)
time.sleep(2)

# YOLO — enable CUDA acceleration if available
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("best.pt", task="detect")
model.to(device)

# ── Pose model for grabbing logic ────────────────────────────── ← added
pose_model = YOLO("yolov8n-pose.pt")
pose_model.to(device)

cap = cv2.VideoCapture(1)

servo_x = 90
servo_y = 90

fire_start_time = 0
fire_visible = False
threat_confirmed = False   # latches True once person+weapon seen together


# ── Grabbing helper (from model1_video_v3) ───────────────────── ← added
def is_grabbing(wrist_point, weapon_box, padding=15):
    """Return True when a wrist keypoint is inside (or very near) a weapon bbox."""
    wx, wy = wrist_point
    bx1, by1, bx2, by2 = weapon_box
    return (bx1 - padding) < wx < (bx2 + padding) and \
           (by1 - padding) < wy < (by2 + padding)


class AimSense(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("AimSense — Defense System")
        self.setGeometry(100,100,1400,750)

        self.initUI()

        self.last_cx = None
        self.last_cy = None
        self.last_box_h = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def initUI(self):

        main = QWidget()
        main.setStyleSheet("background-color:black;")
        self.setCentralWidget(main)

        layout = QHBoxLayout()
        layout.setContentsMargins(10,10,10,10)

        # ---------------- LIVE FEED PANEL ----------------

        video_layout = QVBoxLayout()

        title = QLabel("LIVE FEED")
        title.setStyleSheet("color:#888;font-size:14px")

        self.video = QLabel()
        self.video.setStyleSheet("background-color:black;")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)

        video_layout.addWidget(title)
        video_layout.addWidget(self.video)

        # ---------------- RIGHT PANEL ----------------

        side = QVBoxLayout()

        telemetry_box = QWidget()
        telemetry_box.setStyleSheet("background:#0b0f1a; padding:15px;")
        tele_layout = QVBoxLayout()

        title_tel = QLabel("TELEMETRY")
        title_tel.setStyleSheet("color:#888;font-size:14px")

        self.status_label = QLabel("STATUS    SCANNING")
        self.sector_label = QLabel("SECTOR    --")
        self.range_label = QLabel("RANGE     --")
        self.weapon_label = QLabel("WEAPON    NONE")
        self.device_label = QLabel("DEVICE    --")

        for lbl in (self.status_label, self.sector_label, self.range_label, self.weapon_label, self.device_label):
            lbl.setStyleSheet("color:#00ff90; font-family:monospace;")

        tele_layout.addWidget(title_tel)
        tele_layout.addWidget(self.status_label)
        tele_layout.addWidget(self.sector_label)
        tele_layout.addWidget(self.range_label)
        tele_layout.addWidget(self.weapon_label)
        tele_layout.addWidget(self.device_label)

        telemetry_box.setLayout(tele_layout)

        self.alert = QLabel("ALERT\nNo active alerts")
        self.alert.setStyleSheet("""
        color:white;
        background:#400000;
        padding:15px;
        """)

        self.engage = QPushButton("ENGAGE")

        # Abort button
        self.abort_btn = QPushButton("CLEAR / ABORT")
        self.abort_btn.setStyleSheet("background:#101522;color:#ff4444;height:40px;font-weight:bold;")
        self.abort_btn.clicked.connect(self.abort_action)

        # connect buttons to send_fire
        try:
            self.engage.clicked.connect(self.send_fire)
            self.fire_btn.clicked.connect(self.send_fire)
        except Exception:
            pass

        side.addWidget(telemetry_box)
        side.addWidget(self.alert)
        side.addWidget(self.engage)
        side.addWidget(self.abort_btn)
        side.addStretch()

        layout.addLayout(video_layout,4)
        layout.addLayout(side,1)

        main.setLayout(layout)

    # -------------------------------------------------

    def update_frame(self):

        global servo_x,servo_y,fire_visible,fire_start_time,threat_confirmed

        ret, frame = cap.read()
        if not ret:
            return

        person_present = False
        weapon_present = False

        h,w,_ = frame.shape

        # GRID
        for i in range(1,3):
            cv2.line(frame,(0,i*h//3),(w,i*h//3),(80,80,80),1)
            cv2.line(frame,(i*w//3,0),(i*w//3,h),(80,80,80),1)

        results = model(frame,conf=0.5)
        boxes = results[0].boxes

        
        weapon_boxes = []
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                if label == "weapon":
                    weapon_boxes.append(list(map(int, box.xyxy[0])))

        # ── Run pose model to get wrist keypoints ─────────────── ← added
        grabbing_confirmed = False
        pose_results = pose_model(frame, verbose=False,
                                  device=device,
                                  half=(device == "cuda"))
        for r in pose_results:
            if r.keypoints is None:
                continue
            kps_data = r.keypoints.data.cpu().numpy()   # shape (N, 17, 3)
            for kp in kps_data:
                left_wrist  = kp[9][:2]    # keypoint index 9  = left wrist
                right_wrist = kp[10][:2]   # keypoint index 10 = right wrist
                lw_conf     = kp[9][2]
                rw_conf     = kp[10][2]
                for wbox in weapon_boxes:
                    grab_l = (lw_conf > 0.5) and is_grabbing(left_wrist,  wbox)
                    grab_r = (rw_conf > 0.5) and is_grabbing(right_wrist, wbox)
                    if grab_l or grab_r:
                        grabbing_confirmed = True
                        break
                if grabbing_confirmed:
                    break

        if boxes is not None:

            for box in boxes:

                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                x1,y1,x2,y2 = map(int,box.xyxy[0])

                cx = (x1+x2)//2
                cy = (y1+y2)//2

                if label=="person":

                    person_present=True

                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                    center_x=w//2
                    center_y=h//2

                    if cx < center_x-40:
                        servo_x+=2
                    elif cx > center_x+40:
                        servo_x-=2

                    if cy < center_y-40:
                        servo_y-=2
                    elif cy > center_y+40:
                        servo_y+=2

                    servo_x=max(0,min(180,servo_x))
                    servo_y=max(0,min(180,servo_y))

                    data=f"{servo_x},{servo_y}\n"
                    ser.write(data.encode())

                    self.last_cx = cx
                    self.last_cy = cy
                    self.last_box_h = (y2 - y1)

                if label=="weapon":
                    # ── weapon_present now requires wrist-grab confirmation ── ← changed
                    if grabbing_confirmed:
                        weapon_present=True
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)


        # Update telemetry live data
        try:
            if person_present:
                self.status_label.setText(f"STATUS    {'LOCKED' if weapon_present else 'TRACKING'}")
            else:
                self.status_label.setText("STATUS    SCANNING")

            if self.last_cx is not None and self.last_cy is not None:
                row = min(2, self.last_cy // (h//3))
                col = min(2, self.last_cx // (w//3))
                sector = row*3 + col + 1
                self.sector_label.setText(f"SECTOR    {sector}")
            else:
                self.sector_label.setText("SECTOR    --")

            if self.last_box_h:
                size_pct = int(self.last_box_h / h * 100)
                self.range_label.setText(f"SIZE     {size_pct}%")
            else:
                self.range_label.setText("RANGE     --")

            self.weapon_label.setText(f"WEAPON    {'DETECTED' if weapon_present else 'NONE'}")

            if psutil:
                cpu = psutil.cpu_percent(interval=None)
                self.device_label.setText(f"DEVICE    GPU ({device.upper()})  CPU {cpu}%")
            else:
                self.device_label.setText(f"DEVICE    {device.upper()}")
        except Exception:
            pass

        # FIRE LOGIC

        # Latch threat_confirmed the moment person + weapon are seen together.
        # Once latched it stays True until operator presses CLEAR / ABORT.
        if person_present and weapon_present:
            threat_confirmed = True
            fire_visible = True
            fire_start_time = time.time()

        # Keep ENGAGE armed whenever a confirmed threat is still being tracked.
        if threat_confirmed and person_present:
            try:
                self.engage.setEnabled(True)
                self.engage.setStyleSheet("background:#b30000;color:white;height:40px")
            except Exception:
                pass
        elif not threat_confirmed:
            try:
                self.engage.setEnabled(False)
                self.engage.setStyleSheet("background:#101522;color:gray;height:40px")
            except Exception:
                pass

        if fire_visible:

            elapsed=(time.time()-fire_start_time)*1000

            if elapsed<500:
                cv2.rectangle(frame,(500,20),(620,70),(0,0,255),-1)
                cv2.putText(frame,"FIRE",(525,55),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            else:
                fire_visible=False

        # ------------ FIT VIDEO PERFECTLY ------------

        label_w = self.video.width()
        label_h = self.video.height()

        frame = cv2.resize(frame,(label_w,label_h))

        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        img = QImage(rgb.data,label_w,label_h,3*label_w,QImage.Format_RGB888)

        self.video.setPixmap(QPixmap.fromImage(img))

    def send_fire(self):
        try:
            ser.write(b"FIRE\n")
            try:
                self.alert.setText("ALERT\nFIRE SENT")
            except Exception:
                pass
        except Exception:
            try:
                self.alert.setText("ALERT\nFIRE FAILED")
            except Exception:
                pass

    def abort_action(self):
        """Reset fire state, disable engage, and send ABORT to Arduino."""
        global fire_visible, fire_start_time, servo_x, servo_y, threat_confirmed
        try:
            # Reset fire state
            fire_visible = False
            fire_start_time = 0
            threat_confirmed = False   # clear the threat latch

            # Reset servos to neutral center position
            servo_x = 90
            servo_y = 90
            ser.write(b"90,90\n")

            # Reset UI
            self.engage.setEnabled(False)
            self.engage.setStyleSheet("background:#101522;color:gray;height:40px")
            self.alert.setText("ALERT\nABORTED — System reset")
            self.alert.setStyleSheet("color:white;background:#444400;padding:15px;")
            self.status_label.setText("STATUS    SCANNING")
            self.weapon_label.setText("WEAPON    NONE")
            self.sector_label.setText("SECTOR    --")
            self.range_label.setText("RANGE     --")

            # Clear tracking memory
            self.last_cx = None
            self.last_cy = None
            self.last_box_h = None

            try:
                self.fire_btn.setVisible(False)
            except Exception:
                pass

            # Brief delay then restore alert style
            QTimer.singleShot(2000, self._restore_alert)

        except Exception:
            self.alert.setText("ALERT\nABORT FAILED")

    def _restore_alert(self):
        self.alert.setText("ALERT\nNo active alerts")
        self.alert.setStyleSheet("color:white;background:#400000;padding:15px;")


app = QApplication(sys.argv)

window = AimSense()
window.show()

sys.exit(app.exec_())
