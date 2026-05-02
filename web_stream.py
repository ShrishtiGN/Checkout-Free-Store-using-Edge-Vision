import cv2
import numpy as np
import random
from flask import Flask, Response

app = Flask(__name__)

def draw_3d_box(img, center_x, center_y, w, h, d, color=(0, 0, 255), thickness=2):
    hw, hh, hd = int(w/2), int(h/2), int(d/2)
    z_offset_x = int(hd * 0.5)
    z_offset_y = int(hd * 0.5)

    fl_t = (center_x - hw, center_y - hh)  
    fr_t = (center_x + hw, center_y - hh)  
    fl_b = (center_x - hw, center_y + hh)  
    fr_b = (center_x + hw, center_y + hh)  

    bl_t = (center_x - hw + z_offset_x, center_y - hh - z_offset_y)
    br_t = (center_x + hw + z_offset_x, center_y - hh - z_offset_y)
    bl_b = (center_x - hw + z_offset_x, center_y + hh - z_offset_y)
    br_b = (center_x + hw + z_offset_x, center_y + hh - z_offset_y)

    def draw_line(p1, p2):
        cv2.line(img, p1, p2, color, thickness)

    draw_line(fl_t, fr_t)
    draw_line(fr_t, fr_b)
    draw_line(fr_b, fl_b)
    draw_line(fl_b, fl_t)

    draw_line(bl_t, br_t)
    draw_line(br_t, br_b)
    draw_line(br_b, bl_b)
    draw_line(bl_b, bl_t)

    draw_line(fl_t, bl_t)
    draw_line(fr_t, br_t)
    draw_line(fl_b, bl_b)
    draw_line(fr_b, br_b)

def draw_transparent_containment_zone(img, bounds_2d, color=(255, 150, 0), alpha=0.3):
    overlay = img.copy()
    x_min, y_min, x_max, y_max = bounds_2d
    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(img, "CONTAINMENT FLOAT-BAG", (x_min + 10, y_min + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

class FloatingItemSimulator:
    def __init__(self, item_id, price):
        self.item_id = item_id
        self.price = price
        self.x = random.randint(300, 800)
        self.y = random.randint(100, 600)
        self.z = random.randint(20, 100)
        
        self.vx = random.uniform(-5.0, 5.0)
        self.vy = random.uniform(-5.0, 5.0)
        self.vz = random.uniform(-2.0, 2.0)
        self.v_rot = random.uniform(10.0, 45.0)
        
        self.w = 80
        self.h = 100
        self.d = 40
        self.billed = False
        
    def update(self, frame_w, frame_h):
        self.x += int(self.vx)
        self.y += int(self.vy)
        
        if self.x < 300 or self.x > frame_w - 50:
            self.vx *= -1
        if self.y < 50 or self.y > frame_h - 50:
            self.vy *= -1


items_sim = [
    FloatingItemSimulator(item_id=1, price=4.99),
    FloatingItemSimulator(item_id=2, price=1.50),
    FloatingItemSimulator(item_id=3, price=8.25)
]

def generate_frames():
    FRAME_W, FRAME_H = 1200, 800
    containment_bounds = (400, 200, 800, 600)

    while True:
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        draw_transparent_containment_zone(frame, containment_bounds)
        
        cv2.rectangle(frame, (0, 0), (300, FRAME_H), (30, 30, 30), -1)
        cv2.rectangle(frame, (0, 0), (300, FRAME_H), (100, 100, 100), 2)
        cv2.putText(frame, "TELEMETRY & PHYSICS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.line(frame, (10, 50), (290, 50), (0, 255, 255), 1)

        sidebar_y = 80
        for item in items_sim:
            item.update(FRAME_W, FRAME_H)
            
            cx, cy = item.x, item.y
            z_min_x, z_min_y, z_max_x, z_max_y = containment_bounds
            
            in_zone = (z_min_x <= cx <= z_max_x) and (z_min_y <= cy <= z_max_y)
            
            if in_zone:
                item.billed = True
            
            color = (0, 255, 0) if item.billed else (0, 0, 255)
            draw_3d_box(frame, item.x, item.y, item.w, item.h, item.d, color=color, thickness=2)
            
            if item.billed:
                tag_x, tag_y = item.x + int(item.w/2) + 20, item.y - int(item.h/2)
                cv2.putText(frame, f"CHARGED: ${item.price}", (tag_x, tag_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                tag_x, tag_y = item.x + int(item.w/2) + 20, item.y - int(item.h/2)
                cv2.putText(frame, f"ID-{item.item_id}", (tag_x, tag_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            status_color = (0, 255, 0) if item.billed else (0, 50, 255)
            cv2.putText(frame, f"Item # {item.item_id}  [{'BILLED' if item.billed else 'DRIFTING'}]", 
                        (20, sidebar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            cv2.putText(frame, f" v(x,y,z) = ({item.vx:.1f}, {item.vy:.1f}, {item.vz:.1f})", 
                        (20, sidebar_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(frame, f" rot_spd  = {item.v_rot:.1f} rad/s", 
                        (20, sidebar_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            sidebar_y += 85

        # Frame generation for HTTP format via multipart JPG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return """
    <html>
      <head>
        <title>RetailEye Antigravity View</title>
        <style>
          body { background: #111; color: white; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; font-family: sans-serif; }
          img { border: 2px solid #555; max-width: 90%; max-height: 90%; box-shadow: 0 0 20px rgba(0,0,0,0.8); }
        </style>
      </head>
      <body>
        <h2>RetailEye Microgravity Core</h2>
        <img src="/video_feed" />
      </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
