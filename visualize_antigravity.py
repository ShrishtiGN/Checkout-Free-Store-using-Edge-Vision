import cv2
import numpy as np
import random
import time

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

def draw_transparent_rect(img, pt1, pt2, color, alpha):
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_transparent_containment_zone(img, bounds_2d, color=(255, 150, 0), alpha=0.3):
    x_min, y_min, x_max, y_max = bounds_2d
    draw_transparent_rect(img, (x_min, y_min), (x_max, y_max), color, alpha)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(img, "CONTAINMENT FLOAT-BAG", (x_min + 10, y_min + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

class FloatingItemSimulator:
    def __init__(self, item_id, price):
        self.item_id = item_id
        self.price = price
        self.x = random.randint(350, 800)
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
        
        if self.x < 350 or self.x > frame_w - 50:
            self.vx *= -1
        if self.y < 50 or self.y > frame_h - 50:
            self.vy *= -1

def render_edge_health_dashboard(frame, latency_ms, npu_util):
    dashboard_color = (20, 15, 10)  # BGR
    # Transparent Overlay background
    draw_transparent_rect(frame, (880, 20), (1180, 200), dashboard_color, 0.8)
    cv2.rectangle(frame, (880, 20), (1180, 200), (255, 150, 0), 2)
    
    cv2.putText(frame, "EDGE HEALTH DASHBOARD", (895, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 200, 0), 2)
    
    # Status
    cv2.putText(frame, "Source: ", (895, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, "Status: LOCAL (Edge)", (970, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Model
    cv2.putText(frame, "Model: YOLOv11-Tiny | Size: 42MB", (895, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    
    # Latency (Red if >100ms)
    lat_color = (0, 0, 255) if latency_ms > 100 else (0, 255, 0)
    cv2.putText(frame, "Latency:", (895, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, f"{latency_ms:.1f}ms", (970, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lat_color, 2)
    cv2.putText(frame, "Limit: < 100ms", (1050, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    
    # Power Score logic
    score = "A"
    score_color = (0, 255, 0)
    if npu_util > 50: score, score_color = "B", (0, 200, 255)
    if npu_util > 75: score, score_color = "C", (0, 100, 255)
    if npu_util > 90: score, score_color = "F (THROTTLED)", (0, 0, 255)
    
    cv2.putText(frame, f"NPU Util: {npu_util:.1f}%", (895, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, f"Power Score: {score}", (895, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_color, 2)

def run_antigravity_simulation(video_path="mock_input"):
    print(f"🚀 Initializing Edge Antigravity View -> Load: {video_path}")
    FRAME_W, FRAME_H = 1200, 800
    containment_bounds = (400, 200, 800, 600)
    
    items = [
        FloatingItemSimulator(1, 4.99),
        FloatingItemSimulator(2, 1.50),
        FloatingItemSimulator(3, 8.25)
    ]

    cv2.namedWindow('RetailEye: Microgravity Edge System', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('RetailEye: Microgravity Edge System', FRAME_W, FRAME_H)

    while True:
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        
        # Transparent overlays requires copy logic (handled in functions)
        draw_transparent_containment_zone(frame, containment_bounds)
        
        # Dashboard parameters
        mock_lat = 75 + random.uniform(-10, 10)
        if random.random() < 0.05: mock_lat += random.uniform(30, 50) # Spikes
        mock_npu = 40 + random.uniform(0, 40)
        
        render_edge_health_dashboard(frame, mock_lat, mock_npu)

        # Base HUD
        draw_transparent_rect(frame, (0, 0), (320, FRAME_H), (20, 20, 20), 0.9)
        cv2.rectangle(frame, (0, 0), (320, FRAME_H), (100, 100, 100), 2)
        cv2.putText(frame, "TELEMETRY & PHYSICS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.line(frame, (10, 50), (310, 50), (0, 255, 255), 1)

        sidebar_y = 80
        for item in items:
            item.update(FRAME_W, FRAME_H)
            
            cx, cy = item.x, item.y
            z_min_x, z_min_y, z_max_x, z_max_y = containment_bounds
            
            if (z_min_x <= cx <= z_max_x) and (z_min_y <= cy <= z_max_y):
                item.billed = True 
            
            color = (0, 255, 0) if item.billed else (0, 0, 255) 
            draw_3d_box(frame, item.x, item.y, item.w, item.h, item.d, color=color, thickness=2)
            
            tag = f"CHARGED: ${item.price}" if item.billed else f"ID-{item.item_id}"
            
            hw, hh = int(item.w/2), int(item.h/2)
            
            # --- UNIQUE DISPLAY SOLUTIONS PER ITEM CLASS ---
            if item.item_id == 1:
                # Format 1: Text floating above, speed blocks underneath
                tag_x, tag_y = item.x - hw, item.y - hh - 25
                cv2.putText(frame, tag, (tag_x, tag_y), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"v:[{item.vx:.1f},{item.vy:.1f}]", (item.x - hw, item.y - hh - 5), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0), 1)
                cv2.putText(frame, f"w:{item.v_rot:.1f}", (item.x + hw + 5, item.y + hh), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0), 1)
            
            elif item.item_id == 2:
                # Format 2: Left aligned trailing terminal style
                tag_x, tag_y = item.x + hw + 10, item.y - 10
                cv2.putText(frame, tag, (tag_x, tag_y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"-> vel: {item.vx:.1f}x{item.vy:.1f}", (tag_x, tag_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.putText(frame, f"-> rot: {item.v_rot:.1f} r/s", (tag_x, tag_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
            else:
                # Format 3: Inline ticker bottom center
                tag_x, tag_y = item.x - 30, item.y + hh + 20
                cv2.putText(frame, tag, (tag_x, tag_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 255, 255), 1)
                cv2.putText(frame, f"| v:{item.vx:.1f}^ w:{item.v_rot:.1f} |", (item.x - 50, tag_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

            # Sidebar Log
            status_color = (0, 255, 0) if item.billed else (0, 50, 255)
            cv2.putText(frame, f"Item # {item.item_id}  [{'BILLED' if item.billed else 'DRIFTING'}]", 
                        (20, sidebar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            cv2.putText(frame, f" v(x,y,z) = ({item.vx:.1f}, {item.vy:.1f}, {item.vz:.1f}) | w = {item.v_rot:.1f}", 
                        (20, sidebar_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            sidebar_y += 85

        cv2.imshow('RetailEye: Microgravity Edge System', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_antigravity_simulation("mock_video_input.mp4")
