import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import time
import math

# --- CONFIGURATION ---
MODEL_PATH = "drone_gesture_model.h5"
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.7 
HISTORY_LENGTH = 5          # Anti-Jitter Buffer

# Flight Constraints
MAX_ALTITUDE = 1000  # cm
MIN_FLYING_ALT = 50  # cm (Safety Floor)

CLASSES = {
    0: "STOP (Hover)",
    1: "FLIP (Fist)",    
    2: "TAKEOFF",
    3: "UP",
    4: "DOWN",          
    5: "PATH_MODE" 
}

# --- 1. ROBUST SHAPE ENGINE ---
def smooth_geometric_shape(points):
    if len(points) < 10: return points
    pts = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(pts)
    perimeter = cv2.arcLength(hull, True)
    area = cv2.contourArea(hull)
    if perimeter == 0: return points

    circularity = (4 * math.pi * area) / (perimeter ** 2)
    shape_type = "UNKNOWN"
    x, y, w, h = cv2.boundingRect(hull)
    center = (int(x + w/2), int(y + h/2))
    radius = int(max(w, h) / 2)

    if circularity > 0.85:
        shape_type = "CIRCLE"
    else:
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(hull, epsilon, True)
        corners = len(approx)
        if corners == 3: shape_type = "TRIANGLE"
        elif corners == 4: shape_type = "SQUARE"
        else: shape_type = "POLYGON"

    perfect_shape = []
    if shape_type == "CIRCLE":
        print(f"⭕ Detected CIRCLE (Score: {circularity:.2f})")
        for i in range(36):
            angle = math.radians(i * 10)
            px = int(center[0] + radius * math.cos(angle))
            py = int(center[1] + radius * math.sin(angle))
            perfect_shape.append((px, py))
    elif shape_type == "SQUARE":
        print(f"🔲 Detected SQUARE (Score: {circularity:.2f})")
        side = max(w, h)
        x = center[0] - side//2
        y = center[1] - side//2
        perfect_shape = [(x, y), (x+side, y), (x+side, y+side), (x, y+side), (x, y)]
    elif shape_type == "TRIANGLE":
        print(f"📐 Detected TRIANGLE (Score: {circularity:.2f})")
        for i in range(3):
            angle = math.radians(270 + i * 120) 
            px = int(center[0] + radius * math.cos(angle))
            py = int(center[1] + radius * math.sin(angle))
            perfect_shape.append((px, py))
        perfect_shape.append(perfect_shape[0])
    else:
        # Fallback: Just return the smoothed Hull points
        print(f"🔹 Detected GENERIC SHAPE ({len(hull)} pts)")
        for p in hull:
            perfect_shape.append((p[0][0], p[0][1]))
        perfect_shape.append(perfect_shape[0])

    return perfect_shape


# --- 2. DRONE SIMULATOR ---
class DroneSimulator:
    def __init__(self):
        self.is_flying = False
        self.auto_landing = False
        self.auto_takeoff = False
        self.is_flipping = False    
        self.flip_frame = 0         
        
        self.x = 0.0; self.y = 0.0; self.z = 0
        self.window_size = 600
        self.trail_points = deque(maxlen=200) 

    def draw(self):
        # PHYSICS: Auto Takeoff
        if self.auto_takeoff and self.is_flying:
            self.z += 2
            if self.z >= MIN_FLYING_ALT:
                self.z = MIN_FLYING_ALT
                self.auto_takeoff = False
                print("🛫 TAKEOFF COMPLETE.")

        # PHYSICS: Auto Landing
        if self.auto_landing and self.is_flying:
            self.z -= 3
            if self.z <= 0:
                self.z = 0
                self.is_flying = False
                self.auto_landing = False
                print("🔻 LANDED.")
        
        # PHYSICS: Backflip Animation
        if self.is_flipping:
            self.flip_frame += 1
            self.z += 10 if self.flip_frame < 10 else -10
            if self.flip_frame >= 20:
                self.is_flipping = False
                self.flip_frame = 0
                print("🔄 BACKFLIP COMPLETE.")

        sim_img = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)
        center_x, center_y = self.window_size // 2, self.window_size // 2
        offset_x = center_x - self.x
        offset_y = center_y - self.y
        
        # Grid
        grid_sz = 50
        sx, sy = int(offset_x % grid_sz), int(offset_y % grid_sz)
        for i in range(-grid_sz, self.window_size + grid_sz, grid_sz):
            cv2.line(sim_img, (i + sx, 0), (i + sx, self.window_size), (40, 40, 40), 1)
            cv2.line(sim_img, (0, i + sy), (self.window_size, i + sy), (40, 40, 40), 1)

        # Trail
        if len(self.trail_points) > 1:
            screen_pts = []
            for pt in self.trail_points:
                screen_pts.append([int(pt[0] + offset_x), int(pt[1] + offset_y)])
            cv2.polylines(sim_img, [np.array(screen_pts)], False, (100, 100, 100), 1)

        # Drone Icon
        if self.is_flipping:
            color = (0, 255, 255) 
            radius = 20
        else:
            color = (0, 255, 0) if self.is_flying else (0, 0, 255)
            radius = 15
            
        cv2.circle(sim_img, (center_x, center_y), radius, color, -1 if self.is_flying else 2)
        
        # HUD
        bar_h = int((self.z / MAX_ALTITUDE) * 200)
        cv2.rectangle(sim_img, (550, 500), (570, 300), (50, 50, 50), -1)
        cv2.rectangle(sim_img, (550, 500), (570, 500 - bar_h), (255, 255, 0), -1)
        cv2.putText(sim_img, f"{int(self.z)}cm", (520, 520), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        
        status = "AIRBORNE" if self.is_flying else "GROUNDED"
        if self.auto_landing: status = "AUTO-LANDING..."
        if self.auto_takeoff: status = "TAKING OFF..."
        if self.is_flipping:  status = "DOING BACKFLIP!"
        
        cv2.putText(sim_img, f"STATUS: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Drone Radar Simulator", sim_img)

    def execute(self, command):
        if not self.is_flying and command != "TAKEOFF":
            self.draw(); return

        if command == "TAKEOFF":
            if self.auto_landing:
                print("🛑 LANDING ABORTED! GOING AROUND!")
                self.auto_landing = False
                self.z = max(self.z, MIN_FLYING_ALT) 
            if not self.is_flying:
                print("🛫 INITIATING AUTO-TAKEOFF...")
                self.is_flying = True
                self.z = 0 
                self.auto_takeoff = True 
        
        elif command == "FLIP": 
            if self.is_flying and not self.is_flipping:
                print("🔄 PERFORMING BACKFLIP!")
                self.is_flipping = True

        elif command == "UP":
            if self.is_flying and not self.auto_takeoff:
                if self.z < MAX_ALTITUDE: self.z += 5
        elif command == "DOWN":
            if self.is_flying and not self.auto_takeoff:
                if self.z > MIN_FLYING_ALT: self.z -= 5
        
        # --- NEW NAVIGATION COMMANDS FOR 'FOLLOW ME' ---
        elif command == "FORWARD":
            if self.is_flying: self.y -= 5  # Move North
        elif command == "BACKWARD":
            if self.is_flying: self.y += 5  # Move South
        elif command == "LEFT":
            if self.is_flying: self.x -= 5  # Move West
        elif command == "RIGHT":
            if self.is_flying: self.x += 5  # Move East
        
        elif command == "LAND (Thumbs Down)":
            if self.is_flying and not self.auto_landing:
                print("🔻 INITIATING AUTO-LANDING...")
                self.auto_landing = True
            
        self.draw()

    def run_path(self, waypoints):
        print(f"🧩 SIMULATOR: Flying Shape ({len(waypoints)} pts)")
        if not self.is_flying:
            self.is_flying = True; self.z = MIN_FLYING_ALT

        if not waypoints: return
        
        # Start from the first point of the shape
        start_sx, start_sy = waypoints[0]
        
        for i in range(1, len(waypoints)):
            target_sx, target_sy = waypoints[i]
            dx = (target_sx - start_sx); dy = (target_sy - start_sy)
            start_sx, start_sy = target_sx, target_sy
            steps = 60 
            step_x = dx / steps; step_y = dy / steps
            
            for _ in range(steps):
                self.x += step_x; self.y += step_y
                self.trail_points.append((self.x, self.y))
                self.draw()
                cv2.waitKey(20)
        print("✅ Path Complete")


# --- INITIALIZATION ---
print("Loading AI...")
# Try to load model, but we will use geometry primarily for critical gestures
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("AI Model Loaded.")
except:
    print("Warning: Model not found. Using Geometric Fallbacks.")
    model = None

drone = DroneSimulator()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# --- NEW: FACE DETECTION FOR TRACKING ---
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

path_points = deque(maxlen=200)
is_recording_path = False
gesture_buffer = deque(maxlen=HISTORY_LENGTH)

# Timers
takeoff_timer = 0
TAKEOFF_HOLD_TIME = 1.0 

flip_timer = 0           
FLIP_HOLD_TIME = 1.0     

# --- FOLLOW ME VARIABLES ---
follow_timer = 0
FOLLOW_HOLD_TIME = 1.0
is_following = False
tracker = None
# Fallback tracker logic
try:
    tracker_factory = cv2.TrackerCSRT_create
except AttributeError:
    tracker_factory = cv2.TrackerKCF_create

print("---------------------------------------")
print("SYSTEM READY.")
print("1. HOLD 'TAKEOFF' (Thumbs Up) for 1s")
print("2. HOLD 'FIST' for 1s to BACKFLIP")
print("3. HOLD 'PEACE' for 1s to FOLLOW ME") 
print("4. 'Spiderman' to START Recording")
print("5. 'Index Finger' to DRAW")
print("6. 'PALM' to FINISH & FLY")
print("7. 'THUMBS DOWN' to Auto-Land")
print("---------------------------------------")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run Hands
    result = hands.process(rgb_frame)
    
    drone.draw()
    
    raw_gesture = "Waiting..."
    confirmed_gesture = "Waiting..."
    
    # --- TRACKING LOGIC ---
    if is_following and tracker:
        success, box = tracker.update(frame)
        if success:
            tx, ty, tw, th = [int(v) for v in box]
            center_x = tx + tw // 2
            
            cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (255, 0, 0), 2)
            cv2.putText(frame, "LOCKED ON", (tx, ty-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            
            frame_center = w // 2
            dead_zone = 50
            
            # Yaw Control
            if center_x < frame_center - dead_zone:
                cv2.putText(frame, "<< ROTATE LEFT", (20, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                drone.execute("LEFT") 
            elif center_x > frame_center + dead_zone:
                cv2.putText(frame, "ROTATE RIGHT >>", (w-250, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                drone.execute("RIGHT")
            
            # Pitch Control
            target_width = 150 
            if tw < target_width - 20:
                cv2.putText(frame, "^ FORWARD", (w//2 - 50, h-80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                drone.execute("FORWARD")
            elif tw > target_width + 20:
                cv2.putText(frame, "v BACK", (w//2 - 50, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                drone.execute("BACKWARD")
        else:
            cv2.putText(frame, "LOST TRACKING!", (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    
    if result.multi_hand_landmarks:
        for idx, hand_lms in enumerate(result.multi_hand_landmarks):
            handedness = result.multi_handedness[idx].classification[0].label
            
            # Bbox
            x_min, y_min = w, h; x_max, y_max = 0, 0
            for lm in hand_lms.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x); y_min = min(y_min, y)
                x_max = max(x_max, x); y_max = max(y_max, y)
            pad = 40
            x_min = max(0, x_min-pad); y_min = max(0, y_min-pad)
            x_max = min(w, x_max+pad); y_max = min(h, y_max+pad)

            try:
                wrist = hand_lms.landmark[0]
                thumb_tip = hand_lms.landmark[4]
                thumb_mcp = hand_lms.landmark[2] # Thumb knuckle
                index_mcp = hand_lms.landmark[5]
                pinky_mcp = hand_lms.landmark[17]
                index_tip = hand_lms.landmark[8]
                
                # Helper for Finger Logic
                def dist(lm_idx):
                    pt = hand_lms.landmark[lm_idx]
                    return math.hypot(pt.x - wrist.x, pt.y - wrist.y)
                
                d_index_tip = dist(8); d_index_pip = dist(6)
                d_middle_tip = dist(12); d_middle_pip = dist(10)
                d_ring_tip = dist(16); d_ring_pip = dist(14)
                d_pinky_tip = dist(20); d_pinky_pip = dist(18)

                # -----------------------------------------------------------------
                #  CRITICAL FIX: DISAMBIGUATE FIST vs THUMBS UP vs THUMBS DOWN
                # -----------------------------------------------------------------
                
                # 1. Are the 4 Fingers (Index to Pinky) Curled?
                # Logic: Tip is closer to wrist than the PIP joint
                fingers_curled = (d_index_tip < d_index_pip) and \
                                 (d_middle_tip < d_middle_pip) and \
                                 (d_ring_tip < d_ring_pip) and \
                                 (d_pinky_tip < d_pinky_pip)
                
                if fingers_curled:
                    # 2. Check Thumb Extension relative to Index Knuckle (MCP)
                    # Calculate distance between Thumb Tip and Index MCP
                    thumb_extension = math.hypot(thumb_tip.x - index_mcp.x, thumb_tip.y - index_mcp.y)
                    
                    # Threshold: 0.05 is roughly "close/tucked", >0.1 is "extended"
                    is_thumb_extended = thumb_extension > 0.08  

                    if not is_thumb_extended:
                        # Fingers curled + Thumb tucked = FIST
                        raw_gesture = "FLIP (Fist)"
                    else:
                        # Fingers curled + Thumb out = Thumbs Up OR Down
                        # 3. Check Orientation (Y-axis)
                        # Remember: Y=0 is TOP. Smaller Y is HIGHER.
                        
                        if thumb_tip.y < wrist.y: # Tip above wrist
                            raw_gesture = "TAKEOFF"
                        else: # Tip below wrist
                            raw_gesture = "LAND (Thumbs Down)"

                else:
                    # Fingers are NOT curled. Check Open Hand gestures.

                    # Check Spiderman (Index & Pinky up, Middle & Ring down)
                    is_spiderman = (d_index_tip > d_index_pip) and \
                                   (d_pinky_tip > d_pinky_pip) and \
                                   (d_middle_tip < d_middle_pip) and \
                                   (d_ring_tip < d_ring_pip)
                    
                    if is_spiderman:
                        raw_gesture = "PATH_MODE"

                    # Check Peace (Index & Middle up, Ring & Pinky down)
                    is_peace = (d_index_tip > d_index_pip) and \
                               (d_middle_tip > d_middle_pip) and \
                               (d_ring_tip < d_ring_pip) and \
                               (d_pinky_tip < d_pinky_pip)
                    
                    if is_peace:
                        raw_gesture = "PEACE (Follow)"
                    
                    # Check Stop (Palm Open)
                    # All fingers extended
                    all_open = (d_index_tip > d_index_pip) and \
                               (d_middle_tip > d_middle_pip) and \
                               (d_ring_tip > d_ring_pip) and \
                               (d_pinky_tip > d_pinky_pip)
                               
                    if all_open:
                        raw_gesture = "STOP (Hover)"

                    # AI Model Fallback (Only if we are still unsure)
                    if raw_gesture == "Waiting..." and model is not None:
                        hand_crop = rgb_frame[y_min:y_max, x_min:x_max]
                        if hand_crop.size != 0:
                            if handedness == "Left": hand_crop = cv2.flip(hand_crop, 1)
                            img_input = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE)) / 255.0
                            img_input = np.expand_dims(img_input, axis=0)
                            prediction = model.predict(img_input, verbose=0)
                            class_id = np.argmax(prediction)
                            
                            if np.max(prediction) > CONFIDENCE_THRESHOLD:
                                ai_gesture = CLASSES.get(class_id, "Unknown")
                                # Prevent AI from overriding our solid geometric locks
                                if ai_gesture not in ["TAKEOFF", "LAND", "FLIP (Fist)"]:
                                    raw_gesture = ai_gesture

                # --- DEBOUNCING ---
                gesture_buffer.append(raw_gesture)
                if len(gesture_buffer) == HISTORY_LENGTH and len(set(gesture_buffer)) == 1:
                    confirmed_gesture = gesture_buffer[0]
                else:
                    confirmed_gesture = "Stabilizing..."

                # --- CONTROLLER LOGIC ---
                display_name = confirmed_gesture
                
                # --- SAFETY LOCK: Takeoff ---
                if confirmed_gesture == "TAKEOFF":
                    if takeoff_timer == 0: takeoff_timer = time.time()
                    elapsed = time.time() - takeoff_timer
                    
                    bar_w = int((elapsed / TAKEOFF_HOLD_TIME) * 200)
                    cv2.rectangle(frame, (20, 100), (220, 120), (0,0,0), -1)
                    cv2.rectangle(frame, (20, 100), (20 + bar_w, 120), (0,255,255), -1)
                    cv2.putText(frame, "HOLD TO TAKEOFF...", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                    
                    if elapsed >= TAKEOFF_HOLD_TIME:
                        drone.execute("TAKEOFF")
                        takeoff_timer = 0 
                else:
                    takeoff_timer = 0 

                # --- SAFETY LOCK: Backflip ---
                if confirmed_gesture == "FLIP (Fist)":
                    if flip_timer == 0: flip_timer = time.time()
                    elapsed_flip = time.time() - flip_timer
                    
                    bar_w = int((elapsed_flip / FLIP_HOLD_TIME) * 200)
                    cv2.rectangle(frame, (20, 150), (220, 170), (0,0,0), -1)
                    cv2.rectangle(frame, (20, 150), (20 + bar_w, 170), (255,0,255), -1)
                    cv2.putText(frame, "HOLD TO FLIP...", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
                    
                    if elapsed_flip >= FLIP_HOLD_TIME:
                        drone.execute("FLIP")
                        flip_timer = 0
                else:
                    flip_timer = 0

                # --- SAFETY LOCK: FOLLOW ME ---
                if confirmed_gesture == "PEACE (Follow)":
                    if follow_timer == 0: follow_timer = time.time()
                    elapsed_follow = time.time() - follow_timer
                    
                    bar_w = int((elapsed_follow / FOLLOW_HOLD_TIME) * 200)
                    cv2.rectangle(frame, (20, 200), (220, 220), (0,0,0), -1)
                    cv2.rectangle(frame, (20, 200), (20 + bar_w, 220), (255, 100, 0), -1)
                    
                    action_text = "HOLD TO UN-FOLLOW" if is_following else "HOLD TO FOLLOW..."
                    cv2.putText(frame, action_text, (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

                    if elapsed_follow >= FOLLOW_HOLD_TIME:
                        is_following = not is_following
                        follow_timer = 0
                        
                        if is_following:
                            print("👁️ SEEKING TARGET FOR TRACKING...")
                            face_results = face_detector.process(rgb_frame)
                            if face_results.detections:
                                detection = face_results.detections[0] 
                                bboxC = detection.location_data.relative_bounding_box
                                ix, iy, iw, ih = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                                iy = max(0, iy + int(ih * 0.2)) 
                                ih = int(ih * 1.5); iw = int(iw * 1.2); ix = max(0, ix - int(iw*0.1))
                                
                                tracker = tracker_factory()
                                tracker.init(frame, (ix, iy, iw, ih))
                                print("✅ TARGET LOCKED. FOLLOWING MODE ENGAGED.")
                            else:
                                print("❌ NO FACE FOUND. CANNOT FOLLOW.")
                                is_following = False
                        else:
                            print("⏹️ FOLLOW MODE DISENGAGED.")
                            tracker = None
                else:
                    follow_timer = 0

                if not drone.is_flying and confirmed_gesture != "TAKEOFF":
                     display_name = f"{confirmed_gesture} (LOCKED)"

                # RECORDING
                if confirmed_gesture == "PATH_MODE" and drone.is_flying:
                    if not is_recording_path:
                        print("🔴 Recording STARTED")
                        path_points.clear()
                        is_recording_path = True
                    
                if is_recording_path:
                    # Draw with Index Tip
                    cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                    path_points.append((cx, cy))
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
                    
                    # FINISH TRIGGER: Palm (Stop)
                    if confirmed_gesture == "STOP (Hover)" and len(path_points) > 20: 
                        print("🟢 Shape Finished. PERFECTING & FLYING...")
                        is_recording_path = False
                        
                        # SHAPE PERFECTION
                        raw_list = list(path_points)
                        perfect_list = smooth_geometric_shape(raw_list)
                        
                        drone.run_path(perfect_list)
                        path_points.clear()
                
                # FLIGHT
                elif not is_recording_path and drone.is_flying:
                    if "Stabilizing" not in confirmed_gesture:
                        if is_following:
                             cv2.putText(frame, "AUTO-PILOT ACTIVE", (w-250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                             if confirmed_gesture in ["LAND (Thumbs Down)", "FLIP (Fist)"]:
                                 drone.execute(confirmed_gesture.replace(" (LOCKED)", ""))
                        else:
                            if confirmed_gesture not in ["TAKEOFF", "FLIP (Fist)", "PEACE (Follow)"]: 
                                drone.execute(confirmed_gesture.replace(" (LOCKED)", ""))

            except Exception as e: 
                print(f"Error: {e}")
                pass

            color = (0, 0, 255) if "LOCKED" in display_name or is_recording_path else (0, 255, 0)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, f"{display_name}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if len(path_points) > 1:
        cv2.polylines(frame, [np.array(path_points)], False, (0, 255, 255), 2)

    if is_recording_path:
        cv2.putText(frame, "Drawing... Show PALM to Finish", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    if is_following:
        cv2.putText(frame, "FOLLOW MODE: ON", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

    cv2.imshow("Drone Controller", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()