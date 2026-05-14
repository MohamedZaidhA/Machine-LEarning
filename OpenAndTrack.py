import cv2
import mediapipe as mp
import numpy as np
import time
import threading
try:
    import winsound
    USE_WINSOUND = True
except Exception:
    USE_WINSOUND = False

# Alert thread controls
alert_stop_event = threading.Event()
alert_thread = None

def _alert_loop():
    while not alert_stop_event.is_set():
        if USE_WINSOUND:
            winsound.Beep(1000, 500)
        else:
            # Fallback terminal bell
            print('\a', end='', flush=True)
            time.sleep(0.5)

        # Small pause between beeps, but exit quickly if stopped
        for _ in range(5):
            if alert_stop_event.is_set():
                break
            time.sleep(0.1)

def start_alert():
    global alert_thread, alert_stop_event
    if alert_thread is None or not alert_thread.is_alive():
        alert_stop_event.clear()
        alert_thread = threading.Thread(target=_alert_loop, daemon=True)
        alert_thread.start()

def stop_alert():
    global alert_thread, alert_stop_event
    alert_stop_event.set()
    if alert_thread is not None:
        alert_thread.join(timeout=1)
        alert_thread = None

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def extract_key_points(landmarks):
    """Extract key posture points from landmarks."""
    left_shoulder = np.array([
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
    ])
    right_shoulder = np.array([
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
    ])
    left_hip = np.array([
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
    ])
    right_hip = np.array([
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
    ])
    left_ear = np.array([
        landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y,
    ])
    right_ear = np.array([
        landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y,
    ])
    
    return {
        'left_shoulder': left_shoulder,
        'right_shoulder': right_shoulder,
        'left_hip': left_hip,
        'right_hip': right_hip,
        'left_ear': left_ear,
        'right_ear': right_ear,
    }

def calculate_posture_deviation(current_landmarks, reference_landmarks):
    """Calculate how much current posture deviates from reference."""
    if reference_landmarks is None:
        return None
    
    current = extract_key_points(current_landmarks)
    
    # Calculate normalized distance between key points
    deviations = []
    for key in current.keys():
        dist = np.linalg.norm(current[key] - reference_landmarks[key])
        deviations.append(dist)
    
    avg_deviation = np.mean(deviations)
    return avg_deviation

def check_posture_vs_reference(deviation):
    """Check posture quality based on deviation from reference."""
    if deviation is None:
        return "Calibrating...", (200, 200, 200), "Waiting for reference posture..."
    
    if deviation < 0.05:
        return "Good Posture", (0, 255, 0), "Perfect! Maintain this position."
    elif deviation < 0.10:
        return "Adjust Posture", (0, 165, 255), "Slight adjustment needed - stay close to reference."
    elif deviation < 0.15:
        return "Bad Posture", (0, 0, 255), "Move back to your calibrated position."
    else:
        return "Bad Posture", (0, 0, 255), "Posture significantly different - reset and sit like you did in calibration."

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Setup mode variables
setup_mode = True
reference_landmarks = None
reference_image = None

# Bad posture tracking
bad_posture_start_time = None

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if setup_mode:
        # Setup phase: ask user to sit properly and calibrate
        if results.pose_landmarks:
            # Draw pose landmarks in setup mode
            posture_status, color, posture_tip = check_posture_vs_reference(None)
            landmark_style = mp_drawing.DrawingSpec(color=(100, 200, 255), thickness=2, circle_radius=3)
            connection_style = mp_drawing.DrawingSpec(color=(100, 200, 255), thickness=2, circle_radius=1)

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_style,
                connection_drawing_spec=connection_style,
            )
        
        # Display setup instructions
        cv2.rectangle(frame, (10, 10), (w - 10, 150), (50, 50, 50), -1)
        cv2.putText(frame, "SETUP MODE: Calibrate Your Good Posture", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "1. Sit with proper posture", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(frame, "2. Keep ears over shoulders, back straight", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(frame, "3. Press 'S' to save your posture as reference", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        
        key = cv2.waitKey(10) & 0xFF
        if (key == ord('s') or key == ord('S')) and results.pose_landmarks:
            # Save reference posture
            reference_landmarks = extract_key_points(results.pose_landmarks.landmark)
            reference_image = frame.copy()
            setup_mode = False
            print("Posture calibrated! Switching to monitoring mode...")
    
    else:
        # Monitoring phase: compare current posture to reference
        if results.pose_landmarks:
            # Calculate deviation from reference
            deviation = calculate_posture_deviation(results.pose_landmarks.landmark, reference_landmarks)
            posture_status, color, posture_tip = check_posture_vs_reference(deviation)
            
            # Draw pose landmarks with color-coded quality
            landmark_style = mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=3)
            connection_style = mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=1)

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_style,
                connection_drawing_spec=connection_style,
            )
            
            # Track bad posture duration
            if posture_status == "Bad Posture":
                if bad_posture_start_time is None:
                    bad_posture_start_time = time.time()
                
                bad_posture_duration = time.time() - bad_posture_start_time
                
                # Show alert if bad posture for 10+ seconds
                if bad_posture_duration >= 10:
                    # Start continuous alert until posture improves
                    start_alert()

                    cv2.rectangle(frame, (10, h - 100), (w - 10, h - 10), (0, 0, 255), -1)
                    cv2.putText(frame, "ALERT: Bad posture for 10 seconds!", (20, h - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                    cv2.putText(frame, "Adjust your posture now to avoid strain.", (20, h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            else:
                # Reset timer when posture improves
                bad_posture_start_time = None
                stop_alert()
            
            # Display posture status
            cv2.putText(frame, posture_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2)
            cv2.putText(frame, posture_tip, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)
            
            # Show deviation metric
            if deviation is not None:
                deviation_text = f"Deviation: {deviation:.3f}"
                cv2.putText(frame, deviation_text, (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (200, 200, 200), 2)
            
            # Show bad posture duration counter
            if bad_posture_start_time is not None:
                bad_posture_duration = time.time() - bad_posture_start_time
                counter_text = f"Bad posture: {bad_posture_duration:.1f}s"
                cv2.putText(frame, counter_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)
        
        # Display floating reference window in top-right corner
        if reference_image is not None:
            # Resize reference image to smaller size for display
            ref_height, ref_width = reference_image.shape[:2]
            window_height = int(h * 0.25)  # 25% of frame height
            window_width = int(window_height * ref_width / ref_height)
            
            # Make sure window doesn't exceed frame width
            if window_width > w * 0.3:
                window_width = int(w * 0.3)
                window_height = int(window_width * ref_height / ref_width)
            
            ref_resized = cv2.resize(reference_image, (window_width, window_height))
            
            # Position in top-right corner with padding
            padding = 10
            x_pos = w - window_width - padding
            y_pos = padding
            
            # Create a border for the floating window
            cv2.rectangle(frame, (x_pos - 5, y_pos - 25), (x_pos + window_width + 5, y_pos + window_height + 5), 
                         (100, 200, 255), 2)
            cv2.putText(frame, "Reference Posture", (x_pos, y_pos - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
            
            # Overlay the reference image
            frame[y_pos:y_pos + window_height, x_pos:x_pos + window_width] = ref_resized
        
        # Show reset option
        cv2.putText(frame, "Press 'R' to recalibrate posture", (10, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        
        key = cv2.waitKey(10) & 0xFF
        if key == ord('r') or key == ord('R'):
            setup_mode = True
            reference_landmarks = None
            bad_posture_start_time = None
            stop_alert()
            print("Recalibration mode activated...")
    
    cv2.imshow('Posture Tracker', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
