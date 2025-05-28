import mediapipe as mp
import cv2
import math

class PoseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
                static_image_mode = self.mode, 
                model_complexity = 2, 
                smooth_landmarks = self.smooth, 
                min_detection_confidence =  self.detectionCon, 
                min_tracking_confidence =  self.trackCon
            )
        
    def findPose(self, frame, draw=True):
        # read frame-by-frame
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(frameRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(frame, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return frame

    # * takes pose landmarks per frame, so use findPose first
    def findPosition(self, frame, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
        return self.lmList
                        
    
    # * takes lmList per frame to calculate angles per frame
    def findAngle(self, frame, p1, p2, p3, draw=True):
        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        
        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                            math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        h, w = frame.shape[:2]  # [:2] gets height and width (ignores channels if color image)
        scale = min(w, h) / 1000  # Changed from 480 to 1000 for finer scaling
        
        # Draw angle
        if draw:
            # Adjusted scaling factors
            thickness = max(1, int(1 * scale))  # Thinner lines
            radius = max(3, int(4 * scale))     # Smaller inner circles
            radius_outer = max(5, int(6 * scale)) # Slightly larger outer circles
            font_scale = 0.8 * scale             # Smaller font
            font_thickness = max(1, int(1 * scale)) # Thinner font
            
            # Draw lines and circles
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), thickness)
            cv2.line(frame, (x3, y3), (x2, y2), (255, 255, 255), thickness)
            cv2.circle(frame, (x1, y1), radius, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x1, y1), radius_outer, (0, 0, 255), thickness)
            cv2.circle(frame, (x2, y2), radius, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), radius_outer, (0, 0, 255), thickness)
            cv2.circle(frame, (x3, y3), radius, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x3, y3), radius_outer, (0, 0, 255), thickness)

            # Draw angle text with better positioning
            text = str(int(angle))
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_PLAIN, font_scale, font_thickness)
            
            # Position text near point p2
            text_x = x2 - int(20 * scale)
            text_y = y2 + int(20 * scale)
            
            # Optional background rectangle for better text visibility
            cv2.rectangle(frame, 
                        (text_x - 2, text_y - text_height - 2),
                        (text_x + text_width + 2, text_y + 2),
                        (255, 255, 255), cv2.FILLED)
            
            cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_PLAIN, font_scale, 
                    (0, 0, 255), font_thickness)
        
        return angle
    
    def findKneeAngle(self, frame, p1, p2, p3, draw=True):
        # Get the landmarks
            x1, y1 = self.lmList[p1][1:]
            x2, y2 = self.lmList[p2][1:]
            x3, y3 = self.lmList[p3][1:]
            # Calculate the Angle
            angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                                math.atan2(y1 - y2, x1 - x2))
            if angle < 0:
                angle += 360

            knee_angle = 360 - angle
            # print(angle)

            h, w = frame.shape[:2]  # [:2] gets height and width (ignores channels if color image)
            scale = min(w, h) / 1000  # Changed from 480 to 1000 for finer scaling

            # Draw angle
            if draw:
                # Adjusted scaling factors
                thickness = max(1, int(1 * scale))  # Thinner lines
                radius = max(3, int(4 * scale))     # Smaller inner circles
                radius_outer = max(5, int(6 * scale)) # Slightly larger outer circles
                font_scale = 0.8 * scale             # Smaller font
                font_thickness = max(1, int(1 * scale)) # Thinner font

                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), thickness)
                cv2.line(frame, (x3, y3), (x2, y2), (255, 255, 255), thickness)
                cv2.circle(frame, (x1, y1), radius, (0, 0, 255), cv2.FILLED)
                cv2.circle(frame, (x1, y1), radius_outer, (0, 0, 255), thickness)
                cv2.circle(frame, (x2, y2), radius, (0, 0, 255), cv2.FILLED)
                cv2.circle(frame, (x2, y2), radius_outer, (0, 0, 255), thickness)
                cv2.circle(frame, (x3, y3), radius, (0, 0, 255), cv2.FILLED)
                cv2.circle(frame, (x3, y3), radius_outer, (0, 0, 255), thickness)

                text = str(int(knee_angle))
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_PLAIN, font_scale, font_thickness)
                
                # Position text near point p2
                text_x = x2 - int(20 * scale)
                text_y = y2 + int(20 * scale)
                
                cv2.rectangle(frame, 
                        (text_x - 2, text_y - text_height - 2),
                        (text_x + text_width + 2, text_y + 2),
                        (255, 255, 255), cv2.FILLED)
            
                cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_PLAIN, font_scale, 
                    (0, 0, 255), font_thickness)
            
                return knee_angle
            
    def findHeadPosition1(self, frame, nose_id=0, left_ear_id=7, right_ear_id=8, shoulder_mid_id=None, draw=True):
        if len(self.lmList) > max(nose_id, left_ear_id, right_ear_id):
            x_nose, y_nose = self.lmList[nose_id][1:]
            x_lear, y_lear = self.lmList[left_ear_id][1:]
            x_rear, y_rear = self.lmList[right_ear_id][1:]

            # Midpoint between ears (head center)
            x_head = (x_lear + x_rear) / 2
            y_head = (y_lear + y_rear) / 2

            # Calculate head tilt (side-to-side lean)
            ear_dx = x_rear - x_lear
            ear_dy = y_rear - y_lear
            head_tilt_angle = math.degrees(math.atan2(ear_dy, ear_dx))  # Positive if right ear is lower

            # Calculate forward/backward head lean (nose relative to head center)
            nose_dx = x_nose - x_head
            nose_dy = y_nose - y_head
            head_lean_angle = math.degrees(math.atan2(nose_dy, nose_dx)) - 90  # Upright ~0°

            # Normalize angles
            head_tilt_angle = (head_tilt_angle + 360) % 360  # 0° = horizontal
            head_lean_angle = (head_lean_angle + 180) % 360 - 180  # [-180, 180]

            if draw:
                # Draw head line (ear-to-ear)
                cv2.line(frame, (int(x_lear), int(y_lear)), (int(x_rear), int(y_rear)), (0, 255, 255), 2)
                # Draw nose-to-head line
                cv2.line(frame, (int(x_head), int(y_head)), (int(x_nose), int(y_nose)), (255, 0, 0), 2)
                # Display angles
                cv2.putText(frame, f"Tilt: {head_tilt_angle:.1f}°", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Lean: {head_lean_angle:.1f}°", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            return {
                "head_tilt": head_tilt_angle,  # 0° = level, + = right ear down
                "head_lean": head_lean_angle,  # 0° = upright, + = forward, - = backward
                "is_upright": abs(head_lean_angle) < 10 and abs(head_tilt_angle - 180) < 10  # Thresholds
            }
        else:
            return None

    def findHeadPosition(self, frame, nose_id = 0, leftEye_id = 2, rightEye_id = 5, leftEar_id = 7, rightEar_id = 8, draw=True):
        nose_x, nose_y = self.lmList[nose_id][1:]
        left_eye_x, left_eye_y = self.lmList[leftEye_id][1:]
        right_eye_x, right_eye_y = self.lmList[rightEye_id][1:]
        left_ear_x, left_ear_y = self.lmList[leftEar_id][1:]
        right_ear_x, right_ear_y = self.lmList[rightEar_id][1:]

        # check if looking up or down
        eyes_mid_x = (left_eye_x + right_eye_x) / 2
        eyes_mid_y = (left_eye_y + right_eye_y) / 2
        
        ear_mid = ((left_ear_x + right_ear_x) / 2,
                   (left_ear_y + right_ear_y) / 2)
        
        # Calculate pitch angle (improved calculation)
        dx = nose_x - eyes_mid_x
        dy = nose_y - eyes_mid_y
        pitch_angle = math.degrees(math.atan2(dy, dx)) - 90  # -90 adjusts for natural head orientation

        h, w = frame.shape[:2]  # [:2] gets height and width (ignores channels if color image)
        scale = min(w, h) / 1000  # Changed from 480 to 1000 for finer scaling

        if draw:
            thickness = max(1, int(1 * scale))  # Thinner lines
            radius = max(3, int(4 * scale))     # Smaller inner circles
            radius_outer = max(5, int(6 * scale)) # Slightly larger outer circles
            font_scale = 0.8 * scale             # Smaller font
            font_thickness = max(1, int(1 * scale)) # Thinner font

            cv2.line(frame, (int(ear_mid[0]), int(ear_mid[1])), (int(eyes_mid_x), int(eyes_mid_y)), (255, 0, 0), thickness)
            text = str(int(pitch_angle))
            (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_PLAIN, font_scale, font_thickness)
                
            # Position text near point p2
            text_x = ear_mid[0] - int(20 * scale)
            text_y = ear_mid[1] + int(20 * scale)

            # Background rectangle (scaled padding)
            cv2.rectangle(
                frame,
                (int(text_x - int(2 * scale)), int(text_y - text_height - int(2 * scale))),
                (int(text_x + text_width + int(2 * scale)), int(text_y + int(2 * scale))),
                (255, 255, 255), cv2.FILLED
            )
            
            cv2.putText(frame, text, (int(text_x), int(text_y)),
                        cv2.FONT_HERSHEY_PLAIN, font_scale,
                        (0, 255, 0), font_thickness)

        return pitch_angle
    
    def findTorsoLean(self, frame, left_ear_id=7, right_ear_id=8, left_hip_id=23, right_hip_id=24, draw=True):
        if len(self.lmList) > max(right_ear_id, right_hip_id):
            # Get midpoints
            x_ear = (self.lmList[left_ear_id][1] + self.lmList[right_ear_id][1]) / 2
            y_ear = (self.lmList[left_ear_id][2] + self.lmList[right_ear_id][2]) / 2
            x_hip = (self.lmList[left_hip_id][1] + self.lmList[right_hip_id][1]) / 2
            y_hip = (self.lmList[left_hip_id][2] + self.lmList[right_hip_id][2]) / 2

            dx = x_ear - x_hip
            dy = y_ear - y_hip  # Note: In image coordinates, y increases downward

            # Calculate angle between vertical axis (pointing upward) and hip-to-ear vector
            # Using atan2(dx, -dy) because:
            # - Positive x is rightward (ear to right of hip = positive angle)
            # - Negative y is upward (since image y increases downward)
            angle = math.degrees(math.atan2(dx, -dy))
            
            # Normalize to [-180, 180] (though atan2 already returns in this range)
            normalized_angle = angle
            while normalized_angle > 180:
                normalized_angle -= 360
            while normalized_angle < -180:
                normalized_angle += 360

            h, w = frame.shape[:2]  # [:2] gets height and width (ignores channels if color image)
            scale = min(w, h) / 1000  # Changed from 480 to 1000 for finer scaling

            if draw:
                thickness = max(1, int(1 * scale))  # Thinner lines
                radius = max(3, int(4 * scale))     # Smaller inner circles
                radius_outer = max(5, int(6 * scale)) # Slightly larger outer circles
                font_scale = 0.8 * scale             # Smaller font
                font_thickness = max(1, int(1 * scale)) # Thinner font

                cv2.line(frame, (int(x_hip), int(y_hip)), (int(x_ear), int(y_ear)), (0, 255, 0), thickness)
                text = str(int(normalized_angle))
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, font_scale, font_thickness)
                top_left = (x_hip - 55, y_hip - 40)
                bottom_right = (x_hip - 55 + text_width + 60, y_hip - 40 + text_height + 10)
                text_x = int(x_hip) - int(20 * scale)
                text_y = int(y_hip) - int(20 * scale)
                
                # Background rectangle (scaled padding)
                cv2.rectangle(frame,
                            (text_x - int(2 * scale), text_y - text_height - int(2 * scale)),
                            (text_x + text_width + int(2 * scale), text_y + int(2 * scale)),
                            (255, 255, 255), cv2.FILLED)
                
                cv2.putText(frame, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_PLAIN, font_scale,
                            (0, 255, 0), font_thickness)

                return normalized_angle
            else:
                return None
        
    def findRunPhase(self, frame, curr_lmList, prev_lmList, draw=True):
        # curr_lmList and prev_lmList are lmList for current and previous frames

        # Ensure both lists have enough landmarks
        required_indices = [24, 26, 28, 32]
        if not all(len(curr_lmList) > idx and len(prev_lmList) > idx for idx in required_indices):
            print("Not enough landmarks detected in one of the frames.")
            return "Unknown"
        
        # Get relevant landmarks (using Mediapipe's indices)
        hip = curr_lmList[24]
        knee = curr_lmList[26]
        ankle = curr_lmList[28]
        foot = curr_lmList[32]
        prev_ankle = prev_lmList[28]

        # Calculate vertical movement of ankle
        vertical_movement = ankle[2] - prev_ankle[2]  # y increases downward in images

        # Is ankle below hip? (foot on ground)
        ankle_below_hip = ankle[2] > hip[2]

        self.ankle_min_y = getattr(self, 'ankle_min_y', ankle[2])
        self.ankle_min_y = min(self.ankle_min_y, ankle[2])
        ankle_off_ground = ankle[2] < self.ankle_min_y + 10  # 30 pixels above lowest point (tune as needed)

        # Is ankle moving up or down?
        moving_up = vertical_movement < 0
        moving_down = vertical_movement > 0

        # Example rule-based phase detection
        if ankle_below_hip and moving_down:
            phase = "Stance"
        elif moving_up and ankle_off_ground:
            phase = "Swing"
        elif ankle_below_hip and moving_up:
            phase = "Toe-off"
        else:
            phase = "Unknown"

        if draw:
            cv2.putText(frame, str(phase), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            cv2.putText(frame, str((ankle_off_ground)), (70, 250), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        print(f"Phase: {phase}")
        return phase