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
        self.prev_right_ankle_y = None
        self.right_ankle_history = []
        
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
            
    def findHeadPosition(self, frame, left_eye_id = 2, right_eye_id = 5, left_ear_id = 7, right_ear_id = 8, draw=True):
        print("findHeadPosition called!")

        # landmark existence check
        if not len(self.lmList) > max(left_ear_id, right_ear_id):
            return None

        # calculate midpoints of eyes and ears
        x_eyes_mid = (self.lmList[left_eye_id][1] + self.lmList[right_eye_id][1]) / 2
        y_eyes_mid = (self.lmList[left_eye_id][2] + self.lmList[right_eye_id][2]) / 2
        
        x_ear_mid = (self.lmList[left_ear_id][1] + self.lmList[right_ear_id][1]) / 2
        y_ear_mid = (self.lmList[left_ear_id][2] + self.lmList[right_ear_id][2]) / 2
        
        # compute vector from eye to ear
        dx = x_ear_mid - x_eyes_mid
        dy = y_ear_mid - y_eyes_mid

        head_angle = math.degrees(math.atan2(dy, dx)) # calculate angle relative to horizontal
        c_head_angle = 180 - head_angle

        # draw lines and stuff if needed
        h, w = frame.shape[:2]  # [:2] gets height and width (ignores channels if color image)
        scale = min(w, h) / 1000  # Changed from 480 to 1000 for finer scaling

        if draw:
            thickness = max(1, int(1 * scale))  # Thinner lines
            radius = max(3, int(4 * scale))     # Smaller inner circles
            radius_outer = max(5, int(6 * scale)) # Slightly larger outer circles
            font_scale = 0.8 * scale             # Smaller font
            font_thickness = max(1, int(1 * scale)) # Thinner font

            cv2.line(frame, (int(x_ear_mid), int(y_ear_mid)), (int(x_eyes_mid), int(y_eyes_mid)), (255, 0, 0), thickness)
            cv2.circle(frame, (int(x_eyes_mid), int(y_eyes_mid)), radius, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (int(x_eyes_mid), int(y_eyes_mid)), radius_outer, (0, 0, 255), thickness)
            cv2.circle(frame, (int(x_ear_mid), int(y_ear_mid)), radius, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (int(x_ear_mid), int(y_ear_mid)), radius_outer, (0, 0, 255), thickness)

            text = str(int(c_head_angle))
            (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_PLAIN, font_scale, font_thickness)
                
            # Position text near point p2
            text_x = x_ear_mid - int(20 * scale)
            text_y = y_ear_mid + int(20 * scale)

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
        return c_head_angle

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

    def detectFootLanding(self, frame, right_ankle_id=28, right_foot=32, window_size=5, draw=True):
        # Initialize persistent variables if not already set
        if not hasattr(self, 'prev_right_ankle_y'):
            self.prev_right_ankle_y = None
        if not hasattr(self, 'prev_vy'):
            self.prev_vy = 0

        # Get current y position of ankle
        right_ankle_y = self.lmList[right_ankle_id][2]

        # Calculate vertical velocity if possible
        right_landing = False
        vy = 0

        if self.prev_right_ankle_y is not None:
            vy = right_ankle_y - self.prev_right_ankle_y
            # Detect sign change: previously falling (vy < 0), now rising (vy > 0)
            if self.prev_vy < 0 and vy > 0:
                right_landing = True
        else:
            vy = 0  # On first frame

        # Draw result
        h, w = frame.shape[:2]
        scale = min(w, h) / 1000
        if draw:
            font_scale = 0.8 * scale
            font_thickness = max(1, int(1 * scale))
            thickness = max(1, int(1 * scale))  # Thinner lines
            radius = max(3, int(4 * scale))     # Smaller inner circles
            radius_outer = max(5, int(6 * scale)) # Slightly larger outer circles

            cv2.putText(frame, f"curr ray: {right_ankle_y}", (50, 100), cv2.FONT_HERSHEY_PLAIN, font_scale, (0, 0, 255), font_thickness)
            cv2.putText(frame, f"prev ray: {self.prev_right_ankle_y}", (50, 150), cv2.FONT_HERSHEY_PLAIN, font_scale, (0, 0, 255), font_thickness)
            cv2.putText(frame, f"Foot landed: {right_landing}", (50, 200), cv2.FONT_HERSHEY_PLAIN, font_scale, (0, 0, 255), font_thickness)
            cv2.putText(frame, f"prev velocity: {self.prev_vy}", (50, 250), cv2.FONT_HERSHEY_PLAIN, font_scale, (0, 0, 255), font_thickness)
            cv2.putText(frame, f"curr velocity: {vy}", (50, 300), cv2.FONT_HERSHEY_PLAIN, font_scale, (0, 0, 255), font_thickness)
            cv2.circle(frame, (int(self.lmList[right_ankle_id][1]), int(right_ankle_y)), radius_outer, (0, 255, 0), thickness)
            

        # Update previous values for next frame
        self.prev_vy = vy
        self.prev_right_ankle_y = right_ankle_y

        return right_landing

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