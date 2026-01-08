import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import cv2
import math

class PoseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.8, trackCon=0.8):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        base_options = mp_tasks.BaseOptions(model_asset_path='models/pose_landmarker_heavy.task')  # Path to downloaded model
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE if not self.mode else mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=self.detectionCon,
            min_pose_presence_confidence=self.trackCon,
            min_tracking_confidence=self.trackCon
        )
        self.pose_landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self.lmList = []
        
        self.prev_right_ankle_y = None
        self.right_ankle_history = []
        self.prev_x_hip = None
        self.direction = None
        self.direction_streak = 0
        
    # def findPose(self, frame, draw=True):
    #     # read frame-by-frame
    #     frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     self.results = self.pose.process(frameRGB)
    #     if self.results.pose_landmarks:
    #         if draw:
    #             self._draw_pose_with_limb_colors(frame)
    #     return frame

    def findPose(self, frame, draw=True):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = self.pose_landmarker.detect(mp_image)
        
        if results.pose_landmarks:
            self.lmList = []
            for lm in results.pose_landmarks[0]:  # Assuming single person; adjust if multi-person
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([lm.visibility, cx, cy])  # New format: [visibility, x, y]
            
            if draw:
                self._draw_pose_with_limb_colors(frame)
        
        return frame

    def _draw_pose_with_limb_colors(self, frame):
        if not self.lmList:
            return
        
        h, w, c = frame.shape
        points = {}
        for i, lm in enumerate(self.lmList):
            points[i] = (lm[1], lm[2])  # lm[1] is x, lm[2] is y
        
        # Define left and right limb connections
        left_connections = [
            (11, 13), (13, 15),  # Left arm: shoulder -> elbow -> wrist
            (15, 17), (15, 19), (15, 21),  # Left hand
            (23, 25), (25, 27),  # Left leg: hip -> knee -> ankle
            (27, 29), (27, 31),  # Left foot
        ]
        
        right_connections = [
            (12, 14), (14, 16),  # Right arm: shoulder -> elbow -> wrist
            (16, 18), (16, 20), (16, 22),  # Right hand
            (24, 26), (26, 28),  # Right leg: hip -> knee -> ankle
            (28, 30), (28, 32),  # Right foot
        ]
        
        # Body center connections (neutral color)
        center_connections = [
            (11, 12), (11, 23), (12, 24), (23, 24),  # Torso
            (0, 1), (1, 2), (2, 3), (3, 7),  # Face left
            (0, 4), (4, 5), (5, 6), (6, 8),  # Face right
            (9, 10),  # Mouth
        ]
        
        # Draw connections with different colors
        thickness = 2
        
        # Left limbs in blue
        for start, end in left_connections:
            if start in points and end in points:
                cv2.line(frame, points[start], points[end], (255, 0, 0), thickness)
        
        # Right limbs in red
        for start, end in right_connections:
            if start in points and end in points:
                cv2.line(frame, points[start], points[end], (0, 0, 255), thickness)
        
        # Center/body in green
        for start, end in center_connections:
            if start in points and end in points:
                cv2.line(frame, points[start], points[end], (0, 255, 0), thickness)
        
        # Draw landmarks (joint points) in white
        for point in points.values():
            cv2.circle(frame, point, 3, (255, 255, 255), -1)

    # # * takes pose landmarks per frame, so use findPose first
    # def findPosition(self, frame, draw=True):
    #     self.lmList = []
    #     if self.results.pose_landmarks:
    #         for id, lm in enumerate(self.results.pose_landmarks.landmark):
    #             h, w, c = frame.shape
    #             cx, cy = int(lm.x * w), int(lm.y * h)
    #             self.lmList.append([id, cx, cy])
    #     return self.lmList
    
    def findPosition(self, frame, draw=True):
        # lmList is already populated in findPose; just return it or add drawing
        if draw and self.lmList:
            for lm in self.lmList:
                cv2.circle(frame, (lm[1], lm[2]), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList
    
    # * takes lmList per frame to calculate angles per frame
    def findAngle(self, frame, p1, p2, p3, draw=True):
        # Get the landmarks
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]  # [1] is x, [2] is y
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        x3, y3 = self.lmList[p3][1], self.lmList[p3][2]
        
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
            
            # Draw lines and circles with limb-specific colors
            # Determine color based on which limb is being measured
            if p1 in [11, 13, 15, 23, 25, 27, 29, 31]:  # Left side landmarks
                line_color = (255, 0, 0)  # Blue for left limbs
            elif p1 in [12, 14, 16, 24, 26, 28, 30, 32]:  # Right side landmarks
                line_color = (0, 0, 255)  # Red for right limbs
            else:
                line_color = (0, 255, 0)  # Green for center/other
            
            cv2.line(frame, (x1, y1), (x2, y2), line_color, thickness)
            cv2.line(frame, (x3, y3), (x2, y2), line_color, thickness)
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
            x1, y1 = self.lmList[p1][1], self.lmList[p1][2]  # [1] is x, [2] is y
            x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
            x3, y3 = self.lmList[p3][1], self.lmList[p3][2]
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

                # Draw lines with limb-specific colors
                # Determine color based on which knee is being measured
                if p1 in [23, 25, 27]:  # Left leg landmarks
                    line_color = (255, 0, 0)  # Blue for left leg
                elif p1 in [24, 26, 28]:  # Right leg landmarks
                    line_color = (0, 0, 255)  # Red for right leg
                else:
                    line_color = (0, 255, 0)  # Green for center/other

                cv2.line(frame, (x1, y1), (x2, y2), line_color, thickness)
                cv2.line(frame, (x3, y3), (x2, y2), line_color, thickness)
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

    # def findHeadPosition(self, frame, left_eye_id = 2, right_eye_id = 5, left_ear_id = 7, right_ear_id = 8, draw=True):
    #     # print("findHeadPosition called!")

    #     # landmark existence check
    #     if not len(self.lmList) > max(left_ear_id, right_ear_id):
    #         return None

    #     # calculate midpoints of eyes and ears
    #     x_eyes_mid = (self.lmList[left_eye_id][1] + self.lmList[right_eye_id][1]) / 2
    #     y_eyes_mid = (self.lmList[left_eye_id][2] + self.lmList[right_eye_id][2]) / 2
        
    #     x_ear_mid = (self.lmList[left_ear_id][1] + self.lmList[right_ear_id][1]) / 2
    #     y_ear_mid = (self.lmList[left_ear_id][2] + self.lmList[right_ear_id][2]) / 2
        
    #     # compute vector from eye to ear
    #     dx = x_ear_mid - x_eyes_mid
    #     dy = y_ear_mid - y_eyes_mid

    #     head_angle = math.degrees(math.atan2(dy, dx)) # calculate angle relative to horizontal
    #     c_head_angle = 180 - head_angle
    def findHeadPosition(self, frame, left_eye_id=2, right_eye_id=5, left_ear_id=7, right_ear_id=8, draw=True):
        if not len(self.lmList) > max(left_ear_id, right_ear_id):
            return None

        # Midpoint between eyes
        x_eyes_mid = (self.lmList[left_eye_id][1] + self.lmList[right_eye_id][1]) / 2
        y_eyes_mid = (self.lmList[left_eye_id][2] + self.lmList[right_eye_id][2]) / 2

        # Midpoint between ears
        x_ear_mid = (self.lmList[left_ear_id][1] + self.lmList[right_ear_id][1]) / 2
        y_ear_mid = (self.lmList[left_ear_id][2] + self.lmList[right_ear_id][2]) / 2

        # Vector from ear midpoint to eye midpoint
        dx = x_eyes_mid - x_ear_mid
        dy = y_eyes_mid - y_ear_mid

        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        angle_deg = -angle_deg
        # Normalize to [-180, 180]
        if angle_deg > 180:
            angle_deg -= 360
        elif angle_deg < -180:
            angle_deg += 360

        # draw lines and stuff if needed
        h, w = frame.shape[:2]
        scale = min(w, h) / 1000

        if draw:
            thickness = max(1, int(1 * scale))
            radius = max(3, int(4 * scale))
            radius_outer = max(5, int(6 * scale))
            font_scale = 0.8 * scale
            font_thickness = max(1, int(1 * scale))

            # Head position line (center/neutral - green)
            line_color = (0, 255, 0)  # Green for head/center measurements

            cv2.line(frame, (int(x_ear_mid), int(y_ear_mid)), (int(x_eyes_mid), int(y_eyes_mid)), line_color, thickness)
            cv2.circle(frame, (int(x_eyes_mid), int(y_eyes_mid)), radius, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (int(x_eyes_mid), int(y_eyes_mid)), radius_outer, (0, 0, 255), thickness)
            cv2.circle(frame, (int(x_ear_mid), int(y_ear_mid)), radius, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (int(x_ear_mid), int(y_ear_mid)), radius_outer, (0, 0, 255), thickness)

            text = str(int(angle_deg))
            (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_PLAIN, font_scale, font_thickness)
                
            # Position text near ear midpoint
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
        return angle_deg

    def findTorsoLean(self, frame, left_ear_id=7, right_ear_id=8, left_hip_id=23, right_hip_id=24, draw=True):
        if len(self.lmList) > max(right_ear_id, right_hip_id):
            # Get midpoints
            x_ear = (self.lmList[left_ear_id][1] + self.lmList[right_ear_id][1]) / 2
            y_ear = (self.lmList[left_ear_id][2] + self.lmList[right_ear_id][2]) / 2
            x_hip = (self.lmList[left_hip_id][1] + self.lmList[right_hip_id][1]) / 2
            y_hip = (self.lmList[left_hip_id][2] + self.lmList[right_hip_id][2]) / 2

            # Calculate vector from hip to ear (represents spine/torso direction)
            dx = x_ear - x_hip  # horizontal component
            dy = y_ear - y_hip  # vertical component (Note: y increases downward in image coordinates)

            # Calculate angle between torso and vertical axis
            # We want: 0° = straight up, positive = forward lean, negative = backward lean
            # atan2(-dy, dx) gives angle from horizontal axis to vector (hip->ear)
            # We subtract 90° to get angle from vertical axis
            # Then negate to match our desired convention (forward lean = positive)
            angle = math.degrees(math.atan2(dx, -dy))  # -dy because we want upward as positive
            
            # This gives us the angle from vertical where:
            # 0° = straight up (ear directly above hip)
            # positive angles = forward lean (ear ahead of hip)
            # negative angles = backward lean (ear behind hip)
            normalized_angle = angle
            
            # Keep angle in reasonable range [-180, 180]
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

                # Draw line from hip to ear to visualize torso lean (cyan)
                cv2.line(frame, (int(x_hip), int(y_hip)), (int(x_ear), int(y_ear)), (255, 255, 0), thickness)
                
                # Draw circles at key points
                cv2.circle(frame, (int(x_hip), int(y_hip)), radius, (0, 0, 255), cv2.FILLED)
                cv2.circle(frame, (int(x_hip), int(y_hip)), radius_outer, (0, 0, 255), thickness)
                cv2.circle(frame, (int(x_ear), int(y_ear)), radius, (0, 0, 255), cv2.FILLED)
                cv2.circle(frame, (int(x_ear), int(y_ear)), radius_outer, (0, 0, 255), thickness)
                
                # Display angle text
                text = str(int(normalized_angle))
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, font_scale, font_thickness)
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
            
    def findFootAngle(self, frame, right_ankle_id=28, right_hip_id=24, draw=True):
        xr_ankle=self.lmList[right_ankle_id][1]
        yr_ankle=self.lmList[right_ankle_id][2]
        xr_hip=self.lmList[right_hip_id][1]
        yr_hip=self.lmList[right_hip_id][2]

        dx = xr_ankle - xr_hip
        dy = yr_ankle - yr_hip   # Note: In image coordinates, y increases downward

        angle = math.degrees(math.atan2(dx, dy))
        
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

            # Foot angle line (green as requested)
            line_color = (0, 255, 0)  # Green for foot angle measurements
            
            cv2.line(frame, (int(xr_hip), int(yr_hip)), (int(xr_ankle), int(yr_ankle)), line_color, thickness)
            text = str(int(normalized_angle))
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, font_scale, font_thickness)
            top_left = (xr_hip - 55, yr_hip - 40)
            bottom_right = (xr_hip - 55 + text_width + 60, yr_hip - 40 + text_height + 10)
            text_x = int(xr_ankle) - int(20 * scale)
            text_y = int(yr_ankle) - int(20 * scale)
            
            # Background rectangle (scaled padding)
            cv2.rectangle(frame,
                        (text_x - int(2 * scale), text_y - text_height - int(2 * scale)),
                        (text_x + text_width + int(2 * scale), text_y + int(2 * scale)),
                        (255, 255, 255), cv2.FILLED)
            
            cv2.putText(frame, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_PLAIN, font_scale,
                        (0, 255, 0), font_thickness)

        return normalized_angle


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

    # def findRunPhase(self, frame, curr_lmList, prev_lmList, draw=True):
    #     # curr_lmList and prev_lmList are lmList for current and previous frames

    #     # Ensure both lists have enough landmarks
    #     required_indices = [24, 26, 28, 32]
    #     if not all(len(curr_lmList) > idx and len(prev_lmList) > idx for idx in required_indices):
    #         print("Not enough landmarks detected in one of the frames.")
    #         return "Unknown"
        
    #     # Get relevant landmarks (using Mediapipe's indices)
    #     hip = curr_lmList[24]
    #     knee = curr_lmList[26]
    #     ankle = curr_lmList[28]
    #     foot = curr_lmList[32]
    #     prev_ankle = prev_lmList[28]

    #     # Calculate vertical movement of ankle
    #     vertical_movement = ankle[2] - prev_ankle[2]  # y increases downward in images

    #     # Is ankle below hip? (foot on ground)
    #     ankle_below_hip = ankle[2] > hip[2]

    #     self.ankle_min_y = getattr(self, 'ankle_min_y', ankle[2])
    #     self.ankle_min_y = min(self.ankle_min_y, ankle[2])
    #     ankle_off_ground = ankle[2] < self.ankle_min_y + 10  # 30 pixels above lowest point (tune as needed)

    #     # Is ankle moving up or down?
    #     moving_up = vertical_movement < 0
    #     moving_down = vertical_movement > 0

    #     # Example rule-based phase detection
    #     if ankle_below_hip and moving_down:
    #         phase = "Stance"
    #     elif moving_up and ankle_off_ground:
    #         phase = "Swing"
    #     elif ankle_below_hip and moving_up:
    #         phase = "Toe-off"
    #     else:
    #         phase = "Unknown"

    #     if draw:
    #         cv2.putText(frame, str(phase), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    #         cv2.putText(frame, str((ankle_off_ground)), (70, 250), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    #     print(f"Phase: {phase}")
    #     return phase

    # Enhanced foot landing detection methods for PoseModule.py

    def detectFootLanding_Enhanced(self, frame, left_ankle_id=27, right_ankle_id=28, 
                                left_foot_id=31, right_foot_id=32, 
                                velocity_threshold=2, height_threshold=15, draw=True):
        """
        Enhanced foot landing detection using multiple criteria:
        1. Velocity change (falling to rising)
        2. Relative height to ground/hip
        3. Smoothing to reduce false positives
        """
        
        # Initialize tracking variables if not present
        if not hasattr(self, 'foot_tracking'):
            self.foot_tracking = {
                'left_ankle_history': [],
                'right_ankle_history': [],
                'left_velocity_history': [],
                'right_velocity_history': [],
                'ground_level': None,
                'frame_count': 0
            }
        
        self.foot_tracking['frame_count'] += 1
        
        # Get current positions
        left_ankle_y = self.lmList[left_ankle_id][2] if len(self.lmList) > left_ankle_id else None
        right_ankle_y = self.lmList[right_ankle_id][2] if len(self.lmList) > right_ankle_id else None
        
        left_landing = False
        right_landing = False
        
        # Process left foot
        if left_ankle_y is not None:
            left_landing = self._detect_single_foot_landing(
                left_ankle_y, 'left', velocity_threshold, height_threshold
            )
        
        # Process right foot
        if right_ankle_y is not None:
            right_landing = self._detect_single_foot_landing(
                right_ankle_y, 'right', velocity_threshold, height_threshold
            )
        
        # Update ground level estimation
        self._update_ground_level([left_ankle_y, right_ankle_y])
        
        if draw:
            self._draw_foot_landing_info(frame, left_landing, right_landing)
        
        return {'left_landing': left_landing, 'right_landing': right_landing}

    def _detect_single_foot_landing(self, ankle_y, foot_side, velocity_threshold, height_threshold):
        """Helper method to detect landing for a single foot"""
        history_key = f'{foot_side}_ankle_history'
        velocity_key = f'{foot_side}_velocity_history'
        
        # Add current position to history
        self.foot_tracking[history_key].append(ankle_y)
        
        # Keep only last 10 frames for analysis
        if len(self.foot_tracking[history_key]) > 10:
            self.foot_tracking[history_key].pop(0)
        
        # Need at least 3 frames for velocity calculation
        if len(self.foot_tracking[history_key]) < 3:
            return False
        
        # Calculate velocity (change in y position)
        recent_positions = self.foot_tracking[history_key][-3:]
        velocity = recent_positions[-1] - recent_positions[-3]  # 2-frame difference
        
        # Add velocity to history
        self.foot_tracking[velocity_key].append(velocity)
        if len(self.foot_tracking[velocity_key]) > 5:
            self.foot_tracking[velocity_key].pop(0)
        
        # Detect landing: was moving down (positive velocity), now moving up (negative velocity)
        if len(self.foot_tracking[velocity_key]) >= 2:
            prev_velocity = self.foot_tracking[velocity_key][-2]
            curr_velocity = self.foot_tracking[velocity_key][-1]
            
            # Landing criteria:
            # 1. Was moving down significantly (prev_velocity > threshold)
            # 2. Now moving up or stopped (curr_velocity <= small positive value)
            # 3. Near ground level (if we have ground reference)
            
            velocity_landing = (prev_velocity > velocity_threshold and 
                            curr_velocity < velocity_threshold/2)
            
            # Height-based validation (if ground level is established)
            height_validation = True
            if self.foot_tracking['ground_level'] is not None:
                distance_from_ground = abs(ankle_y - self.foot_tracking['ground_level'])
                height_validation = distance_from_ground < height_threshold
            
            return velocity_landing and height_validation
        
        return False

    def _update_ground_level(self, ankle_positions):
        """Estimate ground level from ankle positions"""
        valid_positions = [pos for pos in ankle_positions if pos is not None]
        
        if not valid_positions:
            return
        
        # Use the maximum y-value (lowest point) as ground reference
        current_max = max(valid_positions)
        
        if self.foot_tracking['ground_level'] is None:
            self.foot_tracking['ground_level'] = current_max
        else:
            # Gradually update ground level (running average)
            self.foot_tracking['ground_level'] = (
                0.95 * self.foot_tracking['ground_level'] + 0.05 * current_max
            )

    def _draw_foot_landing_info(self, frame, left_landing, right_landing):
        """Draw foot landing detection information on frame"""
        h, w = frame.shape[:2]
        scale = min(w, h) / 1000
        font_scale = 0.8 * scale
        font_thickness = max(1, int(1 * scale))
        
        # Status text
        left_status = "LEFT LANDING!" if left_landing else "Left: No"
        right_status = "RIGHT LANDING!" if right_landing else "Right: No"
        ground_level = self.foot_tracking.get('ground_level', 'Unknown')
        
        # Draw status
        cv2.putText(frame, left_status, (10, 50), 
                    cv2.FONT_HERSHEY_PLAIN, font_scale, 
                    (0, 255, 0) if left_landing else (0, 0, 255), font_thickness)
        cv2.putText(frame, right_status, (10, 80), 
                    cv2.FONT_HERSHEY_PLAIN, font_scale, 
                    (0, 255, 0) if right_landing else (0, 0, 255), font_thickness)
        cv2.putText(frame, f"Ground: {int(ground_level) if ground_level != 'Unknown' else ground_level}", 
                    (10, 110), cv2.FONT_HERSHEY_PLAIN, font_scale, (255, 255, 255), font_thickness)

    # Alternative method using foot-ground contact estimation
    def detectFootContact_Alternative(self, frame, left_ankle_id=27, right_ankle_id=28,
                                    left_heel_id=29, right_heel_id=30,
                                    left_toe_id=31, right_toe_id=32, draw=True):
        """
        Alternative approach: Detect foot contact using foot flatness and height
        """
        
        if not hasattr(self, 'contact_tracking'):
            self.contact_tracking = {
                'left_contact_frames': 0,
                'right_contact_frames': 0,
                'left_was_airborne': False,
                'right_was_airborne': False
            }
        
        left_contact = False
        right_contact = False
        left_landing = False
        right_landing = False
        
        # Check if we have all required landmarks
        required_landmarks = [left_ankle_id, right_ankle_id, left_heel_id, right_heel_id, left_toe_id, right_toe_id]
        if all(len(self.lmList) > idx for idx in required_landmarks):
            
            # Left foot analysis
            left_heel_y = self.lmList[left_heel_id][2]
            left_toe_y = self.lmList[left_toe_id][2]
            left_ankle_y = self.lmList[left_ankle_id][2]
            
            # Right foot analysis
            right_heel_y = self.lmList[right_heel_id][2]
            right_toe_y = self.lmList[right_toe_id][2]
            right_ankle_y = self.lmList[right_ankle_id][2]
            
            # Detect foot contact based on foot flatness and relative position
            left_foot_flat = abs(left_heel_y - left_toe_y) < 20  # Adjust threshold as needed
            right_foot_flat = abs(right_heel_y - right_toe_y) < 20
            
            # Additional criteria: foot is at lowest position relative to recent history
            left_contact = left_foot_flat and self._is_foot_grounded(left_ankle_y, 'left')
            right_contact = right_foot_flat and self._is_foot_grounded(right_ankle_y, 'right')
            
            # Detect new landings (transition from airborne to contact)
            left_landing = left_contact and self.contact_tracking['left_was_airborne']
            right_landing = right_contact and self.contact_tracking['right_was_airborne']
            
            # Update tracking state
            self.contact_tracking['left_was_airborne'] = not left_contact
            self.contact_tracking['right_was_airborne'] = not right_contact
            
            if draw:
                self._draw_contact_info(frame, left_contact, right_contact, left_landing, right_landing)
        
        return {
            'left_contact': left_contact,
            'right_contact': right_contact,
            'left_landing': left_landing,
            'right_landing': right_landing
        }

    def _is_foot_grounded(self, ankle_y, foot_side):
        """Check if foot is in ground contact position"""
        history_key = f'{foot_side}_ground_history'
        
        if not hasattr(self, 'ground_tracking'):
            self.ground_tracking = {'left_ground_history': [], 'right_ground_history': []}
        
        # Add current position
        self.ground_tracking[history_key].append(ankle_y)
        
        # Keep last 15 frames
        if len(self.ground_tracking[history_key]) > 15:
            self.ground_tracking[history_key].pop(0)
        
        if len(self.ground_tracking[history_key]) < 5:
            return False
        
        # Check if current position is among the lowest 30% of recent positions
        recent_positions = sorted(self.ground_tracking[history_key])
        threshold_index = int(len(recent_positions) * 0.3)
        return ankle_y >= recent_positions[-threshold_index-1]  # Among the lowest positions

    def _draw_contact_info(self, frame, left_contact, right_contact, left_landing, right_landing):
        """Draw contact detection information"""
        h, w = frame.shape[:2]
        scale = min(w, h) / 1000
        font_scale = 0.7 * scale
        font_thickness = max(1, int(1 * scale))
        
        # Contact status
        left_text = f"Left: {'CONTACT' if left_contact else 'AIRBORNE'}"
        right_text = f"Right: {'CONTACT' if right_contact else 'AIRBORNE'}"
        
        cv2.putText(frame, left_text, (50, 450), 
                    cv2.FONT_HERSHEY_PLAIN, font_scale, 
                    (0, 255, 0) if left_contact else (255, 0, 0), font_thickness)
        cv2.putText(frame, right_text, (50, 500), 
                    cv2.FONT_HERSHEY_PLAIN, font_scale, 
                    (0, 255, 0) if right_contact else (255, 0, 0), font_thickness)
        
        # Landing alerts
        if left_landing:
            cv2.putText(frame, "LEFT FOOT LANDING!", (240, 450), 
                        cv2.FONT_HERSHEY_PLAIN, font_scale, (0, 0, 255), font_thickness)
        if right_landing:
            cv2.putText(frame, "RIGHT FOOT LANDING!", (240, 500), 
                        cv2.FONT_HERSHEY_PLAIN, font_scale, (0, 0, 255), font_thickness)