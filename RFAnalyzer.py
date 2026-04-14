import numpy as np

class RFAnalyzer:
    def __init__(self, user_profile: dict = None):
        self.ideal_angles = {
            "head_position": {"min": 10, "max": 20, "tolerance": 4},
            "back_position": {"min": 6, "max": 12, "tolerance": 4},
            "arm_flexion": {"min": 70, "max": 90, "tolerance": 15},
            "left_knee": {"min": 80, "max": 120, "tolerance": 20},
            "right_knee": {"min": 120, "max": 170, "tolerance": 20},
            "foot_strike": {"min": 5, "max": 10, "tolerance": 5},
        }
        self.height_tolerance_modifiers = self._compute_height_modifiers(user_profile or {})
        self.bmi_tolerance_modifiers = self._compute_bmi_modifiers(user_profile or {})

        self.scores_history = {key: [] for key in self.ideal_angles} # Stores scores per joint across frames
        self.angle_values_history = {key: [] for key in self.ideal_angles} # store actual angles values for user representation

    def reset(self):
        """Clear history arrays to free memory and avoid stale data."""
        for key in self.scores_history:
            self.scores_history[key].clear()
        for key in self.angle_values_history:
            self.angle_values_history[key].clear()

    def _compute_height_modifiers(self, user_profile: dict) -> dict:
        """
        Compute per-joint tolerance modifiers based on the runner's height.

        Rationale:
        - Taller runners (>183 cm) have longer limb segments. Longer femurs and
          tibias naturally produce slightly different knee flexion angles vs.
          a median-height runner, so we widen knee tolerances.
        - Shorter runners (<160 cm) have a shorter torso-to-leg ratio which can
          shift back lean and arm carry angles, so we widen back/arm tolerances.
        - Head position and foot strike are geometry-independent — no adjustment.
        """
        height_cm = user_profile.get("height_cm")
        if not height_cm:
            return {}  # No profile: use base tolerances unchanged

        modifiers = {}
        if height_cm > 183:  # Tall: wider knee tolerance (+10°)
            modifiers["left_knee"] = 10
            modifiers["right_knee"] = 10
        elif height_cm < 160:  # Short: wider back/arm tolerance (+2°, +3°)
            modifiers["back_position"] = 2
            modifiers["arm_flexion"] = 3
        return modifiers

    def _compute_bmi_modifiers(self, user_profile: dict) -> dict:
        """
        Compute per-joint tolerance modifiers based on the runner's BMI.
        
        Rationale:
        Higher BMI correlates with reduced joint mobility and extra mechanical limitations.
        We provide slightly wider tolerances for knee flexion and foot strike 
        so that overweight/obese runners are not excessively penalized.
        """
        height_cm = user_profile.get("height_cm")
        weight_kg = user_profile.get("weight_kg")
        
        if not height_cm or not weight_kg or height_cm <= 0:
            return {}
            
        bmi = weight_kg / ((height_cm / 100) ** 2)
        
        modifiers = {}
        if bmi >= 30: # Obese: wider knee/foot tolerance (+10°, +5°)
            modifiers["left_knee"] = 10
            modifiers["right_knee"] = 10
            modifiers["foot_strike"] = 5
        elif bmi >= 25: # Overweight: wider knee/foot tolerance (+5°, +3°)
            modifiers["left_knee"] = 5
            modifiers["right_knee"] = 5
            modifiers["foot_strike"] = 3
            
        return modifiers

    def calculate_score(self, angle_name, measured_angle):
        if angle_name not in self.ideal_angles:
            return 0
        
        ideal = self.ideal_angles[angle_name]
        min_ideal, max_ideal = ideal['min'], ideal['max']
        # Apply height-based tolerance modifier if available
        tolerance = ideal['tolerance'] + self.height_tolerance_modifiers.get(angle_name, 0)
        # Apply BMI-based tolerance modifier
        tolerance += self.bmi_tolerance_modifiers.get(angle_name, 0)
        mid_ideal = (min_ideal + max_ideal) / 2

        if min_ideal <= measured_angle <= max_ideal:
            return 100
        
        deviation = abs(measured_angle - mid_ideal)
        if deviation > tolerance:
            return 0

        # Linear penalty (e.g., halfway to tolerance → 50%)
        score = max(0, 100 - (deviation / tolerance) * 100)
        return score


    def analyze_frame(self, angles):
        
    # ✓ Add validation and debugging
        if not angles or len(angles) != 6:
            print(f"Invalid angles received: {angles}")
            return {}
        
        # Check for None values
        if any(angle is None for angle in angles):
            print(f"None values in angles: {angles}")
            return {}        

        # score a single frame and store results
        frame_results = {}
        joint_names = ['head_position', 'back_position', 'arm_flexion', 'left_knee', 'right_knee', 'foot_strike']

        for name, angle in zip(joint_names, angles):
            if name in self.ideal_angles and angle is not None:
                try:
                    score = self.calculate_score(name, angle)
                    self.scores_history[name].append(score) # track per angle name
                    self.angle_values_history[name].append(angle) # store raw angle per frame
                    frame_results[name] = {
                        'value': round(float(angle)),
                        'score_percentage': round(float(score)),
                        'is_ideal': (score >= 100)
                    }
                except Exception as e:
                    print(f"Error processing {name} with angle {angle}: {e}")
        return frame_results

    def get_summary(self):
        """Generate median-based scores and comprehensive statistics."""
        summary = {}

        print(f"Scores history: {[(k, len(v)) for k, v in self.scores_history.items()]}")
        print(f"Angle values history: {[(k, len(v)) for k, v in self.angle_values_history.items()]}")

        for joint, scores in self.scores_history.items():
            if scores and len(scores) > 0:
                angles = self.angle_values_history[joint]
                
                # ✓ Add debugging to see what's in the data
                print(f"Processing {joint}: scores={scores}, angles={angles}")
                
                if angles and len(angles) > 0:
                    try:
                        # ✓ Convert to numpy arrays and ensure they're numeric
                        scores_array = np.array(scores, dtype=float)
                        angles_array = np.array(angles, dtype=float)
                        
                        # ✓ Check for any NaN or inf values
                        if np.any(np.isnan(scores_array)) or np.any(np.isinf(scores_array)):
                            print(f"Warning: Invalid scores found for {joint}: {scores}")
                            continue
                        
                        if np.any(np.isnan(angles_array)) or np.any(np.isinf(angles_array)):
                            print(f"Warning: Invalid angles found for {joint}: {angles}")
                            continue
                        
                        # ✓ Calculate statistics safely
                        median_score = float(np.median(scores_array))
                        average_score = float(np.mean(scores_array))
                        min_score = float(np.min(scores_array))
                        max_score = float(np.max(scores_array))

                        
                        typical_angle = float(np.median(angles_array))
                        min_angle = float(np.min(angles_array))
                        max_angle = float(np.max(angles_array))
                        
                        # ✓ Calculate consistency score safely
                        representative_score = float(self.calculate_score(joint, typical_angle))
                        mean_score = np.mean(scores_array)
                        if mean_score > 0:
                            std_score = np.std(scores_array)
                            consistency_score = max(0, 100 - (std_score / mean_score * 100))
                        else:
                            consistency_score = 0
                        
                        summary[joint] = {
                            'median_score': round(float(representative_score)),
                            'raw_median_score': round(float(median_score)),
                            'typical_angle': round(float(typical_angle)),
                            'average_score': round(float(average_score)),
                            'min_score': round(float(min_score)),
                            'max_score': round(float(max_score)),
                            'frames_analyzed': len(scores),
                            'angle_range': f"{round(min_angle)}° - {round(max_angle)}°",
                            'ideal_range': f"{self.ideal_angles[joint]['min']}° - {self.ideal_angles[joint]['max']}°",
                            'consistency_score': round(float(consistency_score))
                        }
                        
                    except Exception as e:
                        print(f"Error processing {joint}: {e}")
                        print(f"Scores: {scores}")
                        print(f"Angles: {angles}")
                        continue

        # ✓ Calculate overall scores safely
        if summary:
            try:
                median_scores = [v['median_score'] for v in summary.values()]
                average_scores = [v['average_score'] for v in summary.values()]
                
                if median_scores:
                    summary['overall_score'] = round(float(np.mean(median_scores)))
                else:
                    summary['overall_score'] = 0
                    
            except Exception as e:
                print(f"Error calculating overall scores: {e}")
                summary['overall_score'] = 0.0
                summary['overall_average'] = 0.0
        else:
            print("No valid joint data found for summary")
            summary = {
                'overall_score': 0,
                'overall_average': 0
            }
        
        return summary