import numpy as np

class RFAnalyzer:
    def __init__(self):
        self.ideal_angles = {
            "head_position": {"min": 10, "max": 20, "tolerance": 4},
            "back_position": {"min": 0, "max": 5, "tolerance": 15},
            "arm_flexion": {"min": 75, "max": 85, "tolerance": 5},
            "left_knee": {"min": 90, "max": 100, "tolerance": 15},
            "right_knee": {"min": 130, "max": 160, "tolerance": 9},
            "foot_strike": {"min": 5, "max": 10, "tolerance": 5},
        }

        self.scores_history = {key: [] for key in self.ideal_angles} # Stores scores per joint across frames
        self.angle_values_history = {key: [] for key in self.ideal_angles} # store actual angles values for user representation

    def calculate_score(self, angle_name, measured_angle):
        if angle_name not in self.ideal_angles:
            return 0 # unknown joint
        
        ideal = self.ideal_angles[angle_name]
        min_ideal, max_ideal = ideal['min'], ideal['max']
        tolerance = ideal['tolerance']
        mid_ideal = (min_ideal + max_ideal) / 2 # get midpoint

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
                        'value': angle,
                        'score_percentage': score,
                        'is_ideal': (score == 100)
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
                        mean_score = np.mean(scores_array)
                        if mean_score > 0:
                            std_score = np.std(scores_array)
                            consistency_score = max(0, 100 - (std_score / mean_score * 100))
                        else:
                            consistency_score = 0
                        
                        summary[joint] = {
                            'median_score': median_score,
                            'average_score': average_score,
                            'min_score': min_score,
                            'max_score': max_score,
                            'frames_analyzed': len(scores),
                            'typical_angle': typical_angle,
                            'angle_range': f"{min_angle:.1f}° - {max_angle:.1f}°",
                            'ideal_range': f"{self.ideal_angles[joint]['min']}° - {self.ideal_angles[joint]['max']}°",
                            'consistency_score': float(consistency_score)
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
                    summary['overall_score'] = float(np.mean(median_scores))
                else:
                    summary['overall_score'] = 0.0
                    
            except Exception as e:
                print(f"Error calculating overall scores: {e}")
                summary['overall_score'] = 0.0
                summary['overall_average'] = 0.0
        else:
            print("No valid joint data found for summary")
            summary = {
                'overall_score': 0.0,
                'overall_average': 0.0
            }
        
        return summary