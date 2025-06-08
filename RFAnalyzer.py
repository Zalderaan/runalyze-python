class RFAnalyzer:
    def __init__(self):
        self.ideal_angles = {
            "head_position": {"min": 10, "max": 20, "tolerance": 4},
            "back_position": {"min": 0, "max": 5, "tolerance": 15},
            "arm_flexion": {"min": 75, "max": 85, "tolerance": 5},
            "left_knee": {"min": 90, "max": 100, "tolerance": 15},
            "right_knee": {"min": 130, "max": 160, "tolerance": 9},
            "foot_strike": {"min": 60, "max": 90, "tolerance": 10},
        }

        # Stores scores per joint across frames
        self.scores_history = {key: [] for key in self.ideal_angles}  
    
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
        # score a single frame and store results
        frame_results = {}
        joint_names = ['head_position', 'back_position', 'arm_flexion', 'left_knee', 'right_knee', 'foot_strike']

        for name, angle in zip(joint_names, angles):
            if name in self.ideal_angles:
                score = self.calculate_score(name, angle)
                self.scores_history[name].append(score) # track per angle name
                frame_results[name] = {
                    'value': angle,
                    'score_percentage': score,
                    'is_ideal': (score == 100)
                }
        return frame_results
    
    def get_summary(self):
        """Generate average scores across all analyzed frames."""
        summary = {}
        for joint, scores in self.scores_history.items():
            if scores:
                summary[joint] = {
                    'average_score': sum(scores) / len(scores),
                    'min_score': min(scores),
                    'max_score': max(scores),
                    'frames_analyzed': len(scores)
                }

        # Add overall score (average of all joint averages)
        if summary:
            summary['overall_score'] = sum(v['average_score'] for v in summary.values()) / len(summary)
        
        return summary