"""
Feedback Generator Module

This module provides functionality for generating dynamic feedback based on 
running form analysis results using MediaPipe pose detection.
"""

import logging
from typing import Dict, Any
from enum import Enum
from drill_suggestions import DrillManager

logger = logging.getLogger(__name__)

class ScoreThresholds(Enum):
    """Performance score thresholds"""
    EXCELLENT = 85
    GOOD = 70
    NEEDS_IMPROVEMENT = 50
    POOR = 30

class FeedbackGenerator:
    """Generates dynamic feedback based on analysis results"""
    
    # Area-specific thresholds and advice
    FEEDBACK_RULES = {
        "head_position": {
            "thresholds": [
                (float('-inf'), 6, "tilted significantly upward", "Focus on looking ahead at the horizon rather than up at the sky."),
                (6, 10, "tilted slightly upward", "Try to keep your gaze more level with the horizon."),
                (10, 20, "well-positioned", "Great head position! Keep maintaining this neutral head alignment."),
                (20, 24, "tilted slightly downward", "Try to look a bit further ahead rather than down."),
                (24, float('inf'), "tilted significantly downward", "Lift your gaze up to look 10-20 feet ahead instead of at the ground."),
                ("default", "positioned", "Continue monitoring your head position.")
            ]
        },
        "back_position": {
            "thresholds": [
                (float('-inf'), 2, "leaning significantly backward", "Focus on a slight forward lean from your ankles, not your waist."),
                (2, 6, "leaning backward or too upright", "Allow for a slight forward lean (0-5°) from your ankles."),
                (6, 12, "well-positioned with good forward lean", "Excellent torso position! Maintain this slight forward lean."),
                (12, 16, "leaning forward more than optimal", "Reduce your forward lean slightly and engage your core."),
                (16, float('inf'), "leaning too far forward", "Try to run more upright with just a slight forward lean from your ankles."),
                ("default", "positioned", "Continue monitoring your torso position.")
            ]
        },
        "arm_flexion": {
            "thresholds": [
                (float('-inf'), 55, "too bent", "Relax your arms slightly to achieve a 70-90 degree elbow angle."),
                (55, 70, "slightly too bent", "Let your arms extend a bit more for better swing mechanics."),
                (70, 90, "well-positioned", "Great arm angle! Keep this optimal elbow bend."),
                (90, 105, "slightly too extended", "Bring your elbows in a bit more for optimal efficiency."),
                (105, float('inf'), "too extended", "Bend your elbows more to achieve a 70-90 degree angle."),
                ("default", "positioned", "Continue monitoring your arm position.")
            ]
        },
        "right_knee": {
            "thresholds": [
                (float('-inf'), 100, "bent too much on landing", "Try to land with less knee bend for better efficiency."),
                (100, 120, "bent slightly more than optimal on landing", "Allow for a bit less knee bend upon landing."),
                (120, 170, "showing good bend upon foot landing", "Great knee position! Keep this optimal bend upon every stride."),
                (170, 190, "slightly too straight upon landing", "Allow your knees a bit more bend upon landing."),
                (190, float('inf'), "too straight upon landing", "Try to let your front knee bend more upon landing."),
                ("default", "positioned", "Continue monitoring your knee position.")
            ]
        },
        "left_knee": {
            "thresholds": [
                (float('-inf'), 60, "showing excessive heel kick", "Try to open up your back knee more when landing your foot."),
                (60, 80, "heel kick angle tighter than optimal", "Slightly open up your back knee more during foot landing."),
                (80, 120, "showing excellent heel kick", "Keep up this optimal heel kick in your back knee for each stride."),
                (120, 140, "slightly low heel kick", "Allow for a slightly tighter heel kick for better stride efficiency."),
                (140, float('inf'), "showing very low heel kick", "Back knee is too open on foot landing - try for more heel kick."),
                ("default", "positioned", "Continue monitoring your heel kick.")
            ]
        },
        "foot_strike": {
            "thresholds": [
                (float('-inf'), 0, "excessive forefoot strike", "Land more on your midfoot rather than your forefoot."),
                (0, 5, "slight forefoot strike", "Try to land a bit more toward your midfoot."),
                (5, 10, "good midfoot landing", "Excellent foot strike pattern! This is optimal for efficiency."),
                (10, 15, "slight heel strike", "Focus on landing closer to your midfoot for better efficiency."),
                (15, float('inf'), "landing on your heel", "Try to land more on your midfoot directly under your center of gravity."),
                ("default", "positioned", "Continue monitoring your foot strike pattern.")
            ]
        }
    }
    
    @classmethod
    def generate_dynamic_feedback(cls, area: str, angle: float, score: float) -> str:
        """
        Generate dynamic feedback based on measured angles and scores.
        
        Args:
            area: Analysis area (e.g., 'head_position')
            angle: Measured angle value
            score: Performance score (0-100)
            
        Returns:
            str: Formatted feedback message
        """
        if area not in cls.FEEDBACK_RULES:
            return f"Analysis shows {angle:.1f}° average for {area.replace('_', ' ')}."
        
        rules = cls.FEEDBACK_RULES[area]
        
        # Find matching threshold
        for rule in rules["thresholds"]:
            if rule[0] == "default":
                direction = rule[1]
                advice = rule[2]
                break
            elif rule[0] <= angle < rule[1]:
                direction = rule[2]
                advice = rule[3]
                break
        else:
            # No matching rule found, use default
            direction = "showing measured angle"
            advice = "Continue monitoring this metric."
            
        # Format response based on area
        if area == "head_position":
            feedback_str = f"Your head is {direction} (avg: {angle:.1f}°). {advice}"
        elif area == "back_position":
            feedback_str = f"Your torso is {direction} (avg: {angle:.1f}°). {advice}"
        elif area == "arm_flexion":
            feedback_str = f"Your arm flexion is {direction} (avg: {angle:.1f}°). {advice}"
        elif area == "right_knee":
            feedback_str = f"Your front knee is {direction} (avg: {angle:.1f}°). {advice}"
        elif area == "left_knee":
            feedback_str = f"Your back knee is {direction} (avg: {angle:.1f}°). {advice}"
        elif area == "foot_strike":
            feedback_str = f"You are {direction} (avg: {angle:.1f}°). {advice}"
        else:
            feedback_str = f"Analysis shows {angle:.1f}° average for {area.replace('_', ' ')}."

        return {
            "feedback": feedback_str,
            "classification": direction
        }

async def generate_feedback(analysis_summary: Dict[str, Any], user_id: str, drill_manager: DrillManager) -> Dict[str, Any]:
    """
    Generate comprehensive feedback with drill suggestions.
    
    Args:
        analysis_summary: Analysis results from video processing
        user_id: User identifier for personalization
        drill_manager: DrillManager instance for drill suggestions
        
    Returns:
        Dict: Complete feedback structure with drills
    """
    try:
        # Score thresholds
        thresholds = {
            "EXCELLENT": ScoreThresholds.EXCELLENT.value,
            "GOOD": ScoreThresholds.GOOD.value,
            "NEEDS_IMPROVEMENT": ScoreThresholds.NEEDS_IMPROVEMENT.value,
        }

        feedback = {
            "overall_assessment": "",
            "strengths": [],
            "priority_areas": [],
            "detailed_feedback": {},
        }

        # Extract typical angles and scores
        typical_angles = {
            area: analysis_summary.get(area, {}).get("typical_angle", 0)
            for area in ["head_position", "back_position", "arm_flexion", "right_knee", "left_knee", "foot_strike"]
        }

        scores = {
            area: analysis_summary.get(area, {}).get("median_score", 0)
            for area in ["head_position", "back_position", "arm_flexion", "right_knee", "left_knee", "foot_strike"]
        }
        
        overall_score = analysis_summary.get("overall_score", 0)

        # Generate overall assessment
        if overall_score >= thresholds["EXCELLENT"]:
            feedback["overall_assessment"] = "Excellent running form, demonstrating strong technique across most areas."
        elif overall_score >= thresholds["GOOD"]:
            feedback["overall_assessment"] = "Good running form with room for some refinement in specific areas."
        elif overall_score >= thresholds["NEEDS_IMPROVEMENT"]:
            feedback["overall_assessment"] = "Your running form shows potential but would benefit from further enhancements."
        else:
            feedback["overall_assessment"] = "Significant improvements needed in multiple areas of your running form."

        # Process each area
        for area, score in scores.items():
            angle = typical_angles[area]
            
            # Determine performance level
            if score >= thresholds["EXCELLENT"]:
                performance_level = "excellent"
                feedback["strengths"].append(area.replace("_", " ").title())
            elif score >= thresholds["GOOD"]:
                performance_level = "good"
            elif score >= thresholds["NEEDS_IMPROVEMENT"]:
                performance_level = "needs_improvement"
                feedback["priority_areas"].append({
                    "area": area.replace("_", " ").title(),
                    "score": score,
                    "priority": "medium"
                })
            else:
                performance_level = "poor"
                feedback["priority_areas"].append({
                    "area": area.replace("_", " ").title(),
                    "score": score,
                    "priority": "high"
                })
            
            # Get drills from database
            drills = await drill_manager.get_drill_suggestions(area, performance_level, angle, user_id)
            
            # Format drills for frontend
            formatted_drills = [drill_manager.format_drill_for_frontend(drill) for drill in drills]
            
            # Generate detailed feedback with drills
            analysis_result = FeedbackGenerator.generate_dynamic_feedback(area, angle, score)
            feedback["detailed_feedback"][area] = {
                "analysis": analysis_result["feedback"],
                "drills": formatted_drills,
                "score": score,
                "angle": angle,
                "performance_level": performance_level,
                "classification": analysis_result["classification"]
            }
        
        logger.info(f"Generated feedback for user {user_id} with overall score {overall_score}")
        return feedback
        
    except Exception as e:
        logger.error(f"Error generating feedback: {e}")
        return {
            "overall_assessment": "Error generating assessment",
            "strengths": [],
            "priority_areas": [],
            "detailed_feedback": {},
        }
