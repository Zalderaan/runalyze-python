"""
DrillManager - Intelligent drill suggestion system for running form analysis.

This module provides personalized drill recommendations based on:
- Running form analysis scores
- Measured biomechanical angles  
- User progress tracking
- Performance level assessment
"""

from supabase import Client
from typing import List, Dict, Optional, Tuple
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceLevel(Enum):
    """Performance level classifications"""
    POOR = "poor"
    NEEDS_IMPROVEMENT = "needs_improvement" 
    GOOD = "good"
    EXCELLENT = "excellent"

class AnalysisArea(Enum):
    """Supported analysis areas"""
    HEAD_POSITION = "head_position"
    BACK_POSITION = "back_position"
    ARM_FLEXION = "arm_flexion"
    RIGHT_KNEE = "right_knee"
    LEFT_KNEE = "left_knee"
    FOOT_STRIKE = "foot_strike"

@dataclass
class DrillCustomization:
    """Data class for drill customizations"""
    focus_note: Optional[str] = None
    intensity: Optional[str] = None
    progression: Optional[str] = None
    angle_specific: bool = False

@dataclass
class UserProgress:
    """Data class for user drill progress"""
    completed_sessions: int = 0
    last_completed: Optional[str] = None
    difficulty_rating: Optional[int] = None
    notes: Optional[str] = None

class DrillManager:
    """
    Manages drill suggestions and customizations for running form improvement.
    
    Features:
    - Database-driven drill recommendations
    - Angle-specific customizations
    - User progress tracking
    - Performance-based prioritization
    """
    
    # Class constants for better maintainability
    AREA_FOCUS_NOTES = {
        AnalysisArea.HEAD_POSITION.value: "Focus on maintaining neutral gaze throughout the drill",
        AnalysisArea.FOOT_STRIKE.value: "Pay attention to landing softly and quietly",
        AnalysisArea.BACK_POSITION.value: "Maintain slight forward lean from ankles, not waist",
        AnalysisArea.ARM_FLEXION.value: "Keep shoulders relaxed while maintaining elbow angle",
        AnalysisArea.RIGHT_KNEE.value: "Focus on smooth, controlled movement patterns",
        AnalysisArea.LEFT_KNEE.value: "Focus on smooth, controlled movement patterns"
    }
    
    PERFORMANCE_RECOMMENDATIONS = {
        PerformanceLevel.POOR.value: {
            "frequency_multiplier": 1.5,
            "focus": "Start slow and focus on proper form",
            "priority": "high"
        },
        PerformanceLevel.NEEDS_IMPROVEMENT.value: {
            "frequency_multiplier": 1.2,
            "focus": "Gradually increase intensity while maintaining form",
            "priority": "medium"
        },
        PerformanceLevel.GOOD.value: {
            "frequency_multiplier": 1.0,
            "focus": "Maintain current form and add variety",
            "priority": "low"
        },
        PerformanceLevel.EXCELLENT.value: {
            "frequency_multiplier": 0.8,
            "focus": "Focus on advanced variations and consistency",
            "priority": "maintenance"
        }
    }
    
    PRIORITY_SCORES = {"high": 15, "medium": 10, "low": 5, "maintenance": 2}
    
    def __init__(self, supabase_client: Client):
        if not supabase_client:
            raise ValueError("Supabase client is required")
        self.supabase = supabase_client
        self._cache = {}  # Simple in-memory cache for drill data

    async def get_drills_from_database(self, area: str = None, performance_level: str = None) -> List[Dict]:
        """
        Fetch drills from database with optional filtering and caching.
        
        Args:
            area: Filter by specific area
            performance_level: Filter by performance level
            
        Returns:
            List of drill dictionaries
        """
        # Create cache key
        cache_key = f"drills_{area or 'all'}_{performance_level or 'all'}"
        
        # Check cache first
        if cache_key in self._cache:
            logger.debug(f"Cache hit for {cache_key}")
            return self._cache[cache_key]
        
        try:
            query = self.supabase.table("drills").select("*")
            
            # Apply filters
            if area and self._is_valid_area(area):
                query = query.eq("area", area)
            if performance_level and self._is_valid_performance_level(performance_level):
                query = query.eq("performance_level", performance_level)

            response = query.execute()
            
            drills = response.data if response.data else []
            logger.debug(f"Drills fetched from DB for area='{area}', level='{performance_level}': {drills}")
            
            # Cache the result
            self._cache[cache_key] = drills
            
            logger.info(f"Fetched {len(drills)} drills for area='{area}', level='{performance_level}'")
            return drills
            
        except Exception as e:
            logger.error(f"Error fetching drills from database: {str(e)}")
            return []
        
    async def get_drill_customizations(self, drill_id: int, area: str, angle: float) -> DrillCustomization:
        """
        Get angle-specific customizations for a drill.
        
        Args:
            drill_id: The drill ID
            area: The performance area
            angle: Current measured angle
            
        Returns:
            DrillCustomization object with angle-specific modifications
        """
        if not self._is_valid_area(area) or angle is None:
            return DrillCustomization()
        
        try:
            response = self.supabase.table("drill_customizations")\
                .select("*")\
                .eq("drill_id", drill_id)\
                .eq("area", area)\
                .execute()
            
            if not response.data:
                return DrillCustomization()
            
            # Find customization that matches the angle range
            for customization in response.data:
                if self._angle_matches_range(angle, customization):
                    return DrillCustomization(
                        focus_note=customization.get("focus_note"),
                        intensity=customization.get("intensity"),
                        progression=customization.get("progression"),
                        angle_specific=True
                    )
            
            return DrillCustomization()
            
        except Exception as e:
            logger.error(f"Error fetching drill customizations: {str(e)}")
            return DrillCustomization()
    
    def _angle_matches_range(self, angle: float, customization: Dict) -> bool:
        """Check if angle falls within customization range."""
        angle_min = customization.get("angle_min")
        angle_max = customization.get("angle_max")
        
        if angle_min is not None and angle_max is not None:
            return angle_min <= angle <= angle_max
        elif angle_min is not None:
            return angle >= angle_min
        elif angle_max is not None:
            return angle <= angle_max
        
        return False

    async def get_drill_suggestions(self, area: str, performance_level: str, 
                                  angle: float = None, user_id: str = None) -> List[Dict]:
        """
        Get customized drill suggestions based on performance area, level, and user data.
        
        Args:
            area: Performance area (e.g., 'head_position', 'foot_strike')
            performance_level: 'poor', 'needs_improvement', 'good', 'excellent'
            angle: Optional current angle for customizations
            user_id: Optional user ID for progress tracking
        
        Returns:
            List of customized drill recommendations
        """
        # Validate inputs
        if not self._is_valid_area(area) or not self._is_valid_performance_level(performance_level):
            logger.warning(f"Invalid area '{area}' or performance_level '{performance_level}'")
            return []
        
        try:
            # Get base drills from database
            drills = await self.get_drills_from_database(area, performance_level)
            
            # ✅ Add this logging
            logger.info(f"Fetched {len(drills)} drills for area='{area}', level='{performance_level}'")
            if len(drills) == 0:
                logger.warning(f"⚠️ NO DRILLS FOUND for {area} + {performance_level}")
            
            if not drills:
                logger.info(f"No drills found for area='{area}', level='{performance_level}'")
                return []
            
            # Process drills concurrently for better performance
            # customization_tasks = []
            # progress_tasks = []
            
            # for drill in drills:
            #     if angle is not None:
            #         customization_tasks.append(
            #             self.get_drill_customizations(drill["id"], area, angle)
            #         )
            #     else:
            #         customization_tasks.append(asyncio.create_task(
            #             self._get_empty_customization()
            #         ))
                
            #     if user_id:
            #         progress_tasks.append(
            #             self.get_user_drill_progress(user_id, drill["id"])
            #         )
            #     else:
            #         progress_tasks.append(asyncio.create_task(
            #             self._get_empty_progress()
            #         ))
            
            # # Wait for all customizations and progress data
            # customizations = await asyncio.gather(*customization_tasks)
            # try:
            #     progress_data = await asyncio.gather(*progress_tasks, return_exceptions=True)
            # except asyncio.CancelledError:
            #     logger.error("Async task was cancelled")
            #     raise
            
            # Build customized drills
            customized_drills = []
            for i, drill in enumerate(drills):
                customized_drill = self._build_customized_drill(
                    drill, DrillCustomization(), UserProgress(), 
                    performance_level, area, angle
                )
                customized_drills.append(customized_drill)
            
            # Sort drills by priority
            prioritized_drills = self._prioritize_drills(customized_drills, performance_level, user_id)
            
            logger.info(f"Generated {len(prioritized_drills)} customized drills")
            logger.info(f"This is prioritized_drills: {prioritized_drills}")
            return prioritized_drills
            
        except Exception as e:
            logger.error(f"Error generating drill suggestions: {str(e)}")
            return []
    
    async def _get_empty_customization(self) -> DrillCustomization:
        """Return empty customization for async compatibility."""
        return DrillCustomization()
    
    async def _get_empty_progress(self) -> UserProgress:
        """Return empty progress for async compatibility."""
        return UserProgress()
    
    def _build_customized_drill(self, drill: Dict, customization: DrillCustomization, 
                               progress: UserProgress, performance_level: str, 
                               area: str, angle: float = None) -> Dict:
        """Build a customized drill with all enhancements."""
        customized_drill = drill.copy()
        
        # # Add customizations
        # if customization.angle_specific:
        #     customized_drill.update({
        #         "focus_note": customization.focus_note,
        #         "intensity": customization.intensity,
        #         "progression": customization.progression,
        #         "angle_specific": True
        #     })
        
        # # Add user progress
        # if progress.completed_sessions > 0:
        #     customized_drill["user_progress"] = {
        #         "completed_sessions": progress.completed_sessions,
        #         "last_completed": progress.last_completed,
        #         "difficulty_rating": progress.difficulty_rating,
        #         "notes": progress.notes
        #     }
        
        # Add performance recommendations
        customized_drill = self._add_performance_recommendations(
            customized_drill, performance_level, area, angle
        )
        
        return customized_drill
    
    async def get_user_drill_progress(self, user_id: str, drill_id: int) -> UserProgress:
        """
        Get user's progress for a specific drill.
        
        Args:
            user_id: User identifier
            drill_id: Drill identifier
            
        Returns:
            UserProgress object with user's drill history
        """
        if not user_id or not drill_id:
            return UserProgress()
        
        try:
            response = self.supabase.table("user_drill_progress")\
                .select("*")\
                .eq("user_id", user_id)\
                .eq("drill_id", drill_id)\
                .execute()
            
            if response.data:
                data = response.data[0]
                return UserProgress(
                    completed_sessions=data.get("completed_sessions", 0),
                    last_completed=data.get("last_completed"),
                    difficulty_rating=data.get("difficulty_rating"),
                    notes=data.get("notes")
                )
                
            return UserProgress()
            
        except Exception as e:
            logger.error(f"Error fetching user drill progress: {str(e)}")
            return UserProgress()

    def _add_performance_recommendations(self, drill: Dict, performance_level: str, 
                                        area: str, angle: float = None) -> Dict:
        """
        Add performance-specific recommendations to drills.
        
        Args:
            drill: Base drill dictionary
            performance_level: User's performance level
            area: Analysis area
            angle: Optional current angle measurement
            
        Returns:
            Enhanced drill with performance recommendations
        """
        # Add performance-based recommendations
        if performance_level in self.PERFORMANCE_RECOMMENDATIONS:
            rec = self.PERFORMANCE_RECOMMENDATIONS[performance_level]
            drill["performance_recommendation"] = {
                "frequency_adjustment": rec["frequency_multiplier"],
                "focus_area": rec["focus"],
                "priority_level": rec["priority"]
            }
        
        # Add area-specific focus notes
        if area in self.AREA_FOCUS_NOTES:
            drill["area_focus_note"] = self.AREA_FOCUS_NOTES[area]
        
        return drill

    def _prioritize_drills(self, drills: List[Dict], performance_level: str, 
                          user_id: str = None) -> List[Dict]:
        """
        Sort drills by priority based on performance level and user progress.
        
        Args:
            drills: List of drill dictionaries
            performance_level: User's performance level
            user_id: Optional user ID for progress-based sorting
            
        Returns:
            Sorted list of drills by priority (highest first)
        """
        def calculate_drill_priority(drill: Dict) -> float:
            """Calculate priority score for a single drill."""
            score = 0.0
            
            # Base priority by difficulty level
            difficulty = drill.get("difficulty_level", 1)
            if performance_level == PerformanceLevel.POOR.value:
                score += (6 - difficulty) * 2  # Prefer easier drills for poor performance
            else:
                score += difficulty  # Prefer challenging drills for better performance
            
            # User progress bonus
            user_progress = drill.get("user_progress", {})
            if user_progress:
                completed = user_progress.get("completed_sessions", 0)
                if completed == 0:
                    score += 10  # Prefer new drills
                elif completed < 5:
                    score += 5   # Prefer drills in progress
                else:
                    score += 2   # Lower priority for frequently used drills
            else:
                score += 10  # Prefer drills without progress (new)
            
            # Performance recommendation priority
            perf_rec = drill.get("performance_recommendation", {})
            priority = perf_rec.get("priority_level", "medium")
            score += self.PRIORITY_SCORES.get(priority, 5)
            
            # Angle-specific customization bonus
            if drill.get("angle_specific", False):
                score += 3
            
            return score
        
        # Sort by priority score (descending)
        sorted_drills = sorted(drills, key=calculate_drill_priority, reverse=True)
        
        logger.debug(f"Prioritized {len(sorted_drills)} drills for {performance_level} performance")
        return sorted_drills
    
    def format_drill_for_frontend(self, drill: Dict) -> Dict:
        """
        Format drill data for frontend consumption.
        
        Args:
            drill: Raw drill dictionary from database
            
        Returns:
            Formatted drill dictionary optimized for frontend rendering
        """
        return {
            "id": drill.get("id"),
            "drill_name": drill.get("drill_name", "Unknown Drill"),
            "description": drill.get("description", ""),
            "duration": drill.get("duration", "Not specified"),
            "frequency": drill.get("frequency", "As needed"),
            "instructions": drill.get("instructions", []),
            "video_url": drill.get("video_url"),
            "justification": drill.get("justification"),
            "reference": drill.get("reference"),
            "difficulty_level": max(1, min(5, drill.get("difficulty_level", 1))),  # Ensure 1-5 range
            "focus_note": drill.get("focus_note"),
            "intensity": drill.get("intensity"),
            "progression": drill.get("progression"),
            "area_focus_note": drill.get("area_focus_note"),
            "performance_recommendation": drill.get("performance_recommendation", {}),
            "user_progress": drill.get("user_progress", {}),
            "safety_note": drill.get("safety_note"),
            "angle_specific": drill.get("angle_specific", False),
            "reps": drill.get("reps"),
            "sets": drill.get("sets"),
            "rep_type": drill.get("rep_type"),
            "created_at": drill.get("created_at"),
            "updated_at": drill.get("updated_at")
        }
    
    def format_feedback_for_frontend(self, feedback: Dict) -> Dict:
        """
        Format the entire feedback structure for easier frontend consumption.
        
        Args:
            feedback: Raw feedback dictionary
            
        Returns:
            Formatted feedback optimized for React rendering
        """
        formatted_feedback = {
            "overall_assessment": feedback.get("overall_assessment", ""),
            "strengths": feedback.get("strengths", []),
            "priority_areas": feedback.get("priority_areas", []),
            "detailed_areas": []  # Convert detailed_feedback object to array
        }
        
        # Convert detailed_feedback object to array for easier React rendering
        detailed_feedback = feedback.get("detailed_feedback", {})
        
        for area_key, area_data in detailed_feedback.items():
            formatted_area = {
                "area_key": area_key,
                "area_name": area_key.replace("_", " ").title(),
                "score": area_data.get("score", 0),
                "angle": area_data.get("angle", 0),
                "analysis": area_data.get("analysis", ""),
                "performance_level": area_data.get("performance_level", ""),
                "drills": [self.format_drill_for_frontend(drill) for drill in area_data.get("drills", [])]
            }
            formatted_feedback["detailed_areas"].append(formatted_area)
        
        # Sort areas by score (worst first) for better UX
        formatted_feedback["detailed_areas"].sort(key=lambda x: x["score"])
        
        return formatted_feedback
    
    # Validation helper methods
    def _is_valid_area(self, area: str) -> bool:
        """Validate if area is supported."""
        return area in [e.value for e in AnalysisArea]
    
    def _is_valid_performance_level(self, level: str) -> bool:
        """Validate if performance level is supported."""
        return level in [e.value for e in PerformanceLevel]
    
    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()
        logger.info("Drill cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics for monitoring."""
        return {
            "cache_size": len(self._cache),
            "cache_keys": list(self._cache.keys())
        }