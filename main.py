"""
RunAnalyze - AI-powered running form analysis API.

This module provides endpoints for video analysis, drill suggestions,
and comprehensive running form feedback using MediaPipe pose detection.
"""

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from supabase import create_client, Client
from drill_suggestions import DrillManager
from feedback_generator import generate_feedback, ScoreThresholds

import cv2
import mediapipe as mp
import numpy as np
import os
import uuid
from datetime import datetime
import asyncio
import logging
import subprocess
import json
from typing import Optional, Tuple, Dict, Any
from dotenv import load_dotenv
from dataclasses import dataclass
from enum import Enum

import PoseModule as pm
import RFAnalyzer as rfa

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
class ProcessingStatus(Enum):
    """Video processing status codes"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class VideoConfig:
    """Video processing configuration"""
    fps: float = 30.0
    codec: str = 'avc1'
    thumbnail_timestamp: float = 1.0
    ffmpeg_crf: int = 28
    ffmpeg_preset: str = 'fast'

# Initialize FastAPI app
app = FastAPI(
    title="RunAnalyze API",
    description="AI-powered running form analysis with personalized drill recommendations",
    version="1.0.0"
)

# CORS configuration
CORS_ORIGINS = [
    "https://runalyze-8x1vbed8s-zalderaans-projects.vercel.app",  # Your Vercel app
]

# Add localhost for development
import os
if os.getenv("ENVIRONMENT") != "production":
    CORS_ORIGINS.extend([
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ])

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe and analysis components
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()
detector = pm.PoseDetector()
analyzer = rfa.RFAnalyzer()

# Load environment variables
load_dotenv()

# Environment configuration
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", 'videos')

# Validate required environment variables
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing required Supabase environment variables")

# Initialize Supabase client and drill manager
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
drill_manager = DrillManager(supabase)

# Configuration
video_config = VideoConfig()

logger.info("Application initialized successfully")

# Ensure tmp directory exists
os.makedirs("tmp", exist_ok=True)

class VideoProcessor:
    """Handles video processing operations"""
    
    @staticmethod
    def extract_thumbnail(video_path: str, output_path: str, timestamp: float = 1.0) -> bool:
        """
        Extract a thumbnail from a video at specified timestamp.
        
        Args:
            video_path: Path to input video
            output_path: Path for thumbnail output
            timestamp: Time in seconds to extract frame
            
        Returns:
            bool: True if successful, False otherwise
        """
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video file: {video_path}")
                return False
                
            # Set position to specified timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            
            if not ret:
                logger.error(f"Cannot read frame at timestamp {timestamp}s")
                return False
            
            success = cv2.imwrite(output_path, frame)
            if success:
                logger.info(f"Thumbnail extracted successfully: {output_path}")
                return True
            else:
                logger.error(f"Failed to save thumbnail to: {output_path}")
                return False

        except Exception as e:
            logger.error(f"Error extracting thumbnail: {e}")
            return False
        finally:
            if cap:
                cap.release()

    @staticmethod
    def get_video_rotation(video_path: str) -> int:
        """
        Get the rotation metadata from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            int: Rotation angle (0, 90, 180, 270)
        """
        try:
            # First try using ffprobe to get video metadata
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_streams', '-select_streams', 'v:0', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if 'streams' in data and len(data['streams']) > 0:
                    stream = data['streams'][0]
                    
                    # Check for rotation in tags (common in mobile videos)
                    if 'tags' in stream:
                        if 'rotate' in stream['tags']:
                            rotation = int(stream['tags']['rotate'])
                            logger.info(f"Found rotation tag: {rotation}")
                            return rotation
                    
                    # Check for side_data_list (newer format)
                    if 'side_data_list' in stream:
                        for side_data in stream['side_data_list']:
                            if side_data.get('side_data_type') == 'Display Matrix':
                                rotation = side_data.get('rotation', 0)
                                if rotation != 0:
                                    logger.info(f"Found rotation in display matrix: {rotation}")
                                    return abs(int(rotation))
                    
                    # Check video dimensions and codec for mobile video indicators
                    width = stream.get('width', 0)
                    height = stream.get('height', 0)
                    codec = stream.get('codec_name', '')
                    
                    logger.info(f"Video dimensions: {width}x{height}, codec: {codec}")
                    
                    # For videos that appear vertical but should be landscape (mobile issue)
                    if height > width and height / width > 1.3:
                        logger.info(f"Detected likely rotated mobile video: {width}x{height}")
                        # Mobile landscape videos usually need 90° clockwise rotation
                        return 90
            
            return 0
        except Exception as e:
            logger.warning(f"Could not detect video rotation with ffprobe: {e}")
            # Fallback: try to detect using OpenCV dimensions
            return VideoProcessor._detect_rotation_from_dimensions(video_path)

    @staticmethod
    def _detect_rotation_from_dimensions(video_path: str) -> int:
        """
        Fallback method to infer rotation from video dimensions.
        Mobile videos shot in landscape but appearing vertical often have
        height > width when they should have width > height.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            int: Inferred rotation angle (0 or 90)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                logger.info(f"Video dimensions analysis: {width}x{height}, frames: {frame_count}")
                
                # For typical mobile landscape videos that appear vertical,
                # the height is usually greater than width due to incorrect orientation
                if height > width:
                    aspect_ratio = height / width
                    logger.info(f"Vertical aspect ratio detected: {aspect_ratio:.2f}")
                    
                    # Strong indication of rotated landscape video
                    if aspect_ratio > 1.3:  # e.g., 1080x1920 instead of 1920x1080
                        logger.info(f"Video appears to be rotated landscape - needs 90° correction")
                        return 90
                    # Mild indication - could be portrait or slightly rotated
                    elif aspect_ratio > 1.1:
                        logger.info(f"Video might be rotated - applying 90° correction")
                        return 90
                
            return 0
        except Exception as e:
            logger.warning(f"Could not detect rotation from dimensions: {e}")
            return 0

    @staticmethod
    def add_faststart(input_path: str) -> str:
        """
        Add faststart metadata to video for web streaming with proper orientation correction.
        
        Args:
            input_path: Path to input video
            
        Returns:
            str: Path to processed video
        """
        try:
            output_path = input_path.replace(".mp4", "_faststart.mp4")
            
            # Detect video rotation
            rotation = VideoProcessor.get_video_rotation(input_path)
            logger.info(f"Detected video rotation: {rotation} degrees")
            
            # Use FFmpeg's autorotate feature as primary method
            autorotate_cmd = (
                f'ffmpeg -noautorotate -i "{input_path}" -c:v libx264 -crf {video_config.ffmpeg_crf} '
                f'-preset {video_config.ffmpeg_preset} -movflags +faststart '
                f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                f'-metadata:s:v rotate=0 "{output_path}" -y'
            )
            
            # Manual transpose method for when autorotate doesn't work
            manual_cmd = None
            if rotation != 0:
                # Debug: Let's try different approaches based on the detected rotation
                logger.info(f"Processing rotation: {rotation}° - determining correct transpose")
                
                # For mobile videos, the rotation metadata can be misleading
                # Let's try a more empirical approach
                if rotation == 90:
                    # Try counter-clockwise first for 90° metadata
                    transpose_filter = "transpose=2,"  # 90 degrees counter-clockwise
                    logger.info("Using transpose=2 (90° counter-clockwise) for 90° rotation")
                elif rotation == 270:
                    # For 270° metadata, try clockwise rotation
                    transpose_filter = "transpose=1,"  # 90 degrees clockwise  
                    logger.info("Using transpose=1 (90° clockwise) for 270° rotation")
                elif rotation == 180:
                    transpose_filter = "transpose=1,transpose=1,"  # 180 degrees
                    logger.info("Using double transpose for 180° rotation")
                else:
                    # For dimension-based detection (no metadata), try clockwise
                    transpose_filter = "transpose=1,"
                    logger.info("Using default transpose=1 (90° clockwise) for dimension-based detection")
                
                # Build the video filter chain with manual rotation
                video_filters = f"{transpose_filter}scale=trunc(iw/2)*2:trunc(ih/2)*2"
                
                manual_cmd = (
                    f'ffmpeg -noautorotate -i "{input_path}" -c:v libx264 -crf {video_config.ffmpeg_crf} '
                    f'-preset {video_config.ffmpeg_preset} -movflags +faststart '
                    f'-vf "{video_filters}" '
                    f'-metadata:s:v rotate=0 "{output_path}" -y'
                )
            
            # Try the appropriate command - prioritize manual rotation if detected
            if rotation != 0 and manual_cmd:
                cmd_to_use = manual_cmd
                logger.info(f"Using manual rotation correction for {rotation}° rotation: {input_path}")
            else:
                cmd_to_use = autorotate_cmd
                logger.info(f"Using autorotate processing (rotation: {rotation}°): {input_path}")
            
            result = os.system(cmd_to_use)
            
            if result == 0:
                logger.info(f"Video processing completed with orientation fix: {output_path}")
                return output_path
            else:
                logger.error(f"FFmpeg processing failed with code: {result}")
                # If manual command failed, try alternative rotations
                if rotation != 0 and manual_cmd and cmd_to_use == manual_cmd:
                    logger.info("Manual rotation failed, trying alternative rotations...")
                    
                    # Try the opposite rotation
                    alt_transpose = ""
                    if "transpose=1" in manual_cmd:
                        alt_transpose = "transpose=2"  # Try counter-clockwise instead
                        logger.info("Trying counter-clockwise rotation")
                    elif "transpose=2" in manual_cmd:
                        alt_transpose = "transpose=1"  # Try clockwise instead
                        logger.info("Trying clockwise rotation")
                    
                    if alt_transpose:
                        alt_output = input_path.replace(".mp4", "_alt_faststart.mp4")
                        alt_cmd = (
                            f'ffmpeg -noautorotate -i "{input_path}" -c:v libx264 -crf {video_config.ffmpeg_crf} '
                            f'-preset {video_config.ffmpeg_preset} -movflags +faststart '
                            f'-vf "{alt_transpose},scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                            f'-metadata:s:v rotate=0 "{alt_output}" -y'
                        )
                        
                        result = os.system(alt_cmd)
                        if result == 0:
                            logger.info(f"Alternative rotation successful: {alt_output}")
                            return alt_output
                    
                    # Final fallback to autorotate
                    logger.info("Trying autorotate fallback...")
                    result = os.system(autorotate_cmd)
                    if result == 0:
                        logger.info(f"Video processing completed with autorotate fallback: {output_path}")
                        return output_path
                
                return input_path
                
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return input_path

    @staticmethod
    def test_video_rotation(video_path: str) -> None:
        """
        Test function to debug video rotation detection.
        
        Args:
            video_path: Path to test video
        """
        try:
            logger.info(f"Testing rotation detection for: {video_path}")
            rotation = VideoProcessor.get_video_rotation(video_path)
            logger.info(f"Detected rotation: {rotation} degrees")
            
            # Also check dimensions
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                logger.info(f"Video properties: {width}x{height} @ {fps:.2f}fps")
            
        except Exception as e:
            logger.error(f"Error testing video rotation: {e}")

    @staticmethod
    def create_rotation_test_videos(input_path: str) -> Dict[str, str]:
        """
        Create test videos with different rotations to find the correct one.
        
        Args:
            input_path: Path to input video
            
        Returns:
            Dict mapping rotation description to output path
        """
        test_videos = {}
        base_path = input_path.replace(".mp4", "")
        
        rotation_tests = [
            ("clockwise_90", "transpose=1"),
            ("counter_clockwise_90", "transpose=2"), 
            ("clockwise_180", "transpose=1,transpose=1"),
            ("flip_horizontal", "hflip"),
            ("flip_vertical", "vflip"),
            ("flip_both", "hflip,vflip")
        ]
        
        for desc, filter_cmd in rotation_tests:
            output_path = f"{base_path}_test_{desc}.mp4"
            cmd = (
                f'ffmpeg -i "{input_path}" -c:v libx264 -crf 28 -preset fast '
                f'-vf "{filter_cmd},scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                f'-metadata:s:v rotate=0 -t 10 "{output_path}" -y'
            )
            
            logger.info(f"Creating test video: {desc}")
            result = os.system(cmd)
            
            if result == 0:
                test_videos[desc] = output_path
                logger.info(f"Created test video: {output_path}")
            else:
                logger.error(f"Failed to create test video: {desc}")
        
        return test_videos

def smart_rotate_frame(frame, rotation):
    """
    Intelligently rotate a frame by trying different directions and detecting
    if the result looks correct (not upside-down).
    
    Args:
        frame: OpenCV frame
        rotation: Rotation angle (0, 90, 180, 270)
        
    Returns:
        Rotated frame
    """
    if rotation == 0:
        return frame
    
    if rotation == 90 or rotation == 270:
        # Try counter-clockwise first (common fix for mobile videos)
        counterclockwise = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # For debugging, let's also try clockwise to compare
        clockwise = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # For now, return counter-clockwise as it often fixes upside-down issues
        logger.info(f"Applied COUNTER-CLOCKWISE rotation for {rotation}° metadata")
        return counterclockwise
        
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    else:
        logger.warning(f"Unknown rotation angle: {rotation}, returning original frame")
        return frame


def rotate_frame(frame, rotation):
    """
    Rotate a frame based on detected rotation angle.
    For mobile videos, we try counter-clockwise rotation first.
    
    Args:
        frame: OpenCV frame
        rotation: Rotation angle (0, 90, 180, 270)
        
    Returns:
        Rotated frame
    """
    return smart_rotate_frame(frame, rotation)


def test_rotation_directions(frame, rotation_angle):
    """
    Test different rotation directions to find the correct one.
    This can help debug mobile video orientation issues.
    """
    if rotation_angle == 0:
        return frame, "no_rotation"
    
    # Try different rotations
    clockwise = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    counterclockwise = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    logger.info(f"Original frame shape: {frame.shape}")
    logger.info(f"Clockwise rotation shape: {clockwise.shape}")
    logger.info(f"Counterclockwise rotation shape: {counterclockwise.shape}")
    
    # For now, return clockwise and log the attempt
    return clockwise, "clockwise"

class StorageManager:
    """Handles file upload operations to Supabase"""
    
    def __init__(self, supabase_client: Client, bucket_name: str):
        self.supabase = supabase_client
        self.bucket_name = bucket_name
    
    def upload_thumbnail(self, uid: str, thumbnail_path: str, video_uuid: str) -> Optional[str]:
        """Upload thumbnail image to Supabase storage."""
        try:
            # Ensure the file exists before uploading
            if not os.path.exists(thumbnail_path):
                logger.error(f"Thumbnail file does not exist: {thumbnail_path}")
                return None
                
            unique_filename = f"annotated-footage/{uid}/{video_uuid}/thumbnail.jpg"
            logger.info(f"Uploading thumbnail: {thumbnail_path} -> {unique_filename}")
            
            with open(thumbnail_path, 'rb') as file:
                file_content = file.read()

            result = self.supabase.storage.from_(self.bucket_name).upload(
                path=unique_filename,
                file=file_content,
                file_options={
                    "content-type": "image/jpeg",
                    "x-upsert": "true"
                }
            )

            if hasattr(result, 'path') and result.path:
                public_url = self.supabase.storage.from_(self.bucket_name).get_public_url(unique_filename)
                logger.info(f"Thumbnail uploaded successfully: {unique_filename}")
                return public_url
            else:
                logger.error(f"Thumbnail upload failed: {getattr(result, 'error', 'Unknown error')}")
                return None
                
        except Exception as e:
            logger.error(f"Error uploading thumbnail: {e}")
            return None

    def upload_video(self, uid: str, file_path: str, file_name: str, video_uuid: str) -> Tuple[Optional[str], Optional[str]]:
        """Upload processed video to Supabase storage."""
        try:
            # Ensure the file exists before uploading
            if not os.path.exists(file_path):
                logger.error(f"Video file does not exist: {file_path}")
                return None, None
                
            folder = "annotated-footage"
            unique_filename = f"{folder}/{uid}/{video_uuid}_{file_name}"
            logger.info(f"Uploading video: {file_path} -> {unique_filename}")
            
            with open(file_path, 'rb') as file:
                file_content = file.read()

            result = self.supabase.storage.from_(self.bucket_name).upload(
                path=unique_filename,
                file=file_content,
                file_options={
                    "content-type": "video/mp4", 
                    "x-upsert": "true"
                }
            )
            
            if hasattr(result, 'path') and result.path:
                public_url = self.supabase.storage.from_(self.bucket_name).get_public_url(unique_filename)
                logger.info(f"Video uploaded successfully: {unique_filename}")
                return public_url, video_uuid
            else:
                logger.error(f"Video upload failed: {getattr(result, 'error', 'Unknown error')}")
                return None, None
                
        except Exception as e:
            logger.error(f"Error uploading video: {e}")
            return None, None
    
class DatabaseManager:
    """Handles database operations for analysis results"""
    
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
    
    def create_analysis(self, user_id: str, video_url: str, thumbnail_url: str, 
                       analysis_summary: Dict[str, Any], feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create analysis records in database with transaction-like behavior.
        
        Args:
            user_id: User identifier
            video_url: URL of processed video
            thumbnail_url: URL of video thumbnail
            analysis_summary: Analysis results summary
            feedback: Generated feedback data
            
        Returns:
            Dict: Operation result with success status and IDs
        """
        inserted_video_id = None
        inserted_feedback_id = None
        
        try:
            # 1. Insert video record
            video_data = {
                "user_id": user_id,
                "video_url": video_url,
                "thumbnail_url": thumbnail_url,
            }
            
            try:
                video_response = self.supabase.table("videos").insert(video_data).execute()
                if not video_response.data:
                    raise Exception(f"Failed to create video record: {video_response.error}")
                
                video_id = video_response.data[0]["video_id"]
                inserted_video_id = video_id
                logger.info(f"Video record created: {video_id}")
                
            except Exception as e:
                logger.error(f"Video table error: {str(e)}")
                return {
                    "success": False,
                    "error": f"Videos table error: {str(e)}",
                    "data": video_data
                }

            # 2. Insert feedback record
            feedback_data = {
                "overall_assessment": feedback.get("overall_assessment", ""),
                "strengths": feedback.get("strengths", []),
                "priority_areas": feedback.get("priority_areas", []),
                "detailed_feedback": feedback.get("detailed_feedback", {})
            }

            try:
                feedback_response = self.supabase.table("feedbacks").insert(feedback_data).execute()
                if not feedback_response.data:
                    raise Exception(f"Failed to create feedback record: {feedback_response.error}")
                
                feedback_id = feedback_response.data[0]["feedback_id"]
                inserted_feedback_id = feedback_id
                logger.info(f"Feedback record created: {feedback_id}")
                
            except Exception as e:
                logger.error(f"Feedback table error: {str(e)}")
                # Rollback video record
                self._rollback_video(inserted_video_id)
                return {
                    "success": False,
                    "error": f"Feedback table error: {str(e)}",
                    "data": feedback_data
                }

            # 3. Insert analysis results
            results_data = {
                "video_id": video_id,
                "user_id": user_id,
                "feedback_id": feedback_id,
                "head_position": analysis_summary.get("head_position", {}).get("median_score", 0),
                "back_position": analysis_summary.get("back_position", {}).get("median_score", 0),
                "arm_flexion": analysis_summary.get("arm_flexion", {}).get("median_score", 0),
                "right_knee": analysis_summary.get("right_knee", {}).get("median_score", 0),
                "left_knee": analysis_summary.get("left_knee", {}).get("median_score", 0),
                "foot_strike": analysis_summary.get("foot_strike", {}).get("median_score", 0),
                "overall_score": analysis_summary.get("overall_score", 0),
            }
            
            try:
                results_response = self.supabase.table("analysis_results").insert(results_data).execute()
                if not results_response.data:
                    raise Exception(f"Failed to create analysis results: {results_response.error}")
                
                analysis_id = results_response.data[0]["id"]
                logger.info(f"Analysis results created: {analysis_id}")
                
            except Exception as e:
                logger.error(f"Analysis results error: {str(e)}")
                # Rollback both records
                self._rollback_feedback(inserted_feedback_id)
                self._rollback_video(inserted_video_id)
                return {
                    "success": False,
                    "error": f"Analysis results error: {str(e)}",
                    "data": results_data
                }
                    
            logger.info("All database records created successfully")
            return {
                "success": True,
                "video_id": video_id,
                "feedback_id": feedback_id,
                "analysis_id": analysis_id
            }
        
        except Exception as e:
            logger.error(f"Unexpected error in create_analysis: {str(e)}")
            # Cleanup any inserted records
            if inserted_feedback_id:
                self._rollback_feedback(inserted_feedback_id)
            if inserted_video_id:
                self._rollback_video(inserted_video_id)
                
            return {
                "success": False,
                "error": f"Error in create_analysis(): {str(e)}"
            }
    
    def _rollback_video(self, video_id: Optional[str]) -> None:
        """Rollback video record."""
        if video_id:
            try:
                self.supabase.table("videos").delete().eq("video_id", video_id).execute()
                logger.info(f"Rolled back video record: {video_id}")
            except Exception as e:
                logger.error(f"Failed to rollback video {video_id}: {str(e)}")
    
    def _rollback_feedback(self, feedback_id: Optional[str]) -> None:
        """Rollback feedback record."""
        if feedback_id:
            try:
                self.supabase.table("feedbacks").delete().eq("feedback_id", feedback_id).execute()
                logger.info(f"Rolled back feedback record: {feedback_id}")
            except Exception as e:
                logger.error(f"Failed to rollback feedback {feedback_id}: {str(e)}")

# Initialize managers
storage_manager = StorageManager(supabase, SUPABASE_BUCKET)
database_manager = DatabaseManager(supabase)

def cleanup_temp(*file_paths):
    """Basic cleanup function - kept for backwards compatibility"""
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted temp file: {file_path}")
        except Exception as e:
            logger.error(f"Error deleting {file_path}: {str(e)}")

def cleanup_temp_enhanced(*file_paths):
    """Enhanced cleanup with better error handling and logging"""
    deleted_count = 0
    failed_count = 0
    
    for file_path in file_paths:
        if not file_path:
            continue
            
        try:
            if os.path.exists(file_path):
                # Check if file is still locked (Windows issue)
                try:
                    # Try to rename file first (tests if it's locked)
                    temp_name = f"{file_path}.deleting"
                    os.rename(file_path, temp_name)
                    os.remove(temp_name)
                    deleted_count += 1
                    logger.info(f"Successfully deleted: {file_path}")
                except OSError as e:
                    if "being used by another process" in str(e).lower():
                        logger.warning(f"File locked, scheduling retry: {file_path}")
                        # Schedule for retry after delay
                        import threading
                        threading.Timer(3.0, retry_delete_file, args=[file_path]).start()
                    else:
                        # Try direct delete as fallback
                        os.remove(file_path)
                        deleted_count += 1
                        logger.info(f"Successfully deleted (fallback): {file_path}")
            else:
                logger.debug(f"File not found for cleanup: {file_path}")
        except Exception as e:
            failed_count += 1
            logger.error(f"Failed to delete {file_path}: {str(e)}")
    
    if deleted_count > 0 or failed_count > 0:
        logger.info(f"Cleanup summary: {deleted_count} deleted, {failed_count} failed")

def retry_delete_file(file_path):
    """Retry deleting a file after delay"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Retry delete successful: {file_path}")
    except Exception as e:
        logger.error(f"Retry delete failed: {file_path} - {e}")

def cleanup_user_tmp_folder(user_id: str):
    """Clean up entire user tmp folder"""
    try:
        user_tmp_dir = f"tmp/{user_id}"
        if os.path.exists(user_tmp_dir):
            import shutil
            shutil.rmtree(user_tmp_dir)
            logger.info(f"Cleaned up user tmp folder: {user_tmp_dir}")
            # Recreate the folders for next use
            os.makedirs(f"{user_tmp_dir}/processed", exist_ok=True)
    except Exception as e:
        logger.error(f"Error cleaning user tmp folder {user_id}: {e}")

def cleanup_old_tmp_files(max_age_hours=24):
    """Clean up tmp files older than specified hours"""
    try:
        current_time = datetime.now().timestamp()
        max_age_seconds = max_age_hours * 3600
        deleted_count = 0
        
        for root, dirs, files in os.walk("tmp"):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_age = current_time - os.path.getctime(file_path)
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        deleted_count += 1
                        logger.info(f"Deleted old tmp file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting old file {file_path}: {e}")
        
        # Clean up empty directories
        for root, dirs, files in os.walk("tmp", topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    if not os.listdir(dir_path):  # Empty directory
                        os.rmdir(dir_path)
                        logger.info(f"Removed empty directory: {dir_path}")
                except Exception as e:
                    logger.debug(f"Could not remove directory {dir_path}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old tmp files")
            
    except Exception as e:
        logger.error(f"Error in cleanup_old_tmp_files: {e}")

@app.post("/process-video/")
async def process_video(
    file: UploadFile = File(...),
    user_id: str = Form(...)    
):
    # Initialize processors
    video_processor = VideoProcessor()
    # storage_manager = StorageManager()
    # database_manager = DatabaseManager()
    
    # initialize all variables at the start
    contact_results = {"right_landing": False}
    final_video_path = None
    cap = None  # Initialize cap variable
    out = None  # Initialize out variable

    print("uid: ", user_id)
    # create dir
    os.makedirs(f"tmp/{user_id}/processed", exist_ok=True) 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_filename = f"{timestamp}_{file.filename}"

    # save video to temporary loc
    video_path = f"tmp/{user_id}/{file.filename}"
    annotated_video_path = f"tmp/{user_id}/processed/processed_{file.filename}"

    try:
        with open(video_path, "wb") as buffer:
            buffer.write(await file.read())

        video_uuid = str(uuid.uuid4())

        thumbnail_path = f"tmp/{user_id}/thumbnail_{timestamp}.jpg"
        thumbnail_extracted = video_processor.extract_thumbnail(video_path, thumbnail_path, 1.0)

        thumbnail_url = None
        if thumbnail_extracted:
            thumbnail_url = storage_manager.upload_thumbnail(user_id, thumbnail_path, video_uuid)

        # process with MP pose
        cap = cv2.VideoCapture(video_path)
        
        # ✓ Detect rotation and fix video orientation BEFORE processing
        rotation = VideoProcessor.get_video_rotation(video_path)
        logger.info(f"Detected rotation for processing: {rotation}°")
        
        # If rotation is needed, create a corrected video file first
        corrected_video_path = video_path
        if rotation != 0:
            logger.info(f"Pre-correcting video orientation before pose analysis")
            corrected_video_path = f"tmp/{user_id}/corrected_{file.filename}"
            
            # Apply rotation correction to the video file itself
            if rotation == 90 or rotation == 270:
                # Use counter-clockwise rotation for mobile videos
                transpose_cmd = "transpose=2"  # 90° counter-clockwise
            elif rotation == 180:
                transpose_cmd = "transpose=1,transpose=1"  # 180°
            else:
                transpose_cmd = "transpose=2"  # Default to counter-clockwise
                
            correction_cmd = (
                f'ffmpeg -i "{video_path}" -vf "{transpose_cmd}" '
                f'-c:v libx264 -crf 23 -preset fast "{corrected_video_path}" -y'
            )
            
            result = os.system(correction_cmd)
            if result != 0:
                logger.warning(f"Video correction failed, using original video")
                corrected_video_path = video_path
            else:
                logger.info(f"Video corrected successfully: {corrected_video_path}")
        
        # Now process the corrected video
        cap = cv2.VideoCapture(corrected_video_path)
        
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Use the actual dimensions from the corrected video
        width = original_width
        height = original_height
        logger.info(f"Processing corrected video dimensions: {width}x{height}")

        # -- ready output
        fourcc = cv2.VideoWriter_fourcc(*'avc1') # codec for output video
        out = cv2.VideoWriter(annotated_video_path, fourcc, 30.0, (width,height), isColor=True)

        prev_lmList = None
        measure_list = None

        if not cap.isOpened():
            raise Exception("Error: cannot open video file")

        # ✓ Log processing information
        logger.info(f"Starting pose processing on corrected video")
        logger.info(f"Video dimensions: {width}x{height}")
        
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read() # process the frame
            if not ret: # end video if error
                print("video ended or frame read failed")
                break

            frame_count += 1
            
            # ✓ Log frame dimensions for first few frames
            if frame_count <= 3:
                h, w = frame.shape[:2]
                logger.info(f"Frame {frame_count} dimensions: {w}x{h}")

            # No need to rotate frames anymore - video is already corrected

            frame = detector.findPose(frame)
            lmList = detector.findPosition(frame)
            if lmList:
                # extract angles per frame
                head_position = detector.findHeadPosition(frame, draw=True)
                back_position = detector.findTorsoLean(frame, draw=True)
                arm_flexion = detector.findAngle(frame, 12, 14, 16, draw=True)
                left_knee = detector.findKneeAngle(frame, 23, 25, 27, draw=True)
                right_knee = detector.findKneeAngle(frame, 24, 26, 28, draw=True)
                foot_strike = detector.findFootAngle(frame, draw=True)
                contact_results = detector.detectFootContact_Alternative(frame, draw=True)
                
                # ✓ Add validation for None values
                if all(angle is not None for angle in [head_position, back_position, arm_flexion, left_knee, right_knee, foot_strike]):
                    if contact_results and contact_results.get('right_landing', False):
                        frame_angles = [head_position, back_position, arm_flexion, 
                                    left_knee, right_knee, foot_strike]
                        analyzer.analyze_frame(frame_angles)  # Score this frame
            else:
                print("No landmarks detected in frame")
                cv2.putText(frame, "No pose detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # After processing all frames:
            out.write(frame) # write annotated frame to the output video

            # press q to quit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
        summary = analyzer.get_summary()  # Get aggregated results
        print("Analysis summary:", summary)

        # Properly release resources before file operations
        if cap:
            cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        # Add small delay to ensure file handles are released
        import time
        time.sleep(0.1)

        final_video_path = video_processor.add_faststart(annotated_video_path)
        download_url, _ = storage_manager.upload_video(user_id, final_video_path, f"processed_{original_filename}", video_uuid)

        if not download_url:
            raise HTTPException(
                status_code=500,
                detail="Failed to upload video to supabase storage"
            )
        
        feedback = await generate_feedback(summary, user_id, drill_manager)
        # print(feedback)

        analysis_result = database_manager.create_analysis(user_id, download_url, thumbnail_url, summary, feedback)
        
        response_data = {
            "user_id": user_id,
            "success": True,
            "message": "Video processing successful",
            "download_url": download_url,
            "thumbnail_url": thumbnail_url,
            "analysis_summary": summary
        }
        response_data["database_records"] = analysis_result

        # print(response_data)

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Full error details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Ensure resources are released even if there's an error
        try:
            if 'cap' in locals() and cap:
                cap.release()
            if 'out' in locals() and out:
                out.release()
            cv2.destroyAllWindows()
        except Exception as e:
            logger.error(f"Error releasing video resources: {e}")
        
        # Add delay before cleanup (important on Windows)
        import time
        time.sleep(0.3)
        
        # Build cleanup list with proper error handling
        cleanup_files = []
        
        # Add files that should always be cleaned up
        if 'video_path' in locals() and video_path:
            cleanup_files.append(video_path)
        if 'thumbnail_path' in locals() and thumbnail_path:
            cleanup_files.append(thumbnail_path)
        if 'annotated_video_path' in locals() and annotated_video_path:
            cleanup_files.append(annotated_video_path)
        
        # Add conditional files
        if 'final_video_path' in locals() and final_video_path:
            cleanup_files.append(final_video_path)
        
        # Add corrected video if it was created and is different from original
        try:
            if 'rotation' in locals() and 'corrected_video_path' in locals():
                if rotation != 0 and corrected_video_path != video_path:
                    cleanup_files.append(corrected_video_path)
        except:
            pass
        
        # Use enhanced cleanup
        cleanup_temp_enhanced(*cleanup_files)
        
        # Also try to clean up the entire user folder if processing failed
        try:
            if 'user_id' in locals() and user_id:
                # Check if there are any leftover files
                user_tmp_dir = f"tmp/{user_id}"
                if os.path.exists(user_tmp_dir):
                    files_in_dir = []
                    for root, dirs, files in os.walk(user_tmp_dir):
                        files_in_dir.extend([os.path.join(root, f) for f in files])
                    
                    if len(files_in_dir) > 5:  # Too many leftover files
                        logger.warning(f"Many leftover files detected for user {user_id}, cleaning up folder")
                        cleanup_user_tmp_folder(user_id)
        except Exception as e:
            logger.error(f"Error in additional cleanup: {e}")

@app.get("/cleanup-status/")
async def cleanup_status():
    """Check for leftover temporary files"""
    tmp_files = []
    total_size = 0
    
    try:
        if os.path.exists("tmp"):
            for root, dirs, files in os.walk("tmp"):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_size = os.path.getsize(file_path)
                        file_time = os.path.getctime(file_path)
                        age_hours = (datetime.now().timestamp() - file_time) / 3600
                        
                        total_size += file_size
                        tmp_files.append({
                            "path": file_path,
                            "size_mb": round(file_size / (1024*1024), 2),
                            "age_hours": round(age_hours, 2),
                            "created": datetime.fromtimestamp(file_time).isoformat()
                        })
                    except Exception as e:
                        logger.error(f"Error getting file info for {file_path}: {e}")
        
        # Sort by age (oldest first)
        tmp_files.sort(key=lambda x: x["age_hours"], reverse=True)
        
    except Exception as e:
        logger.error(f"Error checking tmp files: {e}")
        return {"error": str(e)}
    
    return {
        "tmp_file_count": len(tmp_files),
        "total_size_mb": round(total_size / (1024*1024), 2),
        "oldest_files": tmp_files[:10],  # Show 10 oldest files
        "large_files": sorted([f for f in tmp_files if f["size_mb"] > 10], 
                             key=lambda x: x["size_mb"], reverse=True)[:5]
    }

@app.post("/cleanup-old-files/")
async def cleanup_old_files(max_age_hours: int = 24):
    """Force cleanup of old temporary files"""
    try:
        cleanup_old_tmp_files(max_age_hours)
        return {"message": f"Cleanup completed for files older than {max_age_hours} hours"}
    except Exception as e:
        logger.error(f"Error in cleanup endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@app.post("/force-cleanup/")
async def force_cleanup():
    """Force cleanup of entire tmp directory (use with caution)"""
    try:
        import shutil
        if os.path.exists("tmp"):
            shutil.rmtree("tmp")
            logger.info("Force cleanup: removed entire tmp directory")
        
        # Recreate tmp directory
        os.makedirs("tmp", exist_ok=True)
        logger.info("Recreated tmp directory")
        
        return {"message": "Force cleanup completed - entire tmp directory recreated"}
    except Exception as e:
        logger.error(f"Error in force cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Force cleanup failed: {str(e)}")

@app.post("/cleanup-user/{user_id}")
async def cleanup_user_files(user_id: str):
    """Clean up files for a specific user"""
    try:
        cleanup_user_tmp_folder(user_id)
        return {"message": f"Cleaned up files for user {user_id}"}
    except Exception as e:
        logger.error(f"Error cleaning up user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"User cleanup failed: {str(e)}")


@app.post("/test-both-rotations/")
async def test_both_rotations(file: UploadFile = File(...)):
    """
    Test both clockwise and counter-clockwise rotations to determine which is correct.
    This helps debug upside-down video issues.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_dir = f"tmp/rotation_test_{timestamp}"
        os.makedirs(test_dir, exist_ok=True)
        
        # Save original video
        original_path = f"{test_dir}/original_{file.filename}"
        with open(original_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Get rotation metadata
        rotation = VideoProcessor.get_video_rotation(original_path)
        
        # Read first frame to test rotations
        cap = cv2.VideoCapture(original_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise HTTPException(status_code=400, detail="Could not read video frame")
        
        original_shape = frame.shape
        
        # Test both rotation directions
        clockwise = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        counterclockwise = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Save test frames as images for visual inspection
        cv2.imwrite(f"{test_dir}/original_frame.jpg", frame)
        cv2.imwrite(f"{test_dir}/clockwise_frame.jpg", clockwise)
        cv2.imwrite(f"{test_dir}/counterclockwise_frame.jpg", counterclockwise)
        
        results = {
            "detected_rotation": rotation,
            "original_shape": f"{original_shape[1]}x{original_shape[0]}",
            "clockwise_shape": f"{clockwise.shape[1]}x{clockwise.shape[0]}",
            "counterclockwise_shape": f"{counterclockwise.shape[1]}x{counterclockwise.shape[0]}",
            "test_frames": {
                "original": f"{test_dir}/original_frame.jpg",
                "clockwise": f"{test_dir}/clockwise_frame.jpg", 
                "counterclockwise": f"{test_dir}/counterclockwise_frame.jpg"
            },
            "current_setting": "counter-clockwise (to fix upside-down issue)",
            "recommendation": "Check the test frames to see which rotation looks correct"
        }
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Rotation test error: {e}")
        raise HTTPException(status_code=500, detail=f"Rotation test failed: {str(e)}")


@app.post("/debug-rotation/")
async def debug_rotation(file: UploadFile = File(...)):
    """Debug endpoint to test video rotation detection and correction"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_dir = f"debug/{timestamp}"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save original video
        original_path = f"{debug_dir}/original_{file.filename}"
        with open(original_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Test rotation detection
        VideoProcessor.test_video_rotation(original_path)
        
        # Create test videos with different rotations
        test_videos = VideoProcessor.create_rotation_test_videos(original_path)
        
        return JSONResponse(content={
            "message": "Debug videos created",
            "original_video": original_path,
            "test_videos": test_videos,
            "instructions": "Check each test video to see which orientation is correct"
        })
        
    except Exception as e:
        logger.error(f"Debug rotation error: {e}")
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")

# Startup cleanup - remove old tmp files when server starts
try:
    logger.info("Performing startup cleanup of old tmp files...")
    cleanup_old_tmp_files(max_age_hours=6)  # Clean files older than 6 hours on startup
    logger.info("Startup cleanup completed")
except Exception as e:
    logger.error(f"Startup cleanup failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)