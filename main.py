"""
RunAnalyze - AI-powered running form analysis API.

This module provides endpoints for video analysis, drill suggestions,
and comprehensive running form feedback using MediaPipe pose detection.
"""
import psutil
import tracemalloc
import gc
import time
import threading
import shutil
from per_frame_memory_monitor import (
    PerFrameMemoryMonitor, 
    start_frame_memory_monitoring,
    log_frame_memory,
    stop_frame_memory_monitoring
)

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

# Import analysis modules
import PoseModule as pm
import RFAnalyzer as rfa

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

# Enhanced memory monitoring
class MemoryTracker:
    """Enhanced memory tracking for video processing"""
    
    def __init__(self):
        self.start_memory = None
        self.peak_memory = 0
        self.memory_history = []
        self.stage_memories = {}
        
    def start_tracking(self):
        """Start memory tracking session"""
        tracemalloc.start()
        process = psutil.Process()
        self.start_memory = process.memory_info().rss / (1024**2)  # MB
        self.peak_memory = self.start_memory
        self.memory_history = []
        self.stage_memories = {}
        logger.info(f"MEMORY TRACKING STARTED - Baseline: {self.start_memory:.1f}MB")
        
    def log_memory(self, stage: str = "", video_info: str = ""):
        """Enhanced memory logging with detailed tracking"""
        try:
            # Get current process memory
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024**2)
            memory_percent = process.memory_percent()
            
            # Track peak memory
            if memory_mb > self.peak_memory:
                self.peak_memory = memory_mb
            
            # Get system memory
            system_memory = psutil.virtual_memory()
            
            # Get tracemalloc info if available
            tracemalloc_current = 0
            tracemalloc_peak = 0
            try:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc_current = current / (1024**2)  # MB
                tracemalloc_peak = peak / (1024**2)  # MB
            except:
                pass
            
            # Calculate memory increase from start
            memory_increase = memory_mb - (self.start_memory or memory_mb)
            
            # Store memory data
            memory_data = {
                "timestamp": datetime.now().isoformat(),
                "stage": stage,
                "video_info": video_info,
                "process_memory_mb": round(memory_mb, 1),
                "memory_increase_mb": round(memory_increase, 1),
                "memory_percent": round(memory_percent, 2),
                "system_used_percent": round(system_memory.percent, 1),
                "system_available_gb": round(system_memory.available / (1024**3), 2),
                "tracemalloc_current_mb": round(tracemalloc_current, 1),
                "tracemalloc_peak_mb": round(tracemalloc_peak, 1)
            }
            
            self.memory_history.append(memory_data)
            self.stage_memories[stage] = memory_data
            
            # Enhanced logging
            log_msg = (f"MEMORY [{stage}] - "
                      f"Process: {memory_mb:.1f}MB (+{memory_increase:+.1f}MB) ({memory_percent:.1f}%), "
                      f"System: {system_memory.percent:.1f}% used, "
                      f"Available: {system_memory.available / (1024**3):.1f}GB")
            
            if video_info:
                log_msg += f", Video: {video_info}"
                
            if tracemalloc_current > 0:
                log_msg += f", TraceMalloc: {tracemalloc_current:.1f}MB"
            
            logger.info(log_msg)
            
            return memory_data
            
        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return None
    
    def get_summary(self):
        """Get memory usage summary"""
        if not self.memory_history:
            return {"error": "No memory data available"}
            
        return {
            "start_memory_mb": round(self.start_memory, 1),
            "peak_memory_mb": round(self.peak_memory, 1),
            "total_increase_mb": round(self.peak_memory - self.start_memory, 1),
            "stage_count": len(self.stage_memories),
            "memory_efficient_stages": [s for s, d in self.stage_memories.items() 
                                       if d["memory_increase_mb"] < 50],
            "memory_intensive_stages": [s for s, d in self.stage_memories.items() 
                                       if d["memory_increase_mb"] > 100],
            "final_memory_mb": round(self.memory_history[-1]["process_memory_mb"], 1) if self.memory_history else 0
        }
    
    def stop_tracking(self):
        """Stop memory tracking and return final summary"""
        try:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            summary = self.get_summary()
            summary.update({
                "tracemalloc_final_mb": round(current / (1024**2), 1),
                "tracemalloc_peak_mb": round(peak / (1024**2), 1)
            })
            
            logger.info(f"MEMORY TRACKING STOPPED - Peak: {self.peak_memory:.1f}MB, "
                       f"Total Increase: {self.peak_memory - self.start_memory:.1f}MB")
            
            return summary
        except Exception as e:
            logger.error(f"Error stopping memory tracking: {e}")
            return self.get_summary()

# Initialize global memory tracker
memory_tracker = MemoryTracker()

def log_memory_usage(stage: str = "", video_info: str = ""):
    """Log current memory usage with enhanced tracking"""
    return memory_tracker.log_memory(stage, video_info)

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
    codec: str = 'mp4v'
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
            log_memory_usage("END")
            
            # Get memory allocation stats
            current, peak = tracemalloc.get_traced_memory()
            logger.info(f"TRACEMALLOC - Current: {current / (1024**2):.1f}MB, Peak: {peak / (1024**2):.1f}MB")
            tracemalloc.stop()

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
        Add faststart metadata to video for web streaming with Render.com optimizations.
        
        AVC/H.264 Optimization Strategy:
        - Primary: libx264 with ultrafast preset and baseline profile
        - Fallback 1: Even faster H.264 with veryfast + zerolatency
        - Final: Stream copy (no re-encoding)
        
        Render.com Optimizations:
        - CRF 32 (good compression/speed balance)
        - ultrafast preset (fastest H.264 encoding)
        - baseline profile (maximum compatibility)
        - fastdecode tune (optimized for playback)
        - Limited threads (2 → 1 for fallbacks)
        - Progressive timeouts: 60s → 30s → 15s
        
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
            logger.info("Using Render.com optimized FFmpeg settings: AVC (H.264) with fast encoding")
            
            # Use AVC (H.264) with optimized settings for Render.com
            autorotate_cmd = (
                f'ffmpeg -noautorotate -i "{input_path}" -c:v libx264 '
                f'-crf 32 -preset ultrafast -tune fastdecode -profile:v baseline '
                f'-movflags +faststart -threads 2 -max_muxing_queue_size 1024 '
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
                
                # Build the video filter chain with manual rotation (AVC optimized)
                video_filters = f"{transpose_filter}scale=trunc(iw/2)*2:trunc(ih/2)*2"
                
                manual_cmd = (
                    f'ffmpeg -noautorotate -i "{input_path}" -c:v libx264 '
                    f'-crf 32 -preset ultrafast -tune fastdecode -profile:v baseline '
                    f'-movflags +faststart -threads 2 -max_muxing_queue_size 1024 '
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
            
            # Execute with timeout for Render.com (max 60 seconds for FFmpeg)
            try:
                result = subprocess.run(
                    cmd_to_use, 
                    shell=True, 
                    timeout=60,  # 60 second timeout
                    capture_output=True, 
                    text=True
                )
                
                if result.returncode == 0:
                    logger.info(f"Video processing completed with orientation fix: {output_path}")
                    return output_path
                else:
                    logger.error(f"FFmpeg processing failed with code: {result.returncode}")
                    logger.error(f"FFmpeg error: {result.stderr}")
                    raise subprocess.CalledProcessError(result.returncode, cmd_to_use)
                    
            except subprocess.TimeoutExpired:
                logger.warning("AVC encoding timed out after 60 seconds, trying faster fallbacks")
                
                # Fallback 1: Even faster H.264 encoding
                faster_h264_cmd = (
                    f'ffmpeg -i "{input_path}" -c:v libx264 '
                    f'-crf 35 -preset veryfast -tune zerolatency -profile:v baseline '
                    f'-movflags +faststart -threads 1 -max_muxing_queue_size 512 '
                    f'-metadata:s:v rotate=0 "{output_path}" -y'
                )
                
                try:
                    result = subprocess.run(
                        faster_h264_cmd, 
                        shell=True, 
                        timeout=30,
                        capture_output=True, 
                        text=True
                    )
                    
                    if result.returncode == 0:
                        logger.info(f"Faster H.264 fallback completed: {output_path}")
                        return output_path
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                    logger.warning("Faster H.264 fallback also failed/timed out")
                
                # Fallback 2: Stream copy only (no re-encoding)
                copy_fallback_cmd = (
                    f'ffmpeg -i "{input_path}" -c copy -movflags +faststart '
                    f'-metadata:s:v rotate=0 "{output_path}" -y'
                )
                
                try:
                    result = subprocess.run(
                        copy_fallback_cmd, 
                        shell=True, 
                        timeout=15,
                        capture_output=True, 
                        text=True
                    )
                    
                    if result.returncode == 0:
                        logger.info(f"Stream copy fallback completed: {output_path}")
                        return output_path
                    else:
                        logger.error("Stream copy fallback failed, returning original")
                        return input_path
                        
                except subprocess.TimeoutExpired:
                    logger.error("Even stream copy timed out, returning original video")
                    return input_path
                    
            except subprocess.CalledProcessError:
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
                            f'ffmpeg -noautorotate -i "{input_path}" -c:v libx264 '
                            f'-crf 32 -preset ultrafast -tune fastdecode -profile:v baseline '
                            f'-movflags +faststart -threads 2 -max_muxing_queue_size 1024 '
                            f'-vf "{alt_transpose},scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                            f'-metadata:s:v rotate=0 "{alt_output}" -y'
                        )
                        
                        try:
                            result = subprocess.run(
                                alt_cmd, 
                                shell=True, 
                                timeout=45,  # Shorter timeout for alternative
                                capture_output=True, 
                                text=True
                            )
                            
                            if result.returncode == 0:
                                logger.info(f"Alternative rotation successful: {alt_output}")
                                return alt_output
                        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                            logger.warning("Alternative rotation also failed/timed out")
                    
                    # Final fallback to autorotate with timeout
                    logger.info("Trying autorotate fallback...")
                    try:
                        result = subprocess.run(
                            autorotate_cmd, 
                            shell=True, 
                            timeout=45,
                            capture_output=True, 
                            text=True
                        )
                        
                        if result.returncode == 0:
                            logger.info(f"Video processing completed with autorotate fallback: {output_path}")
                            return output_path
                    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                        logger.warning("Autorotate fallback also failed/timed out")
                
                # Final ultra-fast fallback
                logger.info("All encoding attempts failed, trying copy-only fallback")
                try:
                    copy_cmd = f'ffmpeg -i "{input_path}" -c copy -movflags +faststart "{output_path}" -y'
                    result = subprocess.run(
                        copy_cmd, 
                        shell=True, 
                        timeout=20,
                        capture_output=True, 
                        text=True
                    )
                    
                    if result.returncode == 0:
                        logger.info(f"Copy-only fallback successful: {output_path}")
                        return output_path
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                    logger.error("Even copy-only fallback failed")
                
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

# Memory Optimization Functions
def get_optimal_processing_size(original_width, original_height, max_memory_mb=400):
    """
    Dynamically scale video resolution based on available memory to reduce processing load.
    
    Args:
        original_width: Original video width
        original_height: Original video height
        max_memory_mb: Maximum memory to use for frame processing
        
    Returns:
        Tuple: (new_width, new_height, scale_factor)
    """
    # Calculate memory usage for full resolution (RGBA float32)
    bytes_per_frame = original_width * original_height * 3 * 4  # RGB float
    estimated_memory_mb = bytes_per_frame / (1024**2)
    
    if estimated_memory_mb <= max_memory_mb:
        return original_width, original_height, 1.0
    
    # Calculate scale factor to fit within memory limit
    scale_factor = (max_memory_mb / estimated_memory_mb) ** 0.5
    new_width = int(original_width * scale_factor / 2) * 2  # Even numbers for video encoding
    new_height = int(original_height * scale_factor / 2) * 2
    
    logger.info(f"Scaling video from {original_width}x{original_height} to {new_width}x{new_height} "
               f"(scale: {scale_factor:.2f}) to reduce memory usage from {estimated_memory_mb:.1f}MB to {max_memory_mb}MB")
    
    return new_width, new_height, scale_factor

def process_frame_with_scaling(frame, detector, processing_width, processing_height, 
                              original_width, original_height, scale_factor, rotation=0):
    """
    Process frame at reduced resolution for memory efficiency, with optional rotation, then scale back for output.
    
    Args:
        frame: Original frame
        detector: Pose detector
        processing_width: Width for processing
        processing_height: Height for processing
        original_width: Original frame width
        original_height: Original frame height
        scale_factor: Scaling factor applied
        rotation: Rotation angle to apply (0, 90, 180, 270)
        
    Returns:
        Tuple: (output_frame, lmList, analysis_data)
    """
    try:
        # Apply rotation first if needed
        if rotation != 0:
            frame = smart_rotate_frame(frame, rotation)
        
        # Scale down for processing if needed
        if scale_factor != 1.0:
            processing_frame = cv2.resize(frame, (processing_width, processing_height))
        else:
            processing_frame = frame.copy()
        
        # Do pose detection on smaller frame
        processing_frame = detector.findPose(processing_frame)
        lmList = detector.findPosition(processing_frame)
        
        analysis_data = None
        if lmList:
            # Extract measurements (these work regardless of scale)
            head_position = detector.findHeadPosition(processing_frame, draw=True)
            back_position = detector.findTorsoLean(processing_frame, draw=True)
            arm_flexion = detector.findAngle(processing_frame, 12, 14, 16, draw=True)
            left_knee = detector.findKneeAngle(processing_frame, 23, 25, 27, draw=True)
            right_knee = detector.findKneeAngle(processing_frame, 24, 26, 28, draw=True)
            foot_strike = detector.findFootAngle(processing_frame, draw=True)
            
            # Check for valid measurements
            if all(angle is not None for angle in [head_position, back_position, arm_flexion, left_knee, right_knee, foot_strike]):
                analysis_data = [head_position, back_position, arm_flexion, left_knee, right_knee, foot_strike]
        
        # Scale back to original size for output
        if scale_factor != 1.0:
            output_frame = cv2.resize(processing_frame, (original_width, original_height))
        else:
            output_frame = processing_frame
        
        # Cleanup intermediate frame
        if scale_factor != 1.0:
            del processing_frame
        
        return output_frame, lmList, analysis_data
        
    except Exception as e:
        logger.error(f"Error in process_frame_with_scaling: {e}")
        return frame, None, None

def process_video_streaming_optimized(cap, out, detector, analyzer, total_frames, 
                                    processing_width, processing_height, 
                                    original_width, original_height, scale_factor, rotation=0):
    """
    Stream-based video processing with enhanced per-frame memory optimization and monitoring.
    
    Args:
        cap: Video capture object
        out: Video writer object
        detector: Pose detector
        analyzer: Analysis object
        total_frames: Total number of frames
        processing_width: Width for processing
        processing_height: Height for processing
        original_width: Original frame width
        original_height: Original frame height
        scale_factor: Scaling factor
        rotation: Rotation angle to apply per frame
        
    Returns:
        Tuple: (frame_count, processed_count, processing_time, memory_stats)
    """
    frame_count = 0
    processed_count = 0
    start_time = datetime.now()
    batch_size = 15  # Process in small batches for memory management
    
    logger.info(f"Starting optimized streaming processing: {processing_width}x{processing_height} -> {original_width}x{original_height}")
    
    # Start per-frame memory monitoring
    frame_monitor = start_frame_memory_monitoring(
        log_every_n_frames=25,  # Log every 25 frames
        detailed_log_every_n_frames=100,  # Detailed log every 100 frames
        memory_alert_threshold_mb=2000,  # Alert at 2GB
        enable_tracemalloc=True
    )
    
    try:
        while True:
            # Process in batches to manage memory
            batch_frames = []
            batch_start = frame_count
            
            # Read batch of frames
            for i in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                batch_frames.append(frame)
                frame_count += 1
            
            if not batch_frames:
                break
            
            # Process batch
            for i, frame in enumerate(batch_frames):
                current_frame_num = batch_start + i + 1
                
                try:
                    # Log memory before frame processing
                    log_frame_memory(
                        current_frame_num, 
                        total_frames, 
                        "frame_start",
                        {"batch_size": len(batch_frames), "batch_position": i}
                    )
                    
                    # Process frame with memory optimization and rotation
                    output_frame, lmList, analysis_data = process_frame_with_scaling(
                        frame, detector, processing_width, processing_height,
                        original_width, original_height, scale_factor, rotation
                    )
                    
                    # Log memory after pose detection
                    log_frame_memory(
                        current_frame_num, 
                        total_frames, 
                        "pose_detection",
                        {"landmarks_detected": len(lmList) > 0, "landmarks_count": len(lmList)}
                    )
                    
                    # Analyze frame if valid data
                    if analysis_data:
                        # Check for foot contact (simplified for now)
                        if hasattr(detector, 'detectFootContact_Alternative'):
                            contact_results = detector.detectFootContact_Alternative(output_frame, draw=True)
                        else:
                            contact_results = {"right_landing": current_frame_num % 2 == 0}
                        
                        if contact_results and contact_results.get('right_landing', False):
                            analyzer.analyze_frame(analysis_data)
                            processed_count += 1
                            
                            # Log memory after analysis
                            log_frame_memory(
                                current_frame_num, 
                                total_frames, 
                                "analysis_complete",
                                {"contact_detected": True, "processed_count": processed_count}
                            )
                    
                    # Add frame number overlay for debugging
                    if not lmList:
                        cv2.putText(output_frame, "No pose detected", (50, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Write to output immediately
                    out.write(output_frame)
                    
                    # Log memory after writing
                    log_frame_memory(
                        current_frame_num, 
                        total_frames, 
                        "frame_written",
                        {"video_writing": True}
                    )
                    
                    # Immediate cleanup
                    del output_frame
                    del frame
                    
                except Exception as e:
                    logger.error(f"Error processing frame {current_frame_num}: {e}")
                    # Write original frame if processing fails
                    out.write(frame)
                    del frame
                    
                    # Log memory for failed frame
                    log_frame_memory(
                        current_frame_num, 
                        total_frames, 
                        "frame_error",
                        {"error": str(e)}
                    )
            
            # Clear batch from memory
            del batch_frames
            gc.collect()  # Force garbage collection after each batch
            
            # Log memory after batch cleanup
            if frame_count > 0:
                log_frame_memory(
                    frame_count, 
                    total_frames, 
                    "batch_cleanup",
                    {"batch_completed": True, "gc_triggered": True}
                )
            
            # Progress logging
            if frame_count % 50 == 0:
                elapsed_time = (datetime.now() - start_time).total_seconds()
                fps_processing = frame_count / elapsed_time if elapsed_time > 0 else 0
                percent_complete = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                estimated_remaining = ((total_frames - frame_count) / fps_processing) if fps_processing > 0 else 0
                
                progress_info = f"Frame {frame_count}/{total_frames} ({percent_complete:.1f}%), {fps_processing:.1f}fps, ETA: {estimated_remaining:.1f}s"
                print(f"OPTIMIZED PROCESSING: {progress_info}")
                log_memory_usage(f"STREAMING_BATCH_{frame_count//batch_size}", progress_info)
    
    except Exception as e:
        logger.error(f"Error in streaming processing: {e}")
    
    finally:
        # Stop frame memory monitoring and get statistics
        memory_stats = stop_frame_memory_monitoring()
        
        # Export detailed memory analysis
        if frame_count > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            memory_export_path = f"tmp/memory_analysis_{timestamp}.json"
            frame_monitor.export_frame_data(memory_export_path)
            logger.info(f"📊 Detailed memory analysis exported to: {memory_export_path}")
    
    processing_time = (datetime.now() - start_time).total_seconds()
    return frame_count, processed_count, processing_time, memory_stats

def immediate_resource_cleanup(*objects):
    """
    Immediately cleanup specified objects and force garbage collection.
    
    Args:
        *objects: Objects to delete and cleanup
    """
    try:
        for obj in objects:
            if obj is not None:
                del obj
        gc.collect()  # Force garbage collection
    except Exception as e:
        logger.error(f"Error in immediate_resource_cleanup: {e}")

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
    # Start enhanced memory tracking
    memory_tracker.start_tracking()
    
    # Initialize processors
    video_processor = VideoProcessor()
    
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
        # === STAGE 1: FILE UPLOAD ===
        print("=== STAGE 1: UPLOADING FILE ===")
        with open(video_path, "wb") as buffer:
            file_content = await file.read()
            buffer.write(file_content)

        file_size_mb = len(file_content) / (1024 * 1024)
        log_memory_usage("AFTER_FILE_UPLOAD", f"File: {file.filename} ({file_size_mb:.1f}MB)")

        # === STAGE 2: THUMBNAIL EXTRACTION ===
        print("=== STAGE 2: EXTRACTING THUMBNAIL ===")
        log_memory_usage("BEFORE_THUMBNAIL_EXTRACTION")
        
        video_uuid = str(uuid.uuid4())
        thumbnail_path = f"tmp/{user_id}/thumbnail_{timestamp}.jpg"
        thumbnail_extracted = video_processor.extract_thumbnail(video_path, thumbnail_path, 1.0)
        
        log_memory_usage("AFTER_THUMBNAIL_EXTRACTION")

        thumbnail_url = None
        if thumbnail_extracted:
            print("=== STAGE 3: UPLOADING THUMBNAIL ===")
            log_memory_usage("BEFORE_THUMBNAIL_UPLOAD")
            thumbnail_url = storage_manager.upload_thumbnail(user_id, thumbnail_path, video_uuid)
            log_memory_usage("AFTER_THUMBNAIL_UPLOAD")

        # === STAGE 4: VIDEO ORIENTATION DETECTION ===
        print("=== STAGE 4: DETECTING VIDEO ORIENTATION ===")
        log_memory_usage("BEFORE_ROTATION_DETECTION")
        
        # process with MP pose
        cap = cv2.VideoCapture(video_path)
        
        # ✓ Detect rotation and fix video orientation BEFORE processing
        rotation = VideoProcessor.get_video_rotation(video_path)
        logger.info(f"Detected rotation for processing: {rotation}°")
        
        log_memory_usage("AFTER_ROTATION_DETECTION", f"Rotation: {rotation}°")
        
        # === STAGE 5: OPTIMIZED VIDEO SETUP (STREAM PROCESSING) ===
        print("=== STAGE 5: SETTING UP STREAM PROCESSING ===")
        log_memory_usage("BEFORE_STREAM_SETUP")
        
        # Instead of creating corrected video file, we'll apply rotation per frame
        # This saves disk space and memory by avoiding temporary file creation
        corrected_video_path = video_path  # Keep reference to original
        apply_rotation_per_frame = rotation != 0
        
        if apply_rotation_per_frame:
            logger.info(f"Will apply {rotation}° rotation per frame during streaming (no temp file)")
        else:
            logger.info("No rotation needed, processing original video directly")
        
        log_memory_usage("AFTER_STREAM_SETUP", f"Stream mode, rotation: {rotation}°")
        
        # === STAGE 6: VIDEO ANALYSIS SETUP ===
        print("=== STAGE 6: SETTING UP VIDEO ANALYSIS ===")
        log_memory_usage("BEFORE_VIDEO_ANALYSIS_SETUP")
        
        # Process the original video (rotation applied per frame)
        cap = cv2.VideoCapture(video_path)
        
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate corrected dimensions for output video
        if apply_rotation_per_frame and (rotation == 90 or rotation == 270):
            # For 90/270° rotation, swap width and height
            width = original_height
            height = original_width
            logger.info(f"Output dimensions swapped for rotation: {width}x{height}")
        else:
            width = original_width
            height = original_height
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration_seconds = total_frames / fps if fps > 0 else 0
        
        # Use the actual dimensions from the corrected video
        width = original_width
        height = original_height
        
        video_info = f"{width}x{height}, {total_frames}f, {duration_seconds:.1f}s"
        logger.info(f"Processing corrected video dimensions: {width}x{height}")
        print(f"Video properties: {video_info}")
        
        log_memory_usage("AFTER_VIDEO_ANALYSIS_SETUP", video_info)

        # === STAGE 7: OUTPUT VIDEO WRITER SETUP ===
        print("=== STAGE 7: SETTING UP OUTPUT VIDEO WRITER ===")
        log_memory_usage("BEFORE_OUTPUT_WRITER_SETUP")
        
        # -- ready output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # codec for output video
        out = cv2.VideoWriter(annotated_video_path, fourcc, 30.0, (width,height), isColor=True)

        prev_lmList = None
        measure_list = None

        if not cap.isOpened():
            raise Exception("Error: cannot open video file")

        log_memory_usage("AFTER_OUTPUT_WRITER_SETUP")

        # === STAGE 8: MEMORY-OPTIMIZED PROCESSING ===
        print("=== STAGE 8: STARTING MEMORY-OPTIMIZED PROCESSING ===")
        
        # Get optimal processing resolution for memory efficiency
        processing_width, processing_height, scale_factor = get_optimal_processing_size(
            original_width, original_height, max_memory_mb=400
        )
        
        logger.info(f"Video processing optimization:")
        logger.info(f"  Original: {original_width}x{original_height}")
        logger.info(f"  Processing: {processing_width}x{processing_height}")
        logger.info(f"  Scale factor: {scale_factor:.3f}")
        logger.info(f"  Output: {width}x{height}")
        
        video_info = f"{width}x{height}, {total_frames}f, {duration_seconds:.1f}s, scale: {scale_factor:.2f}"
        log_memory_usage("PROCESSING_START", f"Optimized processing: {video_info}")

        # Use optimized streaming processing with per-frame memory monitoring
        frame_count, processed_frames, processing_time, memory_stats = process_video_streaming_optimized(
            cap, out, detector, analyzer, total_frames,
            processing_width, processing_height, 
            original_width, original_height, scale_factor, rotation
        )
        
        processing_stats = f"{frame_count} frames, {processed_frames} analyzed, {processing_time:.1f}s total"
        memory_summary = f"Peak: {memory_stats.get('summary', {}).get('peak_memory_mb', 0):.1f}MB, Avg FPS: {memory_stats.get('summary', {}).get('average_fps', 0):.1f}"
        print(f"=== OPTIMIZED PROCESSING COMPLETE: {processing_stats} ===")
        print(f"=== MEMORY STATISTICS: {memory_summary} ===")
        log_memory_usage("PROCESSING_COMPLETE", f"{processing_stats}, {memory_summary}")

        # === STAGE 9: ANALYSIS SUMMARY ===
        print("=== STAGE 9: GENERATING ANALYSIS SUMMARY ===")
        log_memory_usage("BEFORE_ANALYSIS_SUMMARY")
        
        summary = analyzer.get_summary()  # Get aggregated results
        print("Analysis summary:", summary)
        
        
        log_memory_usage("AFTER_ANALYSIS_SUMMARY")

        # === STAGE 10: RESOURCE CLEANUP ===
        print("=== STAGE 10: RELEASING VIDEO RESOURCES ===")
        log_memory_usage("BEFORE_RESOURCE_CLEANUP")
        
        # Properly release resources before file operations
        if cap:
            cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        # Add small delay to ensure file handles are released
        import time
        time.sleep(0.1)
        
        # Check output file size
        if os.path.exists(annotated_video_path):
            output_size_mb = os.path.getsize(annotated_video_path) / (1024 * 1024)
            print(f"Annotated video created: {output_size_mb:.1f}MB")

        log_memory_usage("AFTER_RESOURCE_CLEANUP", f"Output: {output_size_mb:.1f}MB")

        # === STAGE 11: VIDEO POST-PROCESSING ===
        print("=== STAGE 11: ADDING FASTSTART TO VIDEO ===")
        log_memory_usage("BEFORE_VIDEO_POSTPROCESSING")
        
        final_video_path = video_processor.add_faststart(annotated_video_path)
        
        if os.path.exists(final_video_path):
            final_size_mb = os.path.getsize(final_video_path) / (1024 * 1024)
            print(f"Final video created: {final_size_mb:.1f}MB")
        
        log_memory_usage("AFTER_VIDEO_POSTPROCESSING", f"Final: {final_size_mb:.1f}MB")

        # === STAGE 12: UPLOADING TO STORAGE ===
        print("=== STAGE 12: UPLOADING PROCESSED VIDEO ===")
        log_memory_usage("BEFORE_VIDEO_UPLOAD")
        
        download_url, _ = storage_manager.upload_video(user_id, final_video_path, f"processed_{original_filename}", video_uuid)

        if not download_url:
            raise HTTPException(
                status_code=500,
                detail="Failed to upload video to supabase storage"
            )
        
        log_memory_usage("AFTER_VIDEO_UPLOAD")

        # === STAGE 13: GENERATING AI FEEDBACK ===
        print("=== STAGE 13: GENERATING AI FEEDBACK ===")
        log_memory_usage("BEFORE_FEEDBACK_GENERATION")
        
        feedback = await generate_feedback(summary, user_id, drill_manager)
        
        log_memory_usage("AFTER_FEEDBACK_GENERATION")

        # === STAGE 14: DATABASE OPERATIONS ===
        print("=== STAGE 14: SAVING TO DATABASE ===")
        log_memory_usage("BEFORE_DATABASE_OPERATIONS")
        
        analysis_result = database_manager.create_analysis(user_id, download_url, thumbnail_url, summary, feedback)
        
        log_memory_usage("AFTER_DATABASE_OPERATIONS")

        # === FINAL RESPONSE ===
        print("=== PROCESSING COMPLETE ===")
        final_memory_summary = memory_tracker.stop_tracking()
        
        response_data = {
            "user_id": user_id,
            "success": True,
            "message": "Video processing successful",
            "download_url": download_url,
            "thumbnail_url": thumbnail_url,
            "analysis_summary": summary,
            "processing_stats": {
                "total_frames": frame_count,
                "processed_frames": processed_frames,
                "processing_time_seconds": processing_time,
                "processing_fps": round(frame_count / processing_time, 2) if processing_time > 0 else 0,
                "input_file_size_mb": round(file_size_mb, 2),
                "output_file_size_mb": round(final_size_mb, 2) if 'final_size_mb' in locals() else None,
                "video_duration_seconds": round(duration_seconds, 2),
                "video_resolution": f"{width}x{height}"
            },
            "memory_stats": final_memory_summary
        }
        response_data["database_records"] = analysis_result

        # Print memory summary for easy viewing
        print("\n=== MEMORY USAGE SUMMARY ===")
        print(f"Peak Memory: {final_memory_summary.get('peak_memory_mb', 0)}MB")
        print(f"Memory Increase: {final_memory_summary.get('total_increase_mb', 0)}MB")
        print(f"Memory Intensive Stages: {final_memory_summary.get('memory_intensive_stages', [])}")
        print("===========================\n")

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Full error details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # === ENHANCED RESOURCE CLEANUP ===
        print("=== STARTING ENHANCED CLEANUP ===")
        log_memory_usage("CLEANUP_START")
        
        # Immediate resource release with error handling
        try:
            if 'cap' in locals() and cap:
                cap.release()
                cap = None
            if 'out' in locals() and out:
                out.release()
                out = None
            cv2.destroyAllWindows()
            
            # Force immediate cleanup of large objects
            immediate_resource_cleanup(cap, out)
            
        except Exception as e:
            logger.error(f"Error releasing video resources: {e}")
        
        # Memory cleanup before file operations
        gc.collect()
        
        # Add delay before cleanup (important on Windows)
        time.sleep(0.3)
        
        # Build comprehensive cleanup list
        cleanup_files = []
        
        # Add files that should always be cleaned up
        if 'video_path' in locals() and video_path:
            cleanup_files.append(video_path)
        if 'thumbnail_path' in locals() and thumbnail_path:
            cleanup_files.append(thumbnail_path)
        if 'annotated_video_path' in locals() and annotated_video_path:
            cleanup_files.append(annotated_video_path)
        
        # Add conditional files
        if 'final_video_path' in locals() and final_video_path and final_video_path != annotated_video_path:
            cleanup_files.append(final_video_path)
        
        # Add corrected video if it was created and is different from original
        try:
            if 'rotation' in locals() and 'corrected_video_path' in locals():
                if rotation != 0 and corrected_video_path != video_path and corrected_video_path not in cleanup_files:
                    cleanup_files.append(corrected_video_path)
        except:
            pass
        
        # Enhanced cleanup with retry logic
        cleanup_temp_enhanced(*cleanup_files)
        
        # Memory cleanup after file operations
        gc.collect()
        log_memory_usage("CLEANUP_FILES_COMPLETE")
        
        # Clean up user directory if too many leftover files
        try:
            if 'user_id' in locals() and user_id:
                user_tmp_dir = f"tmp/{user_id}"
                if os.path.exists(user_tmp_dir):
                    files_in_dir = []
                    for root, dirs, files in os.walk(user_tmp_dir):
                        files_in_dir.extend([os.path.join(root, f) for f in files])
                    
                    if len(files_in_dir) > 5:  # Too many leftover files
                        logger.warning(f"Many leftover files detected for user {user_id}, cleaning up folder")
                        cleanup_user_tmp_folder(user_id)
                        gc.collect()
        except Exception as e:
            logger.error(f"Error in user directory cleanup: {e}")
        
        # Final memory tracking and cleanup
        try:
            if hasattr(memory_tracker, 'memory_history') and memory_tracker.memory_history:
                if len(memory_tracker.memory_history) > 0:  # Check if tracking is still active
                    final_summary = memory_tracker.stop_tracking()
                    logger.info(f"FINAL MEMORY SUMMARY: {final_summary}")
                    
                    # Log memory efficiency
                    peak_memory = final_summary.get('peak_memory_mb', 0)
                    total_increase = final_summary.get('total_increase_mb', 0)
                    print(f"=== MEMORY EFFICIENCY REPORT ===")
                    print(f"Peak Memory: {peak_memory}MB")
                    print(f"Total Increase: {total_increase}MB")
                    print(f"Memory Efficient: {'YES' if total_increase < 500 else 'NO'}")
                    print("================================")
                    
        except Exception as e:
            logger.error(f"Error in final memory tracking: {e}")
        
        # Final garbage collection
        gc.collect()
        print("=== ENHANCED CLEANUP COMPLETE ===")

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

@app.get("/memory-history/")
async def memory_history(limit: int = 50):
    """Get recent memory usage history from the global tracker"""
    try:
        if not hasattr(memory_tracker, 'memory_history') or not memory_tracker.memory_history:
            return {
                "message": "No memory tracking data available",
                "history": [],
                "summary": {}
            }
        
        # Get the last N entries
        recent_history = memory_tracker.memory_history[-limit:] if limit > 0 else memory_tracker.memory_history
        
        # Calculate trends
        if len(recent_history) >= 2:
            first_entry = recent_history[0]
            last_entry = recent_history[-1]
            
            memory_trend = last_entry["process_memory_mb"] - first_entry["process_memory_mb"]
            time_span = (datetime.fromisoformat(last_entry["timestamp"].replace('Z', '+00:00')) - 
                        datetime.fromisoformat(first_entry["timestamp"].replace('Z', '+00:00'))).total_seconds()
            
            # Find peak memory
            peak_memory = max(entry["process_memory_mb"] for entry in recent_history)
            peak_stage = next((entry["stage"] for entry in recent_history 
                             if entry["process_memory_mb"] == peak_memory), "unknown")
            
            trend_analysis = {
                "memory_change_mb": round(memory_trend, 1),
                "time_span_seconds": round(time_span, 1),
                "memory_rate_mb_per_sec": round(memory_trend / time_span, 3) if time_span > 0 else 0,
                "peak_memory_mb": round(peak_memory, 1),
                "peak_stage": peak_stage
            }
        else:
            trend_analysis = {}
        
        return {
            "total_entries": len(memory_tracker.memory_history),
            "returned_entries": len(recent_history),
            "history": recent_history,
            "trend_analysis": trend_analysis,
            "summary": memory_tracker.get_summary()
        }
        
    except Exception as e:
        logger.error(f"Error getting memory history: {e}")
        raise HTTPException(status_code=500, detail=f"Memory history error: {str(e)}")

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

# Add this endpoint to monitor memory in real-time:

@app.get("/memory-status/")
async def memory_status():
    """Get current memory usage statistics with detailed breakdown"""
    try:
        # Process memory
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # System memory
        system_memory = psutil.virtual_memory()
        
        # Disk usage
        disk_usage = psutil.disk_usage('.')
        
        # Get current tmp directory size and file count
        tmp_size = 0
        tmp_files = 0
        tmp_breakdown = {}
        
        if os.path.exists("tmp"):
            for root, dirs, files in os.walk("tmp"):
                folder_size = 0
                for file in files:
                    try:
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path)
                        tmp_size += file_size
                        folder_size += file_size
                        tmp_files += 1
                    except:
                        pass
                
                if folder_size > 0:
                    rel_path = os.path.relpath(root, "tmp")
                    tmp_breakdown[rel_path] = {
                        "size_mb": round(folder_size / (1024**2), 2),
                        "file_count": len(files)
                    }
        
        # Get memory tracker summary if available
        tracker_summary = {}
        if hasattr(memory_tracker, 'memory_history') and memory_tracker.memory_history:
            tracker_summary = memory_tracker.get_summary()
        
        # Calculate memory efficiency metrics
        memory_efficiency = {
            "memory_per_core": round(memory_info.rss / (1024**2) / psutil.cpu_count(), 2),
            "memory_growth_rate": "N/A",
            "memory_pressure": "LOW" if system_memory.percent < 70 else "MEDIUM" if system_memory.percent < 85 else "HIGH"
        }
        
        if tracker_summary and "start_memory_mb" in tracker_summary:
            current_mb = memory_info.rss / (1024**2)
            growth = current_mb - tracker_summary["start_memory_mb"]
            memory_efficiency["memory_growth_rate"] = f"+{growth:.1f}MB"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "process_memory": {
                "rss_mb": round(memory_info.rss / (1024**2), 2),
                "vms_mb": round(memory_info.vms / (1024**2), 2),
                "percent": round(process.memory_percent(), 2),
                "shared_mb": round(getattr(memory_info, 'shared', 0) / (1024**2), 2) if hasattr(memory_info, 'shared') else 0
            },
            "system_memory": {
                "total_gb": round(system_memory.total / (1024**3), 2),
                "available_gb": round(system_memory.available / (1024**3), 2),
                "used_percent": system_memory.percent,
                "free_gb": round(system_memory.free / (1024**3), 2),
                "cached_gb": round(getattr(system_memory, 'cached', 0) / (1024**3), 2) if hasattr(system_memory, 'cached') else 0,
                "buffers_gb": round(getattr(system_memory, 'buffers', 0) / (1024**3), 2) if hasattr(system_memory, 'buffers') else 0
            },
            "disk_usage": {
                "total_gb": round(disk_usage.total / (1024**3), 2),
                "free_gb": round(disk_usage.free / (1024**3), 2),
                "used_percent": round((disk_usage.used / disk_usage.total) * 100, 2)
            },
            "tmp_directory": {
                "total_size_mb": round(tmp_size / (1024**2), 2),
                "file_count": tmp_files,
                "folder_breakdown": tmp_breakdown
            },
            "memory_efficiency": memory_efficiency,
            "tracker_summary": tracker_summary,
            "recommendations": generate_memory_recommendations(system_memory.percent, memory_info.rss / (1024**2))
        }
    except Exception as e:
        logger.error(f"Error getting memory status: {e}")
        raise HTTPException(status_code=500, detail=f"Memory status error: {str(e)}")

def generate_memory_recommendations(system_percent: float, process_mb: float) -> list:
    """Generate memory optimization recommendations"""
    recommendations = []
    
    if system_percent > 85:
        recommendations.append("CRITICAL: System memory usage high - consider reducing video processing quality")
    elif system_percent > 70:
        recommendations.append("WARNING: System memory usage elevated - monitor closely")
    
    if process_mb > 2000:  # 2GB
        recommendations.append("Process using high memory - consider chunked processing for large videos")
    
    if process_mb > 1000:  # 1GB
        recommendations.append("Consider reducing MediaPipe model complexity or frame processing frequency")
    
    if not recommendations:
        recommendations.append("Memory usage is within normal ranges")
    
    return recommendations


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

@app.get("/memory-status/")
async def memory_status():
    """Get current memory status and system information"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            "process_memory_mb": round(memory_info.rss / (1024**2), 1),
            "process_memory_percent": round(process.memory_percent(), 2),
            "system_memory_used_percent": round(system_memory.percent, 1),
            "system_memory_available_gb": round(system_memory.available / (1024**3), 2),
            "system_memory_total_gb": round(system_memory.total / (1024**3), 2),
            "memory_tracker_active": hasattr(memory_tracker, 'memory_history') and len(memory_tracker.memory_history) > 0,
            "recent_peak_memory_mb": memory_tracker.peak_memory if hasattr(memory_tracker, 'peak_memory') else 0
        }
    except Exception as e:
        return {"error": f"Failed to get memory status: {str(e)}"}

@app.post("/optimize-memory/")
async def optimize_memory():
    """Force garbage collection and memory optimization"""
    try:
        import gc
        
        # Force garbage collection
        collected = gc.collect()
        
        # Get memory before and after
        process = psutil.Process()
        memory_after = process.memory_info().rss / (1024**2)
        
        # Clean up old tmp files
        cleanup_old_tmp_files(max_age_hours=1)
        
        return {
            "success": True,
            "garbage_collected": collected,
            "memory_after_mb": round(memory_after, 1),
            "message": "Memory optimization completed"
        }
    except Exception as e:
        return {"error": f"Memory optimization failed: {str(e)}"}

@app.get("/processing-config/")
async def get_processing_config():
    """Get current processing configuration and recommendations"""
    try:
        import psutil
        system_memory = psutil.virtual_memory()
        available_gb = system_memory.available / (1024**3)
        
        # Recommend settings based on available memory
        if available_gb > 8:
            max_memory_mb = 600
            batch_size = 20
            quality_scale = 1.0
        elif available_gb > 4:
            max_memory_mb = 400
            batch_size = 15
            quality_scale = 0.8
        else:
            max_memory_mb = 200
            batch_size = 10
            quality_scale = 0.6
        
        return {
            "system_memory": {
                "total_gb": round(system_memory.total / (1024**3), 1),
                "available_gb": round(available_gb, 1),
                "used_percent": round(system_memory.percent, 1)
            },
            "recommended_config": {
                "max_processing_memory_mb": max_memory_mb,
                "batch_size": batch_size,
                "quality_scale_factor": quality_scale,
                "stream_processing": True,
                "immediate_cleanup": True
            },
            "optimizations_enabled": [
                "Dynamic resolution scaling",
                "Batch frame processing", 
                "Immediate resource cleanup",
                "Stream processing (no temp files)",
                "Per-frame rotation application",
                "Aggressive garbage collection"
            ]
        }
    except Exception as e:
        return {"error": f"Failed to get config: {str(e)}"}

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