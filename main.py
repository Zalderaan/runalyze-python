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
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "*",  # Remove in production for security
]

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
                    
                    # Check for rotation in tags
                    if 'tags' in stream:
                        if 'rotate' in stream['tags']:
                            return int(stream['tags']['rotate'])
                    
                    # Check for side_data_list (newer format)
                    if 'side_data_list' in stream:
                        for side_data in stream['side_data_list']:
                            if side_data.get('side_data_type') == 'Display Matrix':
                                rotation = side_data.get('rotation', 0)
                                return abs(int(rotation))
            
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
                cap.release()
                
                # For typical mobile landscape videos that appear vertical,
                # the height is usually greater than width due to incorrect orientation
                # This is a heuristic and might need adjustment based on your specific use case
                if height > width and height / width > 1.3:  # Strong vertical aspect ratio
                    logger.info(f"Video appears to be rotated landscape (dimensions: {width}x{height})")
                    return 90  # Assume it needs 90-degree rotation
                
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
            
            # Try using autorotate first (most reliable method)
            autorotate_cmd = (
                f'ffmpeg -i "{input_path}" -c:v libx264 -crf {video_config.ffmpeg_crf} '
                f'-preset {video_config.ffmpeg_preset} -movflags +faststart '
                f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                f'-metadata:s:v rotate=0 -auto-alt-ref 0 "{output_path}" -y'
            )
            
            # If autorotate is not available or fails, use manual transpose
            manual_cmd = None
            if rotation != 0:
                # Determine the appropriate transpose filter based on rotation
                transpose_filter = ""
                if rotation == 90:
                    transpose_filter = "transpose=1,"  # 90 degrees clockwise
                elif rotation == 180:
                    transpose_filter = "transpose=2,transpose=2,"  # 180 degrees
                elif rotation == 270:
                    transpose_filter = "transpose=2,"  # 90 degrees counter-clockwise
                
                # Build the video filter chain with manual rotation
                video_filters = f"{transpose_filter}scale=trunc(iw/2)*2:trunc(ih/2)*2"
                
                manual_cmd = (
                    f'ffmpeg -i "{input_path}" -c:v libx264 -crf {video_config.ffmpeg_crf} '
                    f'-preset {video_config.ffmpeg_preset} -movflags +faststart '
                    f'-vf "{video_filters}" '
                    f'-metadata:s:v rotate=0 -metadata:s:v:0 rotate=0 -auto-alt-ref 0 "{output_path}" -y'
                )
            
            # Try the appropriate command
            cmd_to_use = manual_cmd if manual_cmd else autorotate_cmd
            logger.info(f"Processing video with orientation correction (rotation: {rotation}°): {input_path}")
            
            result = os.system(cmd_to_use)
            
            if result == 0:
                logger.info(f"Video processing completed with orientation fix: {output_path}")
                return output_path
            else:
                logger.error(f"FFmpeg processing failed with code: {result}")
                # If manual command failed and we have rotation, try the basic autorotate
                if manual_cmd and result != 0:
                    logger.info("Retrying with basic processing...")
                    result = os.system(autorotate_cmd)
                    if result == 0:
                        logger.info(f"Video processing completed with basic processing: {output_path}")
                        return output_path
                
                return input_path
                
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return input_path

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
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):  # ✓ Check if file_path is not None
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {str({e})}")

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
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # -- ready output
        fourcc = cv2.VideoWriter_fourcc(*'avc1') # codec for output video
        out = cv2.VideoWriter(annotated_video_path, fourcc, 30.0, (width,height), isColor=True)

        prev_lmList = None
        measure_list = None

        if not cap.isOpened():
            raise Exception("Error: cannot open video file")

        while cap.isOpened():
            ret, frame = cap.read() # process the frame
            if not ret: # end video if error
                print("video ended or frame read failed")
                break

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
            if cap:
                cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
        except:
            pass
        
        # Add delay before cleanup
        import time
        time.sleep(0.2)
        
        # Clean up temporary files
        cleanup_temp(video_path, final_video_path, annotated_video_path, thumbnail_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)