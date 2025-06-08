from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from supabase import create_client, Client

import cv2
import mediapipe as mp
import numpy as np
import os
import uuid
from datetime import datetime
import asyncio
from typing import Optional

import PoseModule as pm
import RFAnalyzer as rfa

app = FastAPI()

origins = [
    "http://localhost:3000",
]

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()
detector = pm.PoseDetector()
analyzer = rfa.RFAnalyzer()

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL", 'https://zcshkomjuqcfeepimxwq.supabase.co')
SUPABASE_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY", 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inpjc2hrb21qdXFjZmVlcGlteHdxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc1NDg1MzEsImV4cCI6MjA2MzEyNDUzMX0.RP3pUwR2u9DGWN9h7g0pNt1S5emlDssTngacgQwQ1Go')
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", 'videos')  

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_to_supabase(file_path: str, file_name: str) -> Optional[str]:
    try:
        folder = "annotated-footage"
        unique_filename = f"{folder}/{uuid.uuid4()}_{file_name}" # unique file name
        with open(file_path, 'rb') as file: # read file content
            file_content = file.read()

        result = supabase.storage.from_(SUPABASE_BUCKET).upload(
            path=unique_filename,
            file=file_content,
            file_options={"content-type": "video/mp4"}
        )
        print(result)
        if result.path:
            # Get public URL
            public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(unique_filename)
            return public_url
        else:
            print(f"Upload failed: {result.error}")
            return None
        
    except Exception as e:
        print(f"Error uploading to Supabase: {result.error}")
        return None
    
def cleanup_temp(*file_paths):
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleteing {file_path}: {str({e})}")

# make an endpoint
@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    # create dir
    os.makedirs("tmp/processed", exist_ok=True) 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_filename = f"{timestamp}_{file.filename}"

    # save video to temporary loc
    video_path = f"tmp/{file.filename}"
    annotated_video_path = f"tmp/processed/processed_{file.filename}"
    try:
        with open(video_path, "wb") as buffer:
            buffer.write(await file.read())

        # process with MP pose
        cap = cv2.VideoCapture(video_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # -- ready output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # codec for output video
        out = cv2.VideoWriter(annotated_video_path, fourcc, 30.0, (width,height))

        prev_lmList = None
        measure_list = None

        if not cap.isOpened():
            print("Error: cannot open video file")

        while cap.isOpened():
            ret, frame = cap.read() # process the frame
            if not ret: # end video if error
                print("video ended or frame read failed")
                break

            frame = detector.findPose(frame)
            lmList = detector.findPosition(frame)
            # print(lmList)

            # if prev_lmList is not None and lmList:
            #     detector.findRunPhase(frame, lmList, prev_lmList)

            prev_lmList = lmList

            if lmList:
                # extract angles per frame
                head_position = detector.findHeadPosition(frame, draw=True)
                back_position = detector.findTorsoLean(frame, draw=True)
                arm_flexion = detector.findAngle(frame, 12, 14, 16, draw=True)
                left_knee = detector.findKneeAngle(frame, 23, 25, 27, draw=True)
                right_knee = detector.findKneeAngle(frame, 24, 26, 28, draw=True)
                foot_strike = detector.findFootAngle(frame, draw=True)
                contact_results = detector.detectFootContact_Alternative(frame, draw=True)

            if contact_results['right_landing']:
                frame_angles = [head_position, back_position, arm_flexion, 
                            left_knee, right_knee, foot_strike]
                analyzer.analyze_frame(frame_angles)  # Score this frame

            # After processing all frames:
            out.write(frame) # write annotated frame to the output video

            # press q to quit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
        summary = analyzer.get_summary()  # Get aggregated results
        print(summary)

        # clean up
        cap.release()
        out.release()

        download_url = upload_to_supabase(annotated_video_path, f"processed_{original_filename}")

        if not download_url:
            raise HTTPException(
                status_code=500,
                detail="Failed to upload video to supabase storage"
            )

        response_data = {
            "success": True,
            "message": "Video processing successful",
            "download_url": download_url,
            "analysis_summary": summary
        }

        print(response_data)

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Clean up temporary files
        cleanup_temp(video_path, annotated_video_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)