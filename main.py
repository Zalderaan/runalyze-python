from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

import cv2
import mediapipe as mp
import numpy as np
import os
import requests

import PoseModule as pm
import RFAnalyzer as rfa

app = FastAPI()
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()
detector = pm.PoseDetector()
analyzer = rfa.RFAnalyzer()

# make an endpoint
@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    os.makedirs("tmp/processed", exist_ok=True) # create dir

    # save video to temporary loc
    video_path = f"tmp/{file.filename}"
    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())

    # process with MP pose
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # -- ready output
    annotated_video_path = f"tmp/processed/processed_{file.filename}"
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
            foot_strike = detector.findAngle(frame, 26, 28, 32, draw=True)
            contact_results = detector.detectFootContact_Alternative(frame, draw=True)

            angle_list = [head_position, back_position, arm_flexion, 
                          left_knee, right_knee, foot_strike]
            # print("Head position: ", head_position)
            # print("Back Lean: ", back_position)
            # print("Arm Flexion: ", arm_flexion)
            # print("Left Knee Angle: ", left_knee)
            # print("Right Knee Angle: ", right_knee)
            # print("Foot strike: ", foot_strike)
            # print(f"Contact-based - Left Landing: {contact_results['left_landing']}, Right Landing: {contact_results['right_landing']}")
            # print(f"Contact-based - Left Contact: {contact_results['left_contact']}, Right Contact: {contact_results['right_contact']}")

        # TODO: Process angle positions
        # * 1. Correctly spot foot landing
        # * 2. Measure angles relevant to foot landing
        # * 2.1 define ideal angles
        # ? Where to store the extracted angles?

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

    # return the annotated video as a response
    return FileResponse(annotated_video_path, media_type = 'video/mp4', filename=f"processed_{file.filename}")
    # return JSONResponse({
    #     "analysis_summary": summary
    # })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)