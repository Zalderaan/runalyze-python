"""
Usage Example: How to integrate per-frame memory monitoring 
into your existing video processing pipeline.
"""

import cv2
import time
from per_frame_memory_monitor import start_frame_memory_monitoring, log_frame_memory, stop_frame_memory_monitoring
import PoseModule as pm

def process_video_with_memory_monitoring(input_path, output_path):
    """
    Example of how to add comprehensive memory monitoring to your video processing.
    
    This shows the minimal integration needed to add per-frame memory tracking
    to any existing video processing pipeline.
    """
    
    # === STEP 1: Start Memory Monitoring ===
    # Initialize memory monitoring before any video processing
    monitor = start_frame_memory_monitoring(
        log_every_n_frames=30,          # Log basic stats every 30 frames
        detailed_log_every_n_frames=150, # Detailed log every 150 frames  
        memory_alert_threshold_mb=2000,  # Alert if memory exceeds 2GB
        enable_tracemalloc=True          # Enable detailed Python memory tracking
    )
    
    # === STEP 2: Initialize Your Components ===
    detector = pm.PoseDetector()
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # === STEP 3: Add Memory Logging at Key Points ===
            
            # Log memory after frame loading
            log_frame_memory(frame_count, total_frames, "frame_loaded")
            
            # Your existing processing code
            processed_frame = detector.findPose(frame, draw=True)
            lmList = detector.findPosition(processed_frame)
            
            # Log memory after pose detection
            log_frame_memory(
                frame_count, 
                total_frames, 
                "pose_detection",
                {"landmarks_detected": len(lmList) > 0}  # Optional: add context
            )
            
            # Your analysis code here
            if len(lmList) > 0:
                # Do your pose analysis
                pass
            
            # Log memory after analysis
            log_frame_memory(frame_count, total_frames, "analysis_complete")
            
            # Write frame
            out.write(processed_frame)
            
            # Log memory after writing
            log_frame_memory(frame_count, total_frames, "frame_written")
            
            # Clean up frame references (important for memory management)
            del frame
            del processed_frame
            
    finally:
        # === STEP 4: Clean Up and Get Results ===
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Stop monitoring and get comprehensive statistics
        memory_stats = stop_frame_memory_monitoring()
        
        # The monitor automatically exports detailed analysis to JSON
        print(f"Processing complete!")
        print(f"Peak memory usage: {memory_stats.get('summary', {}).get('peak_memory_mb', 0):.1f}MB")
        print(f"Total memory increase: {memory_stats.get('summary', {}).get('total_memory_increase_mb', 0):.1f}MB")
        print(f"Average FPS: {memory_stats.get('summary', {}).get('average_fps', 0):.1f}")
        
        return memory_stats

def quick_memory_check_example():
    """
    Example of quick memory monitoring for debugging specific parts of your code.
    """
    
    # Start monitoring for a specific section
    monitor = start_frame_memory_monitoring(log_every_n_frames=1)  # Log every frame
    
    # Simulate processing 10 frames
    for i in range(1, 11):
        # Your code here
        dummy_data = [j for j in range(i * 1000)]
        
        # Quick memory check
        log_frame_memory(i, 10, "data_processing", {"data_size": len(dummy_data)})
        
        del dummy_data
    
    # Get results
    stats = stop_frame_memory_monitoring()
    print(f"Quick check: Peak memory {stats.get('summary', {}).get('peak_memory_mb', 0):.1f}MB")

if __name__ == "__main__":
    print("Memory Monitoring Integration Examples")
    print("="*50)
    
    # Run quick example
    print("1. Quick memory check example:")
    quick_memory_check_example()
    
    print("\n2. For full video processing example:")
    print("   Uncomment the lines below and provide video paths")
    
    # Uncomment these lines to test with actual video:
    # input_video = "your_input_video.mp4"
    # output_video = "output_with_monitoring.mp4"
    # process_video_with_memory_monitoring(input_video, output_video)
