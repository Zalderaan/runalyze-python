"""
Comprehensive test script for per-frame memory monitoring during video processing.
This script demonstrates the enhanced memory monitoring system with real video analysis.
"""

import cv2
import os
import sys
import time
import gc
from datetime import datetime

# Add the current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from per_frame_memory_monitor import (
    PerFrameMemoryMonitor,
    start_frame_memory_monitoring,
    log_frame_memory,
    stop_frame_memory_monitoring
)
import PoseModule as pm

def test_per_frame_monitoring_with_video(video_path: str, output_path: str = None):
    """
    Test comprehensive per-frame memory monitoring with video processing.
    
    Args:
        video_path: Path to input video file
        output_path: Path for output video (optional)
    """
    
    print("🎬 Starting comprehensive per-frame memory monitoring test...")
    print("="*70)
    
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        return False
    
    # Initialize pose detector
    print("🔧 Initializing pose detector...")
    detector = pm.PoseDetector()
    
    # Open video
    print("📹 Opening video file...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video file: {video_path}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📊 Video Properties:")
    print(f"   • Resolution: {width}x{height}")
    print(f"   • Total frames: {total_frames}")
    print(f"   • FPS: {fps}")
    print(f"   • Duration: {duration:.1f}s")
    print(f"   • File size: {os.path.getsize(video_path) / (1024**2):.1f}MB")
    
    # Setup output video writer if requested
    out = None
    if output_path:
        print(f"📝 Setting up output video: {output_path}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Start comprehensive memory monitoring
    print("\n🔍 Starting enhanced per-frame memory monitoring...")
    monitor = start_frame_memory_monitoring(
        log_every_n_frames=15,  # Basic log every 15 frames
        detailed_log_every_n_frames=75,  # Detailed log every 75 frames
        memory_alert_threshold_mb=2000,  # Alert at 2GB
        enable_tracemalloc=True
    )
    
    frame_count = 0
    pose_detections = 0
    processing_errors = 0
    start_time = time.time()
    
    try:
        print("⚡ Starting frame-by-frame processing with memory tracking...")
        print("-" * 70)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            try:
                # === STAGE 1: Frame Loading ===
                frame_size_mb = frame.nbytes / (1024**2)
                log_frame_memory(
                    frame_count, 
                    total_frames, 
                    "frame_loaded",
                    {
                        "frame_size_mb": round(frame_size_mb, 3),
                        "frame_dimensions": f"{frame.shape[1]}x{frame.shape[0]}",
                        "frame_channels": frame.shape[2] if len(frame.shape) > 2 else 1
                    }
                )
                
                # === STAGE 2: Pose Detection ===
                processed_frame = detector.findPose(frame, draw=True)
                
                # Get pose landmarks
                lmList = detector.findPosition(processed_frame, draw=False)
                has_pose = len(lmList) > 0
                
                if has_pose:
                    pose_detections += 1
                
                log_frame_memory(
                    frame_count, 
                    total_frames, 
                    "pose_detection_complete",
                    {
                        "pose_detected": has_pose,
                        "landmarks_count": len(lmList),
                        "total_pose_detections": pose_detections,
                        "pose_detection_rate": round(pose_detections / frame_count * 100, 1)
                    }
                )
                
                # === STAGE 3: Frame Annotation ===
                # Add comprehensive frame information
                y_offset = 30
                line_height = 25
                
                # Frame info
                cv2.putText(processed_frame, f"Frame: {frame_count}/{total_frames}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += line_height
                
                # Pose info
                if has_pose:
                    cv2.putText(processed_frame, f"Pose: {len(lmList)} landmarks detected", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(processed_frame, "No pose detected", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y_offset += line_height
                
                # Processing stats
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                cv2.putText(processed_frame, f"FPS: {current_fps:.1f}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                log_frame_memory(
                    frame_count, 
                    total_frames, 
                    "frame_annotation_complete",
                    {
                        "annotations_added": True,
                        "current_fps": round(current_fps, 2)
                    }
                )
                
                # === STAGE 4: Video Writing ===
                if out:
                    out.write(processed_frame)
                    
                    log_frame_memory(
                        frame_count, 
                        total_frames, 
                        "frame_written",
                        {
                            "output_writing": True,
                            "estimated_output_size_mb": frame_count * frame_size_mb
                        }
                    )
                
                # === STAGE 5: Memory Management ===
                # Explicit cleanup every frame
                del frame
                del processed_frame
                
                # Periodic garbage collection
                if frame_count % 50 == 0:
                    collected_objects = gc.collect()
                    log_frame_memory(
                        frame_count, 
                        total_frames, 
                        "garbage_collection",
                        {
                            "gc_triggered": True,
                            "objects_collected": collected_objects,
                            "gc_cycle": frame_count // 50
                        }
                    )
                
                # === STAGE 6: Progress Reporting ===
                if frame_count % 100 == 0:
                    progress_percent = (frame_count / total_frames) * 100
                    eta_seconds = (total_frames - frame_count) / current_fps if current_fps > 0 else 0
                    
                    print(f"📈 Progress: {frame_count}/{total_frames} ({progress_percent:.1f}%) - "
                          f"FPS: {current_fps:.1f} - ETA: {eta_seconds:.1f}s - "
                          f"Poses: {pose_detections}")
                
            except Exception as e:
                processing_errors += 1
                print(f"❌ Error processing frame {frame_count}: {e}")
                
                log_frame_memory(
                    frame_count, 
                    total_frames, 
                    "frame_error",
                    {
                        "error": str(e),
                        "total_errors": processing_errors
                    }
                )
                
                # Continue processing despite errors
                continue
            
            # Early exit for testing (optional)
            # Uncomment the next two lines to process only first 300 frames for testing
            # if frame_count >= 300:
            #     print("⏹️  Early exit for testing (300 frames processed)")
            #     break
    
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Critical error during processing: {e}")
    
    finally:
        print("\n🧹 Cleaning up resources...")
        
        # Clean up video resources
        if cap:
            cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Stop memory monitoring and get comprehensive statistics
        print("📊 Finalizing memory monitoring...")
        final_stats = stop_frame_memory_monitoring()
        
        # Calculate final processing statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        pose_detection_rate = (pose_detections / frame_count * 100) if frame_count > 0 else 0
        
        # Display comprehensive results
        print("\n" + "="*70)
        print("🎯 COMPREHENSIVE PROCESSING SUMMARY")
        print("="*70)
        
        print(f"📹 Video Processing:")
        print(f"   • Frames processed: {frame_count}/{total_frames}")
        print(f"   • Processing time: {total_time:.2f}s")
        print(f"   • Average FPS: {avg_fps:.2f}")
        print(f"   • Processing errors: {processing_errors}")
        
        print(f"\n🏃 Pose Detection:")
        print(f"   • Poses detected: {pose_detections}")
        print(f"   • Detection rate: {pose_detection_rate:.1f}%")
        
        if final_stats.get('summary'):
            summary = final_stats['summary']
            print(f"\n💾 Memory Usage:")
            print(f"   • Peak memory: {summary.get('peak_memory_mb', 0):.1f}MB")
            print(f"   • Total increase: {summary.get('total_memory_increase_mb', 0):.1f}MB")
            print(f"   • Average memory: {summary.get('average_memory_mb', 0):.1f}MB")
            print(f"   • Garbage collections: {summary.get('garbage_collections', 0)}")
            print(f"   • Memory efficient frames: {final_stats.get('memory_efficient_frames', 0)}")
            print(f"   • Memory intensive frames: {final_stats.get('memory_intensive_frames', 0)}")
        
        if output_path and os.path.exists(output_path):
            output_size_mb = os.path.getsize(output_path) / (1024**2)
            print(f"\n📁 Output:")
            print(f"   • Output file: {output_path}")
            print(f"   • Output size: {output_size_mb:.1f}MB")
        
        # Export detailed memory analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = f"detailed_memory_analysis_{timestamp}.json"
        monitor.export_frame_data(export_path)
        print(f"\n📋 Detailed Analysis:")
        print(f"   • Memory report: {export_path}")
        print(f"   • Total data points: {len(final_stats.get('frame_data', []))}")
        
        print("="*70)
        print("✅ Test completed successfully!")
        
        return True

def test_simple_frame_monitoring():
    """Simple test for basic functionality without video file."""
    print("🧪 Running simple per-frame memory monitoring test...")
    
    # Start monitoring
    monitor = start_frame_memory_monitoring(
        log_every_n_frames=5,
        detailed_log_every_n_frames=20,
        memory_alert_threshold_mb=1000,
        enable_tracemalloc=True
    )
    
    print("⚡ Simulating frame processing...")
    
    # Simulate 100 frames of processing
    for frame_num in range(1, 101):
        # Simulate frame loading
        dummy_frame = [[i * j for j in range(640)] for i in range(480)]  # Simulate 640x480 frame
        log_frame_memory(frame_num, 100, "frame_loaded", {"simulated": True})
        
        # Simulate processing
        processed_data = [row[::2] for row in dummy_frame[::2]]  # Simulate downsampling
        log_frame_memory(frame_num, 100, "processing", {"downsampled": True})
        
        # Simulate analysis
        if frame_num % 3 == 0:  # "Detect pose" in every 3rd frame
            landmarks = [(i, i*2) for i in range(33)]  # Simulate 33 landmarks
            log_frame_memory(frame_num, 100, "pose_detection", {"landmarks": len(landmarks)})
        
        # Cleanup
        del dummy_frame
        del processed_data
        
        if frame_num % 20 == 0:
            collected = gc.collect()
            log_frame_memory(frame_num, 100, "cleanup", {"gc_objects": collected})
        
        # Small delay
        time.sleep(0.02)
    
    # Stop monitoring
    stats = stop_frame_memory_monitoring()
    
    print(f"✅ Simple test completed.")
    print(f"   Peak memory: {stats.get('summary', {}).get('peak_memory_mb', 0):.1f}MB")
    print(f"   Total frames: {stats.get('summary', {}).get('total_frames_processed', 0)}")
    
    return stats

if __name__ == "__main__":
    print("🚀 Per-Frame Memory Monitor Comprehensive Test Suite")
    print("="*70)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        print(f"📹 Testing with video: {video_path}")
        if output_path:
            print(f"📝 Output will be saved to: {output_path}")
        
        success = test_per_frame_monitoring_with_video(video_path, output_path)
        
        if not success:
            print("❌ Video test failed, running simple test...")
            test_simple_frame_monitoring()
    else:
        print("📝 No video provided, running simple simulation test...")
        test_simple_frame_monitoring()
        
        # Look for video files in current directory
        video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv')
        video_files = [f for f in os.listdir('.') if f.lower().endswith(video_extensions)]
        
        if video_files:
            print(f"\n💡 Found video files in current directory:")
            for i, video_file in enumerate(video_files[:5], 1):  # Show first 5
                size_mb = os.path.getsize(video_file) / (1024**2)
                print(f"   {i}. {video_file} ({size_mb:.1f}MB)")
            
            print(f"\n💡 To test with a video file, run:")
            print(f"   python {sys.argv[0]} {video_files[0]}")
            print(f"   python {sys.argv[0]} {video_files[0]} output_annotated.mp4")
        else:
            print("\n💡 To test with a video file, run:")
            print(f"   python {sys.argv[0]} your_video.mp4")
            print(f"   python {sys.argv[0]} your_video.mp4 output_annotated.mp4")
    
    print("\n🎉 All tests completed!")
