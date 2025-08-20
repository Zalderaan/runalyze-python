# Per-Frame Memory Monitoring for Video Processing

This module provides comprehensive memory monitoring capabilities for video processing and annotation operations, with detailed per-frame tracking and analysis.

## Features

### 🎯 Per-Frame Memory Tracking
- **Real-time monitoring**: Track memory usage for every frame processed
- **Stage-specific tracking**: Monitor memory at different processing stages (loading, pose detection, analysis, writing)
- **Performance metrics**: Track FPS, processing time, and efficiency
- **Automatic alerts**: Get warned when memory usage exceeds thresholds

### 📊 Comprehensive Statistics
- **Peak memory usage**: Track maximum memory consumption
- **Memory trends**: Analyze memory patterns throughout processing
- **Garbage collection events**: Monitor automatic cleanup operations
- **Processing efficiency**: Measure frames per second and resource utilization

### 🔍 Detailed Analysis
- **Frame-by-frame data**: Export detailed memory usage for each frame
- **JSON export**: Save comprehensive analysis data for further review
- **TraceMalloc integration**: Track Python-specific memory allocations
- **System memory monitoring**: Monitor overall system memory usage

## Quick Start

### Basic Integration

```python
from per_frame_memory_monitor import start_frame_memory_monitoring, log_frame_memory, stop_frame_memory_monitoring

# Start monitoring
monitor = start_frame_memory_monitoring(
    log_every_n_frames=50,          # Log every 50 frames
    memory_alert_threshold_mb=2000   # Alert at 2GB
)

# Process your video
for frame_num in range(total_frames):
    frame = get_next_frame()
    
    # Log memory at key points
    log_frame_memory(frame_num, total_frames, "frame_loaded")
    
    # Your processing code
    processed_frame = process_frame(frame)
    log_frame_memory(frame_num, total_frames, "processing_complete")
    
    write_frame(processed_frame)
    log_frame_memory(frame_num, total_frames, "frame_written")

# Get final statistics
stats = stop_frame_memory_monitoring()
print(f"Peak memory: {stats['summary']['peak_memory_mb']:.1f}MB")
```

### Advanced Configuration

```python
monitor = start_frame_memory_monitoring(
    log_every_n_frames=25,                # Basic logging frequency
    detailed_log_every_n_frames=100,      # Detailed logging frequency
    memory_alert_threshold_mb=1500,       # Memory alert threshold
    gc_threshold_mb=300,                  # Auto garbage collection threshold
    enable_tracemalloc=True               # Enable Python memory tracking
)
```

## Monitoring Levels

### 1. Basic Monitoring
Logs essential memory statistics at regular intervals:
- Current memory usage
- Memory increase from baseline
- Current FPS
- Progress percentage

### 2. Detailed Monitoring  
Provides comprehensive analysis including:
- System memory usage
- TraceMalloc statistics
- Garbage collection events
- Performance trends

### 3. Per-Frame Tracking
Captures memory data for every single frame:
- Stage-specific memory usage
- Frame processing time
- Memory cleanup effectiveness
- Error tracking

## Usage Examples

### Video Processing with Pose Detection

```python
import cv2
from per_frame_memory_monitor import start_frame_memory_monitoring, log_frame_memory, stop_frame_memory_monitoring
import PoseModule as pm

def process_video_with_monitoring(input_path, output_path):
    # Start monitoring
    monitor = start_frame_memory_monitoring()
    
    # Initialize components
    detector = pm.PoseDetector()
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Monitor frame loading
            log_frame_memory(frame_count, total_frames, "frame_loaded")
            
            # Pose detection
            processed_frame = detector.findPose(frame, draw=True)
            lmList = detector.findPosition(processed_frame)
            
            # Monitor pose detection
            log_frame_memory(
                frame_count, 
                total_frames, 
                "pose_detection",
                {"landmarks_detected": len(lmList) > 0, "landmark_count": len(lmList)}
            )
            
            # Analysis
            if len(lmList) > 0:
                # Your analysis code here
                pass
            
            log_frame_memory(frame_count, total_frames, "analysis_complete")
            
            # Write frame
            out.write(processed_frame)
            log_frame_memory(frame_count, total_frames, "frame_written")
            
            # Clean up
            del frame
            del processed_frame
            
    finally:
        cap.release()
        out.release()
        
        # Get final statistics
        stats = stop_frame_memory_monitoring()
        
        print(f"Processing Summary:")
        print(f"  Peak Memory: {stats['summary']['peak_memory_mb']:.1f}MB")
        print(f"  Total Increase: {stats['summary']['total_memory_increase_mb']:.1f}MB")
        print(f"  Average FPS: {stats['summary']['average_fps']:.1f}")
        print(f"  Frames Processed: {stats['summary']['total_frames_processed']}")
        
        return stats
```

### Memory Analysis with Context

```python
# Log memory with additional context
log_frame_memory(
    frame_number=150, 
    total_frames=1000, 
    stage="pose_analysis",
    additional_info={
        "pose_detected": True,
        "landmarks_count": 33,
        "confidence_score": 0.95,
        "analysis_type": "running_form"
    }
)
```

## Memory Statistics Output

The monitoring system provides comprehensive statistics:

```json
{
    "summary": {
        "total_frames_processed": 1000,
        "start_memory_mb": 150.2,
        "peak_memory_mb": 487.8,
        "final_memory_mb": 165.4,
        "total_memory_increase_mb": 337.6,
        "average_memory_mb": 234.1,
        "average_fps": 24.3,
        "total_processing_time": 41.2,
        "garbage_collections": 12
    },
    "frame_count": 1000,
    "memory_efficient_frames": 856,
    "memory_intensive_frames": 23,
    "gc_events": [...],
    "tracemalloc": {
        "final_mb": 45.2,
        "peak_mb": 78.9
    }
}
```

## Testing

### Run Comprehensive Test
```powershell
python test_per_frame_memory.py your_video.mp4 output_annotated.mp4
```

### Run Simple Test
```powershell
python test_per_frame_memory.py
```

### Run Usage Examples
```powershell
python memory_monitoring_examples.py
```

## Integration with Existing Code

### Minimal Integration
Add just 3 lines to your existing video processing:

```python
# 1. Start monitoring
monitor = start_frame_memory_monitoring()

# Your existing video processing loop
for frame_num, frame in enumerate(video_frames):
    processed = your_processing_function(frame)
    
    # 2. Add memory logging (optional - you can log at any stage)
    log_frame_memory(frame_num, total_frames, "processing")
    
    write_output(processed)

# 3. Get final stats
stats = stop_frame_memory_monitoring()
```

### Detailed Integration
Add memory logging at specific stages for detailed analysis:

```python
monitor = start_frame_memory_monitoring()

for frame_num, frame in enumerate(video_frames):
    # Log at each processing stage
    log_frame_memory(frame_num, total_frames, "frame_loaded")
    
    pose_data = detect_pose(frame)
    log_frame_memory(frame_num, total_frames, "pose_detection")
    
    analysis = analyze_pose(pose_data)
    log_frame_memory(frame_num, total_frames, "pose_analysis")
    
    annotated = annotate_frame(frame, analysis)
    log_frame_memory(frame_num, total_frames, "annotation")
    
    write_frame(annotated)
    log_frame_memory(frame_num, total_frames, "frame_written")

stats = stop_frame_memory_monitoring()
```

## Memory Management Best Practices

### 1. Explicit Cleanup
```python
# Clean up frame references immediately after use
del frame
del processed_frame
```

### 2. Batch Processing
```python
# Process in batches to limit memory buildup
batch_size = 15
for i in range(0, total_frames, batch_size):
    # Process batch
    # Clean up batch
    gc.collect()  # Force garbage collection
```

### 3. Memory Alerts
```python
# Set appropriate thresholds for your system
monitor = start_frame_memory_monitoring(
    memory_alert_threshold_mb=2000,  # Alert at 2GB
    gc_threshold_mb=500              # Auto-cleanup at 500MB increase
)
```

## Output Files

The monitoring system generates several output files:

- **`memory_analysis_YYYYMMDD_HHMMSS.json`**: Detailed frame-by-frame data
- **Console logs**: Real-time memory statistics  
- **Performance summary**: Final processing statistics

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `log_every_n_frames` | 50 | Basic memory logging frequency |
| `detailed_log_every_n_frames` | 200 | Detailed logging frequency |
| `memory_alert_threshold_mb` | 2000 | Memory alert threshold (MB) |
| `gc_threshold_mb` | 500 | Auto garbage collection threshold (MB) |
| `enable_tracemalloc` | True | Enable Python memory tracking |

## Requirements

- `psutil` - For system memory monitoring
- `tracemalloc` - For Python memory tracking (built-in)
- `gc` - For garbage collection management (built-in)
- `json` - For data export (built-in)

## Troubleshooting

### High Memory Usage
1. Reduce batch size in processing
2. Increase garbage collection frequency  
3. Use explicit `del` statements for large objects
4. Monitor the detailed logs to identify memory-intensive stages

### Performance Impact
The monitoring system is designed to have minimal impact:
- Lightweight memory queries
- Configurable logging frequency
- Optional TraceMalloc (can be disabled for production)

### Alert Threshold Tuning
Adjust thresholds based on your system:
- Development: Lower thresholds for early detection
- Production: Higher thresholds to avoid false alarms
- Testing: Very low thresholds to stress-test memory management

## Integration in Main Application

The memory monitoring is already integrated into the main video processing pipeline in `main.py`. The `process_video_streaming_optimized` function now includes:

- Automatic per-frame memory monitoring
- Stage-specific tracking for pose detection, analysis, and video writing
- Garbage collection optimization
- Comprehensive statistics export

This provides production-ready memory monitoring for the RunAnalyze application.
