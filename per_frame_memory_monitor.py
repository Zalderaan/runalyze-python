"""
Enhanced Memory Monitor for Per-Frame Video Processing
Provides detailed memory tracking during video annotation and writing operations.
"""

import psutil
import tracemalloc
import gc
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import json
import os

logger = logging.getLogger(__name__)

class PerFrameMemoryMonitor:
    """
    Advanced memory monitoring system for per-frame video processing.
    Tracks memory usage, garbage collection, and performance metrics.
    """
    
    def __init__(self, 
                 log_every_n_frames: int = 50,
                 detailed_log_every_n_frames: int = 200,
                 memory_alert_threshold_mb: int = 2000,
                 gc_threshold_mb: int = 500):
        """
        Initialize per-frame memory monitor.
        
        Args:
            log_every_n_frames: Log basic memory stats every N frames
            detailed_log_every_n_frames: Log detailed stats every N frames
            memory_alert_threshold_mb: Alert when memory exceeds this threshold
            gc_threshold_mb: Force garbage collection when memory increase exceeds this
        """
        self.log_every_n_frames = log_every_n_frames
        self.detailed_log_every_n_frames = detailed_log_every_n_frames
        self.memory_alert_threshold_mb = memory_alert_threshold_mb
        self.gc_threshold_mb = gc_threshold_mb
        
        # Memory tracking variables
        self.process = psutil.Process()
        self.start_memory_mb = 0
        self.peak_memory_mb = 0
        self.baseline_memory_mb = 0
        
        # Frame tracking
        self.frame_memories: List[Dict] = []
        self.stage_memories: Dict[str, Dict] = {}
        self.gc_events: List[Dict] = []
        
        # Performance tracking
        self.start_time = None
        self.last_frame_time = None
        self.fps_samples = []
        
        # Tracemalloc tracking
        self.tracemalloc_enabled = False
        
        # Alert tracking
        self.last_alert_time = 0
        self.alert_cooldown = 30  # seconds between alerts
        
    def start_monitoring(self, enable_tracemalloc: bool = True):
        """Start comprehensive memory monitoring session."""
        try:
            # Start tracemalloc if requested
            if enable_tracemalloc:
                tracemalloc.start()
                self.tracemalloc_enabled = True
            
            # Get baseline memory
            memory_info = self.process.memory_info()
            self.start_memory_mb = memory_info.rss / (1024**2)
            self.baseline_memory_mb = self.start_memory_mb
            self.peak_memory_mb = self.start_memory_mb
            
            # Initialize timing
            self.start_time = time.time()
            self.last_frame_time = self.start_time
            
            logger.info(f"🎬 FRAME MEMORY MONITORING STARTED")
            logger.info(f"📊 Baseline Memory: {self.start_memory_mb:.1f}MB")
            logger.info(f"⚙️  TraceMalloc: {'Enabled' if self.tracemalloc_enabled else 'Disabled'}")
            logger.info(f"🔔 Alert Threshold: {self.memory_alert_threshold_mb}MB")
            
        except Exception as e:
            logger.error(f"Error starting memory monitoring: {e}")
    
    def log_frame_memory(self, 
                        frame_number: int, 
                        total_frames: int = None,
                        stage: str = "processing",
                        additional_info: Dict = None) -> Dict[str, Any]:
        """
        Log memory usage for a specific frame with enhanced tracking.
        
        Args:
            frame_number: Current frame number (1-indexed)
            total_frames: Total frames in video (for progress calculation)
            stage: Processing stage (e.g., "pose_detection", "analysis", "writing")
            additional_info: Additional metadata to include
            
        Returns:
            Dictionary with memory statistics
        """
        try:
            current_time = time.time()
            
            # Get memory information
            memory_info = self.process.memory_info()
            current_memory_mb = memory_info.rss / (1024**2)
            memory_percent = self.process.memory_percent()
            
            # Update peak memory
            if current_memory_mb > self.peak_memory_mb:
                self.peak_memory_mb = current_memory_mb
            
            # Calculate memory increase from baseline
            memory_increase_mb = current_memory_mb - self.baseline_memory_mb
            
            # Get system memory info
            system_memory = psutil.virtual_memory()
            
            # Get tracemalloc info if enabled
            tracemalloc_current_mb = 0
            tracemalloc_peak_mb = 0
            if self.tracemalloc_enabled:
                try:
                    current_traced, peak_traced = tracemalloc.get_traced_memory()
                    tracemalloc_current_mb = current_traced / (1024**2)
                    tracemalloc_peak_mb = peak_traced / (1024**2)
                except:
                    pass
            
            # Calculate FPS
            frame_time = current_time - self.last_frame_time
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            self.fps_samples.append(current_fps)
            if len(self.fps_samples) > 100:  # Keep last 100 samples
                self.fps_samples.pop(0)
            avg_fps = sum(self.fps_samples) / len(self.fps_samples)
            
            # Calculate progress
            progress_percent = (frame_number / total_frames * 100) if total_frames else 0
            
            # Estimate ETA
            elapsed_time = current_time - self.start_time
            eta_seconds = 0
            if frame_number > 0 and total_frames:
                frames_remaining = total_frames - frame_number
                avg_time_per_frame = elapsed_time / frame_number
                eta_seconds = frames_remaining * avg_time_per_frame
            
            # Create comprehensive memory data
            memory_data = {
                "timestamp": datetime.now().isoformat(),
                "frame_number": frame_number,
                "total_frames": total_frames,
                "stage": stage,
                "process_memory_mb": round(current_memory_mb, 2),
                "memory_increase_mb": round(memory_increase_mb, 2),
                "peak_memory_mb": round(self.peak_memory_mb, 2),
                "memory_percent": round(memory_percent, 2),
                "system_memory_percent": round(system_memory.percent, 1),
                "system_available_gb": round(system_memory.available / (1024**3), 2),
                "tracemalloc_current_mb": round(tracemalloc_current_mb, 2),
                "tracemalloc_peak_mb": round(tracemalloc_peak_mb, 2),
                "current_fps": round(current_fps, 1),
                "average_fps": round(avg_fps, 1),
                "progress_percent": round(progress_percent, 1),
                "eta_seconds": round(eta_seconds, 1),
                "elapsed_seconds": round(elapsed_time, 1)
            }
            
            # Add additional info if provided
            if additional_info:
                memory_data.update(additional_info)
            
            # Store frame memory data
            self.frame_memories.append(memory_data)
            
            # Logging based on frame interval
            should_log_basic = frame_number % self.log_every_n_frames == 0
            should_log_detailed = frame_number % self.detailed_log_every_n_frames == 0
            
            if should_log_detailed:
                self._log_detailed_frame_stats(memory_data)
            elif should_log_basic:
                self._log_basic_frame_stats(memory_data)
            
            # Check for memory alerts
            self._check_memory_alerts(memory_data)
            
            # Check if garbage collection is needed
            self._check_garbage_collection(memory_increase_mb)
            
            # Update timing
            self.last_frame_time = current_time
            
            return memory_data
            
        except Exception as e:
            logger.error(f"Error logging frame memory for frame {frame_number}: {e}")
            return {}
    
    def _log_basic_frame_stats(self, memory_data: Dict):
        """Log basic frame statistics."""
        logger.info(
            f"🎭 Frame {memory_data['frame_number']}/{memory_data.get('total_frames', '?')} "
            f"({memory_data['progress_percent']:.1f}%) - "
            f"Memory: {memory_data['process_memory_mb']:.1f}MB "
            f"(+{memory_data['memory_increase_mb']:+.1f}MB), "
            f"FPS: {memory_data['current_fps']:.1f}, "
            f"ETA: {memory_data['eta_seconds']:.0f}s"
        )
    
    def _log_detailed_frame_stats(self, memory_data: Dict):
        """Log detailed frame statistics."""
        logger.info(
            f"🎭📊 DETAILED Frame {memory_data['frame_number']}/{memory_data.get('total_frames', '?')} "
            f"({memory_data['progress_percent']:.1f}%) - "
            f"Process: {memory_data['process_memory_mb']:.1f}MB "
            f"(+{memory_data['memory_increase_mb']:+.1f}MB, "
            f"Peak: {memory_data['peak_memory_mb']:.1f}MB), "
            f"System: {memory_data['system_memory_percent']:.1f}% "
            f"({memory_data['system_available_gb']:.1f}GB free), "
            f"FPS: {memory_data['current_fps']:.1f} "
            f"(avg: {memory_data['average_fps']:.1f}), "
            f"TraceMalloc: {memory_data['tracemalloc_current_mb']:.1f}MB"
        )
    
    def _check_memory_alerts(self, memory_data: Dict):
        """Check for memory alerts and warnings."""
        current_time = time.time()
        current_memory = memory_data['process_memory_mb']
        
        # High memory usage alert
        if (current_memory > self.memory_alert_threshold_mb and 
            current_time - self.last_alert_time > self.alert_cooldown):
            
            logger.warning(
                f"🚨 HIGH MEMORY USAGE ALERT - Frame {memory_data['frame_number']}: "
                f"{current_memory:.1f}MB exceeds threshold ({self.memory_alert_threshold_mb}MB)"
            )
            self.last_alert_time = current_time
        
        # System memory warning
        if memory_data['system_memory_percent'] > 85:
            logger.warning(
                f"⚠️  SYSTEM MEMORY WARNING - Frame {memory_data['frame_number']}: "
                f"System memory usage at {memory_data['system_memory_percent']:.1f}%"
            )
    
    def _check_garbage_collection(self, memory_increase_mb: float):
        """Check if garbage collection should be triggered."""
        if memory_increase_mb > self.gc_threshold_mb:
            logger.info(f"🗑️  Triggering garbage collection (memory increase: {memory_increase_mb:.1f}MB)")
            collected = gc.collect()
            
            # Log GC event
            gc_event = {
                "timestamp": datetime.now().isoformat(),
                "objects_collected": collected,
                "memory_before_mb": self.process.memory_info().rss / (1024**2)
            }
            
            # Get memory after GC
            time.sleep(0.1)  # Brief pause for GC to complete
            memory_after_mb = self.process.memory_info().rss / (1024**2)
            gc_event["memory_after_mb"] = memory_after_mb
            gc_event["memory_freed_mb"] = gc_event["memory_before_mb"] - memory_after_mb
            
            self.gc_events.append(gc_event)
            
            logger.info(
                f"🗑️  GC Complete: {collected} objects collected, "
                f"{gc_event['memory_freed_mb']:.1f}MB freed"
            )
    
    def log_stage_memory(self, stage: str, additional_info: str = ""):
        """Log memory usage for a processing stage."""
        try:
            memory_info = self.process.memory_info()
            current_memory_mb = memory_info.rss / (1024**2)
            memory_increase = current_memory_mb - self.baseline_memory_mb
            
            stage_data = {
                "timestamp": datetime.now().isoformat(),
                "stage": stage,
                "memory_mb": round(current_memory_mb, 1),
                "memory_increase_mb": round(memory_increase, 1),
                "additional_info": additional_info
            }
            
            self.stage_memories[stage] = stage_data
            
            logger.info(
                f"🎯 STAGE [{stage}] - "
                f"Memory: {current_memory_mb:.1f}MB "
                f"(+{memory_increase:+.1f}MB)"
                f"{f' - {additional_info}' if additional_info else ''}"
            )
            
            return stage_data
            
        except Exception as e:
            logger.error(f"Error logging stage memory for {stage}: {e}")
            return {}
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics."""
        try:
            if not self.frame_memories:
                return {"error": "No frame memory data available"}
            
            # Calculate statistics
            memory_values = [f["process_memory_mb"] for f in self.frame_memories]
            memory_increases = [f["memory_increase_mb"] for f in self.frame_memories]
            fps_values = [f["current_fps"] for f in self.frame_memories if f["current_fps"] > 0]
            
            stats = {
                "summary": {
                    "total_frames_processed": len(self.frame_memories),
                    "start_memory_mb": round(self.start_memory_mb, 1),
                    "peak_memory_mb": round(self.peak_memory_mb, 1),
                    "final_memory_mb": round(memory_values[-1], 1),
                    "total_memory_increase_mb": round(self.peak_memory_mb - self.start_memory_mb, 1),
                    "average_memory_mb": round(sum(memory_values) / len(memory_values), 1),
                    "max_memory_increase_mb": round(max(memory_increases), 1),
                    "min_memory_increase_mb": round(min(memory_increases), 1),
                    "average_fps": round(sum(fps_values) / len(fps_values), 1) if fps_values else 0,
                    "total_processing_time": round(time.time() - self.start_time, 1),
                    "garbage_collections": len(self.gc_events)
                },
                "frame_count": len(self.frame_memories),
                "stage_count": len(self.stage_memories),
                "gc_events": self.gc_events,
                "memory_efficient_frames": len([f for f in self.frame_memories if f["memory_increase_mb"] < 10]),
                "memory_intensive_frames": len([f for f in self.frame_memories if f["memory_increase_mb"] > 50])
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating memory statistics: {e}")
            return {"error": str(e)}
    
    def export_frame_data(self, output_path: str = None) -> str:
        """Export detailed frame memory data to JSON file."""
        try:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"memory_analysis_{timestamp}.json"
            
            export_data = {
                "monitoring_session": {
                    "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "total_frames": len(self.frame_memories),
                    "settings": {
                        "log_every_n_frames": self.log_every_n_frames,
                        "detailed_log_every_n_frames": self.detailed_log_every_n_frames,
                        "memory_alert_threshold_mb": self.memory_alert_threshold_mb,
                        "gc_threshold_mb": self.gc_threshold_mb
                    }
                },
                "statistics": self.get_memory_statistics(),
                "frame_data": self.frame_memories,
                "stage_data": self.stage_memories,
                "garbage_collection_events": self.gc_events
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"📊 Memory analysis exported to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting frame data: {e}")
            return ""
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return final statistics."""
        try:
            # Stop tracemalloc if enabled
            final_traced_memory = 0
            peak_traced_memory = 0
            
            if self.tracemalloc_enabled:
                try:
                    current, peak = tracemalloc.get_traced_memory()
                    final_traced_memory = current / (1024**2)
                    peak_traced_memory = peak / (1024**2)
                    tracemalloc.stop()
                except:
                    pass
            
            # Get final statistics
            stats = self.get_memory_statistics()
            stats["tracemalloc"] = {
                "final_mb": round(final_traced_memory, 1),
                "peak_mb": round(peak_traced_memory, 1)
            }
            
            logger.info(f"🏁 FRAME MEMORY MONITORING STOPPED")
            logger.info(f"📊 Final Statistics:")
            logger.info(f"   • Frames Processed: {stats['summary']['total_frames_processed']}")
            logger.info(f"   • Peak Memory: {stats['summary']['peak_memory_mb']:.1f}MB")
            logger.info(f"   • Total Increase: {stats['summary']['total_memory_increase_mb']:.1f}MB")
            logger.info(f"   • Average FPS: {stats['summary']['average_fps']:.1f}")
            logger.info(f"   • Processing Time: {stats['summary']['total_processing_time']:.1f}s")
            logger.info(f"   • Garbage Collections: {stats['summary']['garbage_collections']}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error stopping memory monitoring: {e}")
            return {"error": str(e)}


# Global instance for easy access
frame_memory_monitor = PerFrameMemoryMonitor()

def start_frame_memory_monitoring(log_every_n_frames: int = 50, 
                                 detailed_log_every_n_frames: int = 200,
                                 memory_alert_threshold_mb: int = 2000,
                                 enable_tracemalloc: bool = True):
    """Convenience function to start frame memory monitoring."""
    global frame_memory_monitor
    frame_memory_monitor = PerFrameMemoryMonitor(
        log_every_n_frames=log_every_n_frames,
        detailed_log_every_n_frames=detailed_log_every_n_frames,
        memory_alert_threshold_mb=memory_alert_threshold_mb
    )
    frame_memory_monitor.start_monitoring(enable_tracemalloc=enable_tracemalloc)
    return frame_memory_monitor

def log_frame_memory(frame_number: int, 
                    total_frames: int = None,
                    stage: str = "processing",
                    additional_info: Dict = None):
    """Convenience function to log frame memory."""
    return frame_memory_monitor.log_frame_memory(
        frame_number, total_frames, stage, additional_info
    )

def stop_frame_memory_monitoring():
    """Convenience function to stop frame memory monitoring."""
    return frame_memory_monitor.stop_monitoring()
