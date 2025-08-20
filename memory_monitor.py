#!/usr/bin/env python3
"""
Memory Monitor for RunAnalyze Video Processing

This script monitors memory usage of the FastAPI application in real-time
and provides detailed insights into memory consumption during video processing.

Usage:
    python memory_monitor.py

Features:
- Real-time memory monitoring
- Memory usage graphs
- Process detection and tracking
- Memory alerts and recommendations
"""

import psutil
import time
import requests
import json
from datetime import datetime, timedelta
import argparse
import sys

class MemoryMonitor:
    def __init__(self, api_url="http://localhost:8000", update_interval=2):
        self.api_url = api_url
        self.update_interval = update_interval
        self.memory_history = []
        self.start_time = datetime.now()
        self.max_memory = 0
        self.process_pids = []
        
    def find_python_processes(self):
        """Find Python processes (likely FastAPI app)"""
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if 'main.py' in cmdline or 'uvicorn' in cmdline or 'fastapi' in cmdline:
                        memory_mb = proc.info['memory_info'].rss / (1024 * 1024)
                        python_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'memory_mb': memory_mb,
                            'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return python_processes
    
    def get_api_memory_status(self):
        """Get memory status from the FastAPI app"""
        try:
            response = requests.get(f"{self.api_url}/memory-status/", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API returned status {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}
    
    def display_header(self):
        """Display monitoring header"""
        print("=" * 80)
        print("  RUNALYZE MEMORY MONITOR")
        print(f"  Monitoring: {self.api_url}")
        print(f"  Update Interval: {self.update_interval}s")
        print(f"  Started: {self.start_time.strftime('%H:%M:%S')}")
        print("=" * 80)
        print()
    
    def format_memory_data(self, api_data, processes):
        """Format memory data for display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        # API data
        if "error" not in api_data:
            process_memory = api_data.get("process_memory", {})
            system_memory = api_data.get("system_memory", {})
            tmp_directory = api_data.get("tmp_directory", {})
            tracker_summary = api_data.get("tracker_summary", {})
            
            process_mb = process_memory.get("rss_mb", 0)
            system_percent = system_memory.get("used_percent", 0)
            available_gb = system_memory.get("available_gb", 0)
            tmp_size_mb = tmp_directory.get("total_size_mb", 0)
            tmp_files = tmp_directory.get("file_count", 0)
            
            # Track maximum memory
            if process_mb > self.max_memory:
                self.max_memory = process_mb
            
            # Store history
            self.memory_history.append({
                "timestamp": timestamp,
                "elapsed": elapsed,
                "process_mb": process_mb,
                "system_percent": system_percent,
                "tmp_size_mb": tmp_size_mb
            })
            
            # Keep only last 100 entries
            if len(self.memory_history) > 100:
                self.memory_history.pop(0)
            
            return {
                "timestamp": timestamp,
                "elapsed": f"{elapsed:.0f}s",
                "process_mb": process_mb,
                "max_mb": self.max_memory,
                "system_percent": system_percent,
                "available_gb": available_gb,
                "tmp_size_mb": tmp_size_mb,
                "tmp_files": tmp_files,
                "tracker_summary": tracker_summary,
                "processes": processes
            }
        else:
            return {"error": api_data["error"], "processes": processes}
    
    def display_memory_status(self, data):
        """Display formatted memory status"""
        if "error" in data:
            print(f"❌ {data['timestamp']} - API Error: {data['error']}")
            if data['processes']:
                print("   Direct Process Monitor:")
                for proc in data['processes']:
                    print(f"   PID {proc['pid']}: {proc['memory_mb']:.1f}MB - {proc['name']}")
            print()
            return
        
        # Main status line
        status_line = (
            f"🕐 {data['timestamp']} ({data['elapsed']}) | "
            f"📊 Process: {data['process_mb']:.1f}MB (Peak: {data['max_mb']:.1f}MB) | "
            f"💻 System: {data['system_percent']:.1f}% | "
            f"💾 Available: {data['available_gb']:.1f}GB"
        )
        
        # Color coding based on memory usage
        if data['process_mb'] > 2000:  # > 2GB
            status_line = f"🔴 {status_line}"
        elif data['process_mb'] > 1000:  # > 1GB
            status_line = f"🟠 {status_line}"
        elif data['process_mb'] > 500:  # > 500MB
            status_line = f"🟡 {status_line}"
        else:
            status_line = f"🟢 {status_line}"
        
        print(status_line)
        
        # Tmp directory info
        if data['tmp_size_mb'] > 0:
            print(f"   📁 Tmp: {data['tmp_size_mb']:.1f}MB ({data['tmp_files']} files)")
        
        # Tracker summary (if video is being processed)
        if data['tracker_summary'] and 'peak_memory_mb' in data['tracker_summary']:
            summary = data['tracker_summary']
            print(f"   🎥 Video Processing: Peak {summary.get('peak_memory_mb', 0):.1f}MB, "
                  f"Increase +{summary.get('total_increase_mb', 0):.1f}MB")
        
        # Process list (if available)
        if data['processes']:
            for proc in data['processes'][:3]:  # Show top 3 processes
                print(f"   🐍 PID {proc['pid']}: {proc['memory_mb']:.1f}MB")
        
        print()
    
    def display_summary(self):
        """Display session summary"""
        if not self.memory_history:
            return
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        avg_memory = sum(h['process_mb'] for h in self.memory_history) / len(self.memory_history)
        
        print("\n" + "=" * 60)
        print("  MONITORING SESSION SUMMARY")
        print("=" * 60)
        print(f"Total Monitoring Time: {total_time:.1f} seconds")
        print(f"Data Points Collected: {len(self.memory_history)}")
        print(f"Peak Memory Usage: {self.max_memory:.1f}MB")
        print(f"Average Memory Usage: {avg_memory:.1f}MB")
        
        # Show memory trend
        if len(self.memory_history) >= 2:
            first_memory = self.memory_history[0]['process_mb']
            last_memory = self.memory_history[-1]['process_mb']
            trend = last_memory - first_memory
            trend_direction = "📈" if trend > 10 else "📉" if trend < -10 else "➡️"
            print(f"Memory Trend: {trend_direction} {trend:+.1f}MB")
        
        print("=" * 60)
    
    def monitor(self, duration=None):
        """Start monitoring with optional duration limit"""
        self.display_header()
        
        try:
            end_time = datetime.now() + timedelta(seconds=duration) if duration else None
            
            while True:
                if end_time and datetime.now() >= end_time:
                    break
                
                # Get data from both API and direct process monitoring
                api_data = self.get_api_memory_status()
                processes = self.find_python_processes()
                
                # Format and display
                formatted_data = self.format_memory_data(api_data, processes)
                self.display_memory_status(formatted_data)
                
                # Check for alerts
                if "process_mb" in formatted_data and formatted_data["process_mb"] > 2000:
                    print("⚠️  HIGH MEMORY USAGE ALERT: Consider optimizing video processing")
                
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Monitoring error: {e}")
        finally:
            self.display_summary()

def main():
    parser = argparse.ArgumentParser(description="Monitor RunAnalyze memory usage")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="FastAPI application URL (default: http://localhost:8000)")
    parser.add_argument("--interval", type=int, default=2,
                       help="Update interval in seconds (default: 2)")
    parser.add_argument("--duration", type=int,
                       help="Monitoring duration in seconds (default: unlimited)")
    parser.add_argument("--processes", action="store_true",
                       help="Also monitor Python processes directly")
    
    args = parser.parse_args()
    
    monitor = MemoryMonitor(api_url=args.url, update_interval=args.interval)
    
    print("Starting RunAnalyze Memory Monitor...")
    print("Press Ctrl+C to stop monitoring\n")
    
    monitor.monitor(duration=args.duration)

if __name__ == "__main__":
    main()
