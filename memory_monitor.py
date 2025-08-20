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
import os
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
        self.api_failures = 0
        self.max_api_failures = 3
        self.fallback_mode = False
        
    def find_python_processes(self):
        """Find Python processes (likely FastAPI app) with enhanced detection"""
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'cpu_percent']):
            try:
                proc_info = proc.info
                if not proc_info['name']:
                    continue
                    
                # Look for Python processes
                if 'python' in proc_info['name'].lower() or 'py' in proc_info['name'].lower():
                    cmdline = ' '.join(proc_info['cmdline']) if proc_info['cmdline'] else ''
                    
                    # Enhanced detection for FastAPI/main.py processes
                    is_target_process = any([
                        'main.py' in cmdline,
                        'uvicorn' in cmdline,
                        'fastapi' in cmdline,
                        'runalyze' in cmdline.lower(),
                        ('python' in cmdline and 'main' in cmdline)
                    ])
                    
                    if is_target_process or len(python_processes) < 5:  # Include top 5 Python processes
                        memory_info = proc_info['memory_info']
                        memory_mb = memory_info.rss / (1024 * 1024)
                        
                        python_processes.append({
                            'pid': proc_info['pid'],
                            'name': proc_info['name'],
                            'memory_mb': memory_mb,
                            'memory_vms_mb': memory_info.vms / (1024 * 1024),
                            'cpu_percent': proc_info.get('cpu_percent', 0),
                            'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline,
                            'is_target': is_target_process
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
            except Exception as e:
                # Don't let process enumeration errors stop monitoring
                continue
                
        # Sort by memory usage, prioritizing target processes
        python_processes.sort(key=lambda x: (not x['is_target'], -x['memory_mb']))
        return python_processes
    
    def get_api_memory_status(self):
        """Get memory status from the FastAPI app with enhanced error handling"""
        if self.fallback_mode:
            return {"error": "API in fallback mode - using direct process monitoring"}
            
        try:
            # Longer timeout for video processing scenarios
            timeout = 30 if self.api_failures == 0 else 10
            response = requests.get(f"{self.api_url}/memory-status/", timeout=timeout)
            
            if response.status_code == 200:
                self.api_failures = 0  # Reset failure counter on success
                return response.json()
            else:
                self.api_failures += 1
                return {"error": f"API returned status {response.status_code}"}
                
        except requests.exceptions.Timeout:
            self.api_failures += 1
            return {"error": "API timeout - server may be processing video"}
        except requests.exceptions.ConnectionError:
            self.api_failures += 1
            return {"error": "API connection failed - server may be down"}
        except requests.exceptions.RequestException as e:
            self.api_failures += 1
            return {"error": f"API request failed: {str(e)}"}
        finally:
            # Switch to fallback mode if too many failures
            if self.api_failures >= self.max_api_failures:
                if not self.fallback_mode:
                    print(f"⚠️  Switching to fallback mode after {self.api_failures} API failures")
                self.fallback_mode = True
    
    def get_direct_memory_stats(self, processes):
        """Get memory stats directly from processes when API is unavailable"""
        if not processes:
            return {}
            
        # Find the main process (target process with highest memory)
        main_process = None
        for proc in processes:
            if proc.get('is_target', False):
                main_process = proc
                break
        
        if not main_process and processes:
            main_process = processes[0]  # Fallback to highest memory process
            
        if not main_process:
            return {}
            
        # Get system memory info
        system_memory = psutil.virtual_memory()
        
        return {
            "process_memory": {
                "rss_mb": main_process['memory_mb'],
                "vms_mb": main_process.get('memory_vms_mb', 0),
                "cpu_percent": main_process.get('cpu_percent', 0)
            },
            "system_memory": {
                "used_percent": system_memory.percent,
                "available_gb": system_memory.available / (1024**3),
                "total_gb": system_memory.total / (1024**3)
            },
            "tmp_directory": {
                "total_size_mb": self.get_tmp_directory_size(),
                "file_count": self.get_tmp_file_count()
            },
            "fallback_mode": True,
            "main_process_pid": main_process['pid']
        }
    
    def get_tmp_directory_size(self):
        """Get size of tmp directory"""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk("tmp"):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(file_path)
                    except (OSError, IOError):
                        pass
            return total_size / (1024 * 1024)  # Convert to MB
        except:
            return 0
    
    def get_tmp_file_count(self):
        """Get count of files in tmp directory"""
        try:
            file_count = 0
            for dirpath, dirnames, filenames in os.walk("tmp"):
                file_count += len(filenames)
            return file_count
        except:
            return 0
        """Display monitoring header"""
        print("=" * 80)
        print("  RUNALYZE MEMORY MONITOR")
        print(f"  Monitoring: {self.api_url}")
        print(f"  Update Interval: {self.update_interval}s")
    def display_header(self):
        """Display monitoring header"""
        print("=" * 80)
        print("  RUNALYZE MEMORY MONITOR")
        print(f"  Monitoring: {self.api_url}")
        print(f"  Update Interval: {self.update_interval}s")
        print(f"  Started: {self.start_time.strftime('%H:%M:%S')}")
        print(f"  Fallback Mode: {'Enabled' if self.fallback_mode else 'Disabled'}")
        print("=" * 80)
        print()
    
    def format_memory_data(self, api_data, processes):
        """Format memory data for display with enhanced fallback support"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        # Use direct process monitoring if API fails
        if "error" in api_data and processes:
            direct_stats = self.get_direct_memory_stats(processes)
            if direct_stats:
                process_memory = direct_stats.get("process_memory", {})
                system_memory = direct_stats.get("system_memory", {})
                tmp_directory = direct_stats.get("tmp_directory", {})
                
                process_mb = process_memory.get("rss_mb", 0)
                system_percent = system_memory.get("used_percent", 0)
                available_gb = system_memory.get("available_gb", 0)
                tmp_size_mb = tmp_directory.get("total_size_mb", 0)
                tmp_files = tmp_directory.get("file_count", 0)
                
                # Track maximum memory
                if process_mb > self.max_memory:
                    self.max_memory = process_mb
                
                # Store history for fallback mode
                history_entry = {
                    "timestamp": timestamp,
                    "elapsed": elapsed,
                    "process_mb": process_mb,
                    "system_percent": system_percent,
                    "tmp_size_mb": tmp_size_mb,
                    "fallback_mode": True
                }
                self.memory_history.append(history_entry)
                
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
                    "tracker_summary": {},
                    "processes": processes,
                    "error": None,
                    "fallback_mode": True,
                    "api_error": api_data.get("error", "")
                }
        
        # API data processing (original logic)
        if "error" not in api_data:
            try:
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
                
                # Store history with error handling (for both API and fallback data)
                history_entry = {
                    "timestamp": timestamp,
                    "elapsed": elapsed,
                    "process_mb": process_mb,
                    "system_percent": system_percent,
                    "tmp_size_mb": tmp_size_mb,
                    "fallback_mode": False
                }
                self.memory_history.append(history_entry)
                
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
                    "processes": processes,
                    "error": None,
                    "fallback_mode": False
                }
            except Exception as e:
                return {
                    "timestamp": timestamp,
                    "elapsed": f"{elapsed:.0f}s",
                    "error": f"Data parsing error: {str(e)}",
                    "processes": processes,
                    "fallback_mode": False
                }
        else:
            return {
                "timestamp": timestamp,
                "elapsed": f"{elapsed:.0f}s",
                "error": api_data["error"], 
                "processes": processes,
                "fallback_mode": self.fallback_mode
            }
    
    def display_memory_status(self, data):
        """Display formatted memory status with enhanced error handling and fallback support"""
        # Ensure we have basic fields
        timestamp = data.get("timestamp", datetime.now().strftime("%H:%M:%S"))
        elapsed = data.get("elapsed", "0s")
        fallback_mode = data.get("fallback_mode", False)
        api_error = data.get("api_error", "")
        
        # Handle error cases but still show process info if available
        if data.get("error") and not fallback_mode:
            error_msg = f"❌ {timestamp} ({elapsed}) - Error: {data['error']}"
            if "timeout" in data['error'].lower():
                error_msg += " [Server may be processing video]"
            print(error_msg)
            
            if data.get('processes'):
                print("   📊 Direct Process Monitor:")
                for proc in data['processes'][:3]:
                    target_indicator = "🎯" if proc.get('is_target', False) else "  "
                    cpu_info = f" ({proc.get('cpu_percent', 0):.1f}% CPU)" if proc.get('cpu_percent', 0) > 0 else ""
                    print(f"   {target_indicator} PID {proc['pid']}: {proc['memory_mb']:.1f}MB - {proc['name']}{cpu_info}")
            print()
            return
        
        # Safe access to all data fields
        process_mb = data.get('process_mb', 0)
        max_mb = data.get('max_mb', 0)
        system_percent = data.get('system_percent', 0)
        available_gb = data.get('available_gb', 0)
        tmp_size_mb = data.get('tmp_size_mb', 0)
        tmp_files = data.get('tmp_files', 0)
        tracker_summary = data.get('tracker_summary', {})
        processes = data.get('processes', [])
        
        # Main status line with fallback indicator
        mode_indicator = "🔄" if fallback_mode else "🌐"
        status_line = (
            f"{mode_indicator} {timestamp} ({elapsed}) | "
            f"📊 Process: {process_mb:.1f}MB (Peak: {max_mb:.1f}MB) | "
            f"💻 System: {system_percent:.1f}% | "
            f"💾 Available: {available_gb:.1f}GB"
        )
        
        # Color coding based on memory usage
        if process_mb > 2000:  # > 2GB
            status_line = f"🔴 {status_line}"
        elif process_mb > 1000:  # > 1GB
            status_line = f"🟠 {status_line}"
        elif process_mb > 500:  # > 500MB
            status_line = f"🟡 {status_line}"
        else:
            status_line = f"🟢 {status_line}"
        
        print(status_line)
        
        # Show mode info
        if fallback_mode:
            print(f"   🔄 Fallback Mode: Direct process monitoring (API: {api_error})")
        
        # Tmp directory info
        if tmp_size_mb > 0:
            print(f"   📁 Tmp: {tmp_size_mb:.1f}MB ({tmp_files} files)")
        
        # Tracker summary (if video is being processed)
        if tracker_summary and isinstance(tracker_summary, dict):
            peak_mb = tracker_summary.get('peak_memory_mb', 0)
            increase_mb = tracker_summary.get('total_increase_mb', 0)
            if peak_mb > 0:
                print(f"   🎥 Video Processing: Peak {peak_mb:.1f}MB, Increase +{increase_mb:.1f}MB")
        
        # Enhanced process list
        if processes:
            target_processes = [p for p in processes if p.get('is_target', False)]
            if target_processes:
                print("   🎯 Target Processes:")
                for proc in target_processes[:2]:
                    cpu_info = f" ({proc.get('cpu_percent', 0):.1f}% CPU)" if proc.get('cpu_percent', 0) > 0 else ""
                    print(f"      PID {proc['pid']}: {proc['memory_mb']:.1f}MB{cpu_info}")
            
            other_processes = [p for p in processes if not p.get('is_target', False)]
            if other_processes and len(other_processes) > 0:
                print("   🐍 Other Python Processes:")
                for proc in other_processes[:2]:
                    cpu_info = f" ({proc.get('cpu_percent', 0):.1f}% CPU)" if proc.get('cpu_percent', 0) > 0 else ""
                    print(f"      PID {proc['pid']}: {proc['memory_mb']:.1f}MB - {proc['name'][:20]}{cpu_info}")
        
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
        """Start monitoring with enhanced resilience during video processing"""
        self.display_header()
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        try:
            end_time = datetime.now() + timedelta(seconds=duration) if duration else None
            
            while True:
                if end_time and datetime.now() >= end_time:
                    break
                
                try:
                    # Always get process info first (more reliable)
                    processes = self.find_python_processes()
                    
                    # Get API data (may fail during heavy processing)
                    api_data = self.get_api_memory_status()
                    
                    # Format and display with fallback support
                    formatted_data = self.format_memory_data(api_data, processes)
                    self.display_memory_status(formatted_data)
                    
                    # Enhanced alerts based on memory and CPU usage
                    process_mb = formatted_data.get("process_mb", 0)
                    if process_mb > 3000:
                        print("🚨 CRITICAL MEMORY USAGE: >3GB - Video processing may fail")
                    elif process_mb > 2000:
                        print("⚠️  HIGH MEMORY USAGE: >2GB - Monitor closely")
                    elif process_mb > 1500:
                        print("⚡ ELEVATED MEMORY: >1.5GB - Video processing in progress")
                    
                    # Check for high CPU usage (indicates active processing)
                    if processes:
                        max_cpu = max(proc.get('cpu_percent', 0) for proc in processes)
                        if max_cpu > 80:
                            print(f"🔥 HIGH CPU USAGE: {max_cpu:.1f}% - Intensive processing detected")
                    
                    # Reset error counter on successful cycle
                    consecutive_errors = 0
                    
                    # Dynamic sleep interval based on processing intensity
                    sleep_interval = self.update_interval
                    if process_mb > 1000:  # Reduce frequency during heavy processing
                        sleep_interval = min(self.update_interval * 1.5, 10)
                    
                    time.sleep(sleep_interval)
                    
                except Exception as e:
                    consecutive_errors += 1
                    print(f"⚠️  Monitoring cycle error #{consecutive_errors}: {e}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"❌ Too many consecutive errors ({consecutive_errors}). Switching to basic monitoring...")
                        # Try basic process monitoring only
                        try:
                            processes = self.find_python_processes()
                            if processes:
                                print("📊 Basic Process Monitor:")
                                for proc in processes[:3]:
                                    print(f"   PID {proc['pid']}: {proc['memory_mb']:.1f}MB - {proc['name']}")
                        except:
                            print("❌ Basic monitoring also failed. Retrying in 10 seconds...")
                        consecutive_errors = 0  # Reset counter
                        time.sleep(10)
                    else:
                        time.sleep(self.update_interval)
                    continue
                    
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Fatal monitoring error: {e}")
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
