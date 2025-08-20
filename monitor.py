#!/usr/bin/env python3
"""
Quick launcher for memory monitoring during video processing

Usage examples:
  python monitor.py                    # Start basic monitoring
  python monitor.py --heavy           # Monitoring optimized for heavy processing
  python monitor.py --debug          # Debug mode with extra logging
  python monitor.py --fallback       # Force fallback mode
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Launch memory monitor with preset configurations"""
    
    # Default arguments
    args = ["python", "memory_monitor.py"]
    
    # Parse simple arguments
    if "--help" in sys.argv or "-h" in sys.argv:
        args.append("--help")
    elif "--heavy" in sys.argv:
        # Configuration for heavy video processing
        args.extend(["--interval", "3", "--url", "http://localhost:8000"])
        print("🎥 Heavy Processing Mode: 3-second intervals, extended timeouts")
    elif "--debug" in sys.argv:
        # Configuration for debugging
        args.extend(["--interval", "1", "--processes"])
        print("🐛 Debug Mode: 1-second intervals, process details")
    elif "--fallback" in sys.argv:
        # Force fallback mode by using invalid URL
        args.extend(["--url", "http://localhost:9999", "--interval", "2"])
        print("🔄 Fallback Mode: Direct process monitoring only")
    elif "--quick" in sys.argv:
        # Quick test for 30 seconds
        args.extend(["--duration", "30", "--interval", "2"])
        print("⚡ Quick Test: 30-second monitoring session")
    else:
        # Default mode
        args.extend(["--interval", "2"])
        print("📊 Standard Mode: 2-second intervals")
    
    print(f"Launching: {' '.join(args)}")
    print("-" * 50)
    
    try:
        # Launch the memory monitor
        subprocess.run(args)
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped")
    except Exception as e:
        print(f"❌ Error launching monitor: {e}")

if __name__ == "__main__":
    main()
