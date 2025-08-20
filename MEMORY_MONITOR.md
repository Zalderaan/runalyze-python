# Enhanced Memory Monitor for RunAnalyze

## Overview
The enhanced memory monitor provides robust tracking of memory usage during video processing, with automatic fallback when the main API is under heavy load.

## Key Features

### 🔄 Automatic Fallback Mode
- Switches to direct process monitoring when API is unresponsive
- Continues tracking even during intensive video processing
- Smart recovery when API becomes available again

### 🎯 Enhanced Process Detection
- Identifies main.py/FastAPI processes automatically
- Tracks CPU usage during processing
- Monitors multiple Python processes simultaneously

### 📊 Comprehensive Monitoring
- Real-time memory usage tracking
- System memory and available space
- Temporary directory size monitoring
- Processing stage detection

### ⚡ Adaptive Monitoring
- Longer timeouts during video processing
- Dynamic update intervals based on load
- Resilient error handling

## Quick Start

### Basic Usage
```bash
# Standard monitoring
python memory_monitor.py

# Quick test (30 seconds)
python monitor.py --quick

# Heavy processing mode (optimized for video processing)
python monitor.py --heavy

# Debug mode (1-second intervals)
python monitor.py --debug

# Force fallback mode
python monitor.py --fallback
```

### Advanced Usage
```bash
# Custom configuration
python memory_monitor.py --interval 3 --url http://localhost:8000

# Monitor for specific duration
python memory_monitor.py --duration 300  # 5 minutes

# Show help
python memory_monitor.py --help
```

## Output Indicators

- 🟢 **Green**: Normal memory usage (<500MB)
- 🟡 **Yellow**: Elevated memory usage (500MB-1GB)
- 🟠 **Orange**: High memory usage (1-2GB)
- 🔴 **Red**: Critical memory usage (>2GB)

- 🌐 **API Mode**: Connected to FastAPI application
- 🔄 **Fallback Mode**: Direct process monitoring
- 🎯 **Target Process**: Main application process
- 🐍 **Python Process**: Other Python processes

## Monitoring During Video Processing

The monitor is specifically designed to handle the memory-intensive video processing operations:

1. **Longer Timeouts**: 30-second API timeouts during processing
2. **Fallback Detection**: Automatic switch when API is busy
3. **Process Tracking**: Direct memory monitoring of main.py process
4. **Smart Alerts**: Different alerts based on processing intensity

## Troubleshooting

### API Connection Issues
- Monitor automatically switches to fallback mode
- Direct process monitoring continues uninterrupted
- Recovery is automatic when API becomes responsive

### High Memory Usage Alerts
- 🚨 **Critical (>3GB)**: Processing may fail
- ⚠️ **High (>2GB)**: Monitor closely
- ⚡ **Elevated (>1.5GB)**: Normal during video processing

### Performance Tips
- Use `--heavy` mode during video processing
- Increase interval for less frequent updates: `--interval 5`
- Monitor tmp directory for file cleanup needs
