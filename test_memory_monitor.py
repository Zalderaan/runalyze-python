#!/usr/bin/env python3
"""
Test script for the enhanced memory monitor
"""

from memory_monitor import MemoryMonitor
import time

def test_memory_monitor():
    """Test the memory monitor functionality"""
    print("🧪 Testing Enhanced Memory Monitor")
    print("=" * 50)
    
    # Initialize monitor
    monitor = MemoryMonitor(api_url="http://localhost:8000", update_interval=1)
    
    # Test process detection
    print("1. Testing process detection...")
    processes = monitor.find_python_processes()
    print(f"   Found {len(processes)} Python processes")
    
    for i, proc in enumerate(processes[:3]):
        target_indicator = "🎯" if proc.get('is_target', False) else "  "
        print(f"   {target_indicator} #{i+1}: PID {proc['pid']} - {proc['memory_mb']:.1f}MB - {proc['name']}")
    
    # Test API call
    print("\n2. Testing API connection...")
    api_data = monitor.get_api_memory_status()
    if "error" in api_data:
        print(f"   ❌ API Error: {api_data['error']}")
        print("   ✅ Fallback mode will be used")
    else:
        print("   ✅ API connection successful")
    
    # Test direct memory stats
    print("\n3. Testing direct memory stats...")
    direct_stats = monitor.get_direct_memory_stats(processes)
    if direct_stats:
        process_memory = direct_stats.get("process_memory", {})
        system_memory = direct_stats.get("system_memory", {})
        print(f"   Process Memory: {process_memory.get('rss_mb', 0):.1f}MB")
        print(f"   System Memory: {system_memory.get('used_percent', 0):.1f}% used")
        print(f"   Available: {system_memory.get('available_gb', 0):.1f}GB")
    
    # Test tmp directory monitoring
    print("\n4. Testing tmp directory monitoring...")
    tmp_size = monitor.get_tmp_directory_size()
    tmp_files = monitor.get_tmp_file_count()
    print(f"   Tmp directory: {tmp_size:.1f}MB, {tmp_files} files")
    
    # Test data formatting
    print("\n5. Testing data formatting...")
    formatted_data = monitor.format_memory_data(api_data, processes)
    print(f"   Timestamp: {formatted_data['timestamp']}")
    print(f"   Process Memory: {formatted_data.get('process_mb', 0):.1f}MB")
    print(f"   Fallback Mode: {formatted_data.get('fallback_mode', False)}")
    
    print("\n✅ All tests completed!")
    print("\n🚀 Ready to monitor video processing!")
    
    return True

if __name__ == "__main__":
    try:
        test_memory_monitor()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
