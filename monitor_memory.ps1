# PowerShell Memory Monitor for RunAnalyze
# Usage: .\monitor_memory.ps1

param(
    [string]$ApiUrl = "http://localhost:8000",
    [int]$IntervalSeconds = 3,
    [int]$DurationMinutes = 0
)

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "  RUNALYZE MEMORY MONITOR (PowerShell)" -ForegroundColor Cyan
Write-Host "  API: $ApiUrl" -ForegroundColor Cyan
Write-Host "  Update Interval: $IntervalSeconds seconds" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""

$startTime = Get-Date
$maxMemory = 0
$endTime = if ($DurationMinutes -gt 0) { $startTime.AddMinutes($DurationMinutes) } else { $null }

function Get-ColorForMemory($memoryMB) {
    if ($memoryMB -gt 2000) { return "Red" }
    elseif ($memoryMB -gt 1000) { return "Yellow" }
    elseif ($memoryMB -gt 500) { return "Green" }
    else { return "White" }
}

function Format-FileSize($bytes) {
    if ($bytes -gt 1GB) { return "{0:N2} GB" -f ($bytes / 1GB) }
    elseif ($bytes -gt 1MB) { return "{0:N2} MB" -f ($bytes / 1MB) }
    elseif ($bytes -gt 1KB) { return "{0:N2} KB" -f ($bytes / 1KB) }
    else { return "{0} bytes" -f $bytes }
}

try {
    while ($true) {
        if ($endTime -and (Get-Date) -gt $endTime) {
            Write-Host "Duration limit reached. Stopping monitor." -ForegroundColor Yellow
            break
        }

        $timestamp = Get-Date -Format "HH:mm:ss"
        $elapsed = [math]::Round(((Get-Date) - $startTime).TotalSeconds, 0)

        # Get API memory status
        try {
            $response = Invoke-RestMethod -Uri "$ApiUrl/memory-status/" -Method GET -TimeoutSec 5
            
            $processMB = [math]::Round($response.process_memory.rss_mb, 1)
            $systemPercent = [math]::Round($response.system_memory.used_percent, 1)
            $availableGB = [math]::Round($response.system_memory.available_gb, 1)
            $tmpSizeMB = [math]::Round($response.tmp_directory.total_size_mb, 1)
            $tmpFiles = $response.tmp_directory.file_count
            
            # Track max memory
            if ($processMB -gt $maxMemory) {
                $maxMemory = $processMB
            }
            
            # Color for memory usage
            $memoryColor = Get-ColorForMemory $processMB
            
            # Main status line
            $statusLine = "$timestamp ({0}s) | Process: {1}MB (Peak: {2}MB) | System: {3}% | Available: {4}GB" -f $elapsed, $processMB, $maxMemory, $systemPercent, $availableGB
            
            Write-Host $statusLine -ForegroundColor $memoryColor
            
            # Additional info
            if ($tmpSizeMB -gt 0) {
                Write-Host "   Tmp: ${tmpSizeMB}MB ($tmpFiles files)" -ForegroundColor Gray
            }
            
            # Tracker info (if video processing)
            if ($response.tracker_summary -and $response.tracker_summary.peak_memory_mb) {
                $peakMB = [math]::Round($response.tracker_summary.peak_memory_mb, 1)
                $increaseMB = [math]::Round($response.tracker_summary.total_increase_mb, 1)
                Write-Host "   Video Processing: Peak ${peakMB}MB (+${increaseMB}MB)" -ForegroundColor Magenta
            }
            
            # Memory recommendations
            if ($response.recommendations) {
                foreach ($rec in $response.recommendations) {
                    if ($rec -notlike "*normal*") {
                        Write-Host "   ⚠️  $rec" -ForegroundColor Yellow
                    }
                }
            }
            
        }
        catch {
            Write-Host "$timestamp - API Error: $($_.Exception.Message)" -ForegroundColor Red
            
            # Fallback: Check Python processes directly
            $pythonProcs = Get-Process | Where-Object { $_.ProcessName -like "*python*" }
            if ($pythonProcs) {
                foreach ($proc in $pythonProcs) {
                    $memoryMB = [math]::Round($proc.WorkingSet / 1MB, 1)
                    Write-Host "   Python PID $($proc.Id): ${memoryMB}MB" -ForegroundColor Cyan
                }
            }
        }
        
        Start-Sleep -Seconds $IntervalSeconds
    }
}
catch [System.Management.Automation.RuntimeException] {
    Write-Host "`nMonitoring stopped by user (Ctrl+C)" -ForegroundColor Yellow
}
finally {
    $totalTime = [math]::Round(((Get-Date) - $startTime).TotalSeconds, 1)
    Write-Host "`n" + "=" * 40 -ForegroundColor Cyan
    Write-Host "MONITORING SUMMARY" -ForegroundColor Cyan
    Write-Host "=" * 40 -ForegroundColor Cyan
    Write-Host "Total Time: ${totalTime}s" -ForegroundColor White
    Write-Host "Peak Memory: ${maxMemory}MB" -ForegroundColor White
    Write-Host "=" * 40 -ForegroundColor Cyan
}
