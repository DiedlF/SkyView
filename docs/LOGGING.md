# Skyview Logging Implementation

## Overview

Consolidated logging has been implemented across the Skyview backend using Python's standard `logging` module. All components now use a unified logging configuration with consistent formatting and output to both console and rotating log files.

## Structure

### `logging_config.py`
Shared logging configuration module that provides the `setup_logging()` function:

- **Format**: `YYYY-MM-DD HH:MM:SS | LEVEL | module | message`
- **Console handler**: INFO level and above
- **File handler**: Rotating file at `logs/skyview.log`
  - Max size: 10 MB per file
  - Backup count: 5 files
  - Captures all log levels (configurable per module)

### Log Files
Location: `/root/.openclaw/workspace/skyview/backend/logs/`

- `skyview.log` - Current log file
- `skyview.log.1` through `skyview.log.5` - Rotated backups

## Implementation by Module

### `ingest.py` (Data Ingestion)
**Log levels used:**
- `INFO`: Ingestion start/end, step completion, data availability, cleanup operations
- `WARNING`: Run not available, download failures for required variables
- `ERROR`: Decompression failures, GRIB parse errors
- `DEBUG`: Step skipping (already exists), optional variable fallbacks

**Key logged events:**
- Start of ingestion runs with model, run ID, and step range
- Individual step progress with grid dimensions
- Download and processing errors
- Old run cleanup
- Completion summary (steps ingested)

### `app.py` (FastAPI Server)
**Log levels used:**
- `INFO`: Server startup, API requests, available data summary
- `ERROR`: Data file not found
- `DEBUG`: Cache hits, data loading operations

**Key logged events:**
- Server startup with configuration paths
- Available model runs count and latest run info
- All HTTP requests: `METHOD /path - STATUS - duration_ms`
- Data loading operations and cache hits
- Missing data file errors

### `classify.py` (Cloud Classification)
**Log levels used:**
- `DEBUG`: Grid dimensions, classification distribution summary

**Key logged events:**
- Cloud type classification for each grid
- Distribution of classified types (clear, cu_hum, cb, etc.)

### `cron-ingest.sh` (Cron Wrapper)
**Changes:**
- Removed timestamped `echo` statements (Python logging handles this)
- Python script output flows through logging system
- Shell script now only logs stale lock warnings to stderr
- Simplified: relies on Python's unified logging

## Usage Examples

### View Real-time Logs
```bash
# Follow the log file
tail -f /root/.openclaw/workspace/skyview/backend/logs/skyview.log

# Show only INFO and above
tail -f /root/.openclaw/workspace/skyview/backend/logs/skyview.log | grep -E 'INFO|WARNING|ERROR'
```

### Run Ingestion with Logging
```bash
cd /root/.openclaw/workspace/skyview/backend

# Normal run (INFO to console + all to file)
python3 ingest.py latest --model icon-d2 --steps all

# The logs will show:
# - Ingestion start with run details
# - Progress for each step
# - Any errors or warnings
# - Completion summary
```

### Start API Server with Logging
```bash
cd /root/.openclaw/workspace/skyview/backend
python3 app.py

# Console shows:
# - Startup information
# - Available data summary
# - Each API request with timing
```

## Log Levels

- **DEBUG**: Detailed diagnostic information (classification details, cache hits, step skips)
- **INFO**: Normal operation flow (ingestion progress, API requests, startup)
- **WARNING**: Recoverable issues (data not ready, optional variables missing)
- **ERROR**: Failures requiring attention (download failures, parse errors, missing files)

## Benefits

1. **Unified format**: Consistent timestamps and structure across all components
2. **Rotating files**: Automatic log rotation prevents disk fill
3. **Dual output**: Console for real-time monitoring, files for history
4. **Proper levels**: Easy filtering and alerting
5. **Module identification**: Clear source of each log message
6. **No print() statements**: All output goes through proper logging

## Maintenance

The logging system requires no maintenance. Log rotation is automatic. To adjust:

- **Log level**: Change `level="INFO"` in `setup_logging()` calls (per module)
- **File size**: Modify `maxBytes` in `logging_config.py`
- **Retention**: Modify `backupCount` in `logging_config.py`
