# MoveEnetFlow - Event-based Human Pose Estimation

This directory contains tools for event-based human pose estimation using the MoveEnet architecture with EROS (Event-based Representation Of Surfaces) and velocity estimation.

## Contents

### Executables

1. **moveEnet_flow** - Offline HPE processing (NEW)
   - Processes event data from .log files offline
   - Generates EROS, SAE, and binary event representations
   - Detects human poses using MoveEnet
   - Estimates joint velocities
   - Saves results to CSV and video files
   - Single-threaded sequential processing

## Dependencies

- **YARP** (>= 3.3) - Communication framework
- **OpenCV** (>= 4.0) - Computer vision and image processing
- **event-driven** - Event processing library
- **hpe-core** - Human pose estimation core library (includes MoveEnet)
- **ROS** (optional, only for edpr-april and extract_eros)

## Building

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build all executables
make -j$(nproc)

# Install (optional)
sudo make install
```

## Usage

### moveEnet_flow - Offline HPE Processing

Process event data from .log files and extract human pose with velocity estimation.

#### Basic Usage

```bash
./moveEnet_flow --log_path /path/to/data.log --output_csv results.csv --output_video output.avi
```

#### Command-line Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--help` | flag | - | Show help message |
| `--name` | string | `/offline_hpe` | YARP module name for ports |
| `--log_path` | string | *required* | Path to input .log file with event data |
| `--w` | int | 640 | Camera width in pixels |
| `--h` | int | 480 | Camera height in pixels |
| `--frequency` | float | 100.0 | Batch processing frequency in Hz |
| `--roi` | int | 20 | ROI size for velocity estimation (pixels) |
| `--confidence` | float | 0.4 | Skeleton confidence threshold (0-1) |
| `--checkpoint_path` | string | *see below* | Path to MoveEnet model checkpoint |
| `--output_csv` | string | - | Path to save CSV results (optional) |
| `--output_video` | string | - | Path to save video output (optional) |
| `--no_viz` | flag | false | Disable real-time visualization |

**Default checkpoint path:**
```
/usr/local/src/hpe-core/example/movenet/models/e97_valacc0.81209.pth
```

#### Example Commands

1. **Process with CSV output only:**
```bash
./moveEnet_flow --log_path /data/recording.log --output_csv poses.csv --no_viz
```

2. **Process with video output:**
```bash
./moveEnet_flow --log_path /data/recording.log \
    --output_video output.avi \
    --frequency 50 \
    --w 640 --h 480
```

3. **Full processing with all outputs:**
```bash
./moveEnet_flow --log_path /data/recording.log \
    --output_csv results.csv \
    --output_video visualization.avi \
    --checkpoint_path /path/to/model.pth \
    --frequency 100 \
    --roi 25 \
    --confidence 0.5
```

4. **Preview only (no saving):**
```bash
./moveEnet_flow --log_path /data/recording.log --frequency 30
```

#### Output Format

**CSV File Structure:**
```
timestamp,joint0_x,joint0_y,joint0_vx,joint0_vy,joint0_conf,...,joint12_x,joint12_y,joint12_vx,joint12_vy,joint12_conf
0.010000,320.5,240.3,1.2,-0.5,0.95,...
0.020000,321.2,239.8,1.1,-0.6,0.94,...
```

Each row contains:
- `timestamp`: Time in seconds
- For each of 13 joints:
  - `jointN_x`, `jointN_y`: Joint position (pixels)
  - `jointN_vx`, `jointN_vy`: Joint velocity (pixels/s)
  - `jointN_conf`: Detection confidence (0-1)

**Video Output:**
- Format: MJPEG (.avi)
- Frame rate: Same as `--frequency` parameter
- Content: EROS representation with skeleton overlay (red) and velocity vectors (yellow)
- Overlay text: Current timestamp and processing frequency

#### Interactive Controls

When visualization is enabled (default):
- **ESC**: Stop processing and exit
- Window shows: EROS background + detected skeleton + velocity vectors

## Event Representations

The system generates three different event representations:

1. **EROS (Event-based Representation of Surfaces)**
   - Kernel size: 7x7 (default)
   - Decay factor (alpha): 0.3
   - Used for pose detection with MoveEnet
   - Smoothed with Gaussian blur before sending to detector

2. **SAE (Surface of Active Events)**
   - Maintains timestamp of last event at each pixel
   - Used for velocity estimation
   - Enables optical flow-like computation in event domain

3. **Binary Events**
   - Simple binary accumulation of events
   - Used for auxiliary visualization

## Architecture Overview

### moveEnet_flow Processing Pipeline

```
.log file
    ↓
Event Loader (ev::offlineLoader)
    ↓
Batch Events @ frequency Hz
    ↓
    ├→ EROS Handler → MoveEnet → Joint Positions + Confidence
    ├→ SAE Handler ───────────────┐
    └→ Binary Handler             │
                                  ↓
                    Velocity Estimator (pwtripletvelocity)
                                  ↓
                          Joint Velocities
                                  ↓
                    ┌─────────────┴─────────────┐
                    ↓                           ↓
                CSV Output                  Video Output
        (positions, velocities,          (EROS + skeleton
         confidences)                     + velocity vectors)
```

### Key Components

- **Event Loader**: Reads events from .log files incrementally
- **EROS Handler**: Generates surface representation for pose detection
- **SAE Handler**: Maintains event timing for velocity estimation
- **MoveEnet**: Deep learning model for pose detection (runs as separate Python process)
- **Velocity Estimator**: Computes joint velocities using SAE and ROI-based approach
- **Output Handlers**: Save CSV data and video visualization

## Troubleshooting

### Common Issues

1. **"Could not connect to YARP"**
   - Ensure YARP server is running: `yarp detect`
   - If not, start it: `yarpserver &`

2. **"Could not open data file"**
   - Check that the .log file path is correct
   - Verify the file format is compatible with `ev::offlineLoader`

3. **"MoveEnet not found"**
   - Ensure hpe-core is installed with MoveEnet support
   - Check checkpoint path is correct
   - Verify Python3 and required dependencies are installed

4. **Video not saving**
   - Check disk space
   - Verify you have write permissions to output directory
   - Try different video codec if MJPEG fails

5. **Low frame rate in visualization**
   - Reduce `--frequency` parameter
   - Use `--no_viz` for faster processing without display
   - Processing speed depends on MoveEnet inference time

### Performance Tips

- **Faster processing**: Use `--no_viz` and lower `--frequency`
- **Better accuracy**: Use higher `--frequency` for finer temporal resolution
- **Smoother visualization**: Match `--frequency` to display refresh rate (~60 Hz)
- **Large datasets**: Process in batches or reduce frequency

## Citation

If you use this code in your research, please cite the relevant papers for:
- MoveEnet architecture
- EROS representation
- Event-driven vision processing

## License

Check with your institution for licensing information.

## Authors

- Daria Berretta
- Based on edpr-april by Franco Di Pietro and Arren Glover
