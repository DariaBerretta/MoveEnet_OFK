# Complete Pipeline Usage Guide

This guide explains how to use the `complete_pipeline.py` script for training, validating, and testing GraphEnet-v2 models with video visualization.

## What the Script Does

The `complete_pipeline.py` script provides a unified workflow that:

1. **Creates and trains** a GraphEnet-v2 model on SCARF graph data
2. **Validates** the model performance with metrics (PCK, MPJPE)
3. **Tests** the model on new video data 
4. **Creates video output** showing ground truth vs predicted poses in real-time

## Quick Start

### Full Pipeline (Training + Testing + Video)
```bash
python complete_pipeline.py \
    --data_path /home/dberretta-iit.local/data/new_scarfGNN \
    --video_data_path /home/dberretta-iit.local/data/cam2_S1_Directions \
    --epochs 30 \
    --output_video my_pose_prediction.mp4
```

### Skip Training (Use Existing Model)
```bash
python complete_pipeline.py \
    --skip_training \
    --ckpt_path /path/to/checkpoint.ckpt \
    --video_data_path /home/dberretta-iit.local/data/cam2_S1_Directions \
    --output_video test_video.mp4
```

### Training Only (No Video Generation)
```bash
python complete_pipeline.py \
    --training_only \
    --data_path /home/dberretta-iit.local/data/new_scarfGNN \
    --epochs 50 \
    --batch_size 128
```

## Command Line Arguments

### Data Paths
- `--data_path`: Path to training dataset (default: `/home/dberretta-iit.local/data/new_scarfGNN`)
- `--video_data_path`: Path to test data for video generation

### Pipeline Control
- `--skip_training`: Skip training and use existing checkpoint
- `--training_only`: Only perform training, skip testing
- `--ckpt_path`: Path to existing checkpoint (required if `--skip_training`)

### Model Configuration
- `--arch`: Model architecture (`single_weight` or `two_weights`, default: `single_weight`)
- `--epochs`: Number of training epochs (default: 30)
- `--batch_size`: Training batch size (default: 64)
- `--learning_rate`: Learning rate (default: 0.01)

### Video Configuration
- `--video_duration`: Duration of output video in seconds (default: 10.0)
- `--video_fps`: Video frame rate (default: 30)
- `--output_video`: Output video filename (default: `pose_prediction_video.mp4`)

### Other Options
- `--experiment_name`: Experiment name for logging (default: `complete_pipeline`)

## Data Structure Requirements

### Training Data
```
/data/path/
├── raw/
│   ├── session1_action1/
│   │   ├── ch0dvs/data.log              # Event data
│   │   └── ch0GT50Hzskeleton/data.log   # Ground truth poses
│   ├── session2_action2/
│   │   ├── ch0dvs/data.log
│   │   └── ch0GT50Hzskeleton/data.log
│   └── ...
└── processed/                           # Auto-generated .pt files
```

### Video Test Data
```
/video/data/path/
├── ch0dvs/data.log                      # Event data for video
└── ch0GT50Hzskeleton/data.log           # Ground truth poses for comparison
```

## Output Files

### Training Outputs
- `lightning_logs/experiment_name/version_X/checkpoints/`: Model checkpoints
- `lightning_logs/experiment_name/version_X/`: TensorBoard logs

### Video Output
- `pose_prediction_video.mp4` (or custom filename): Video showing:
  - SCARF event surface (background)
  - Graph nodes and edges (blue/cyan)
  - Ground truth pose skeleton (green)
  - Predicted pose skeleton (red)
  - Timestamp and legend

## Performance Metrics

The script reports:
- **PCK (Probability of Correct Keypoint)**: Percentage of joints within threshold distance
- **MPJPE (Mean Per Joint Position Error)**: Average pixel error per joint

## Example Workflows

### 1. Rapid Prototyping (Fast Test)
```bash
# Quick 5-epoch training + short video
python complete_pipeline.py \
    --data_path /your/data \
    --video_data_path /your/video/data \
    --epochs 5 \
    --video_duration 5.0 \
    --experiment_name quick_test
```

### 2. Full Production Training
```bash
# Complete training with validation
python complete_pipeline.py \
    --data_path /your/training/data \
    --video_data_path /your/test/data \
    --epochs 50 \
    --batch_size 128 \
    --learning_rate 0.005 \
    --video_duration 30.0 \
    --experiment_name production_model
```

### 3. Model Comparison
```bash
# Test existing model on new data
python complete_pipeline.py \
    --skip_training \
    --ckpt_path /path/to/best_model.ckpt \
    --video_data_path /path/to/new/test/data \
    --output_video comparison_video.mp4
```

## Troubleshooting

### Common Issues

1. **"Checkpoint not found"**: Ensure `--ckpt_path` points to a valid `.ckpt` file
2. **"Data path not found"**: Verify data paths exist and have correct structure
3. **Out of memory**: Reduce `--batch_size` (try 32 or 16)
4. **SplineConv warnings**: Normal on CPU, use GPU for better performance

### Performance Tips

- Use GPU for faster training: Script automatically detects CUDA
- Increase `batch_size` if you have enough memory
- Use `single_weight` architecture for more stable training
- Start with shorter videos (`--video_duration 5.0`) for testing

## Advanced Usage

### Custom SCARF Parameters
Edit the `create_config()` function in `complete_pipeline.py`:

```python
'scarf_params': {
    'rf_size': 14,     # Receptive field size
    'alpha': 1.0,      # Temporal decay
    'C': 0.3,          # Contrast threshold  
    'res': (640, 480), # Resolution
    'dt': 0.01         # Time step
}
```

### Custom Model Architecture
Modify the `hidden` parameter in config:

```python
'hidden': [64, 128, 256, 128, 64]  # Deeper network
```

## Integration with Existing Workflows

This script is designed to work alongside:
- `extract_predictions_visualize.py`: For detailed static visualizations
- `improved_train.py`: For training-focused workflows  
- `realtime_predict.py`: For live inference

The checkpoints created by `complete_pipeline.py` are compatible with all other scripts in the project.
