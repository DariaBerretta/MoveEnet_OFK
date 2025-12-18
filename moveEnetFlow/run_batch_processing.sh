#!/bin/bash
# Batch processing script for MoveEnet Flow
# Processes all data.log files in subdirectories and creates corresponding CSV files

# Usage: ./run_batch_processing.sh [options]

# Default parameters
DATA_ROOT="/data/new_scarfGNN_full/raw"
OUTPUT_DIR="/home/moveEnetFlow/csv_file"
DETECTION_FREQ=10

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --f_det)
      DETECTION_FREQ="$2"
      shift 2
      ;;
    --help)
      echo "Batch processing script for MoveEnet Flow"
      echo ""
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --data_root <path>    Root directory containing subdirectories with data.log files (default: /data/new_scarfGNN_full/raw)"
      echo "  --output_dir <path>   Directory to save output CSV files (default: /home/moveEnetFlow/csv_file)"
      echo "  --f_det <int>         Detection frequency in Hz (default: 10)"
      echo "  --help                Show this help message"
      echo ""
      echo "Example:"
      echo "  $0 --data_root /data/my_dataset/raw --output_dir /home/results --f_det 15"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

echo "Starting batch processing..."
echo "Data root: $DATA_ROOT"
echo "Output directory: $OUTPUT_DIR"
echo "Detection frequency: ${DETECTION_FREQ} Hz"
echo ""

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Run the batch processing
./moveEnet_flow_batch --data_root "$DATA_ROOT" --output_csv_dir "$OUTPUT_DIR" --f_det $DETECTION_FREQ

echo ""
echo "Batch processing completed!"
echo "CSV files saved in: $OUTPUT_DIR"