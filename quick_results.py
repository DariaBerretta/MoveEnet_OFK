#!/usr/bin/env python3
"""Quick visualization fix for the improved model results.
This script summarizes the results and provides a quick visual comparison
between the original and improved model performance."""

import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def quick_visualize():
    """Quick visualization fix for the improved model results."""
    
    # Load the results
    results_file = '/home/dberretta-iit.local/Documents/Repos/GraphEnet-v2/prediction_results/results_summary.json'
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("\n" + "="*60)
    print("IMPROVED MODEL PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Average PCK:  {results['mean_pck']:.2f}%")
    print(f"Average MPJPE: {results['mean_mpjpe']:.2f} pixels")
    print(f"Samples tested: {results['num_samples']}")
    print("="*60)
    
    # Try to load actual prediction data if available
    try:
        pred_file = '/home/dberretta-iit.local/Documents/Repos/GraphEnet-v2/prediction_results/results_summary.json'
        if os.path.exists(pred_file):
            with open(pred_file, 'r') as f:
                pred_data = json.load(f)
            
            print(f"\nFound {len(pred_data['samples_data'])} prediction samples")
            
            # Create a simple visualization directory
            viz_dir = '/home/dberretta-iit.local/Documents/Repos/GraphEnet-v2/improved_results/quick_viz'
            os.makedirs(viz_dir, exist_ok=True)
            
            # Visualize first 3 samples
            for i in range(min(3, len(pred_data['samples_data']))):
                sample = pred_data['samples_data'][i]
                
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                
                # Extract coordinates
                gt = np.array(sample['ground_truth']).reshape(-1, 2)
                pred = np.array(sample['prediction']).reshape(-1, 2)
                
                # Plot joints
                ax.scatter(gt[:, 0], gt[:, 1], c='blue', s=100, label='Ground Truth', alpha=0.8, marker='o')
                ax.scatter(pred[:, 0], pred[:, 1], c='red', s=100, label='Prediction', alpha=0.8, marker='x')
                
                # Add joint numbers
                for j in range(len(gt)):
                    ax.annotate(f'{j}', (gt[j, 0], gt[j, 1]), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8, color='blue')
                    ax.annotate(f'{j}', (pred[j, 0], pred[j, 1]), xytext=(-15, 5), 
                               textcoords='offset points', fontsize=8, color='red')
                
                ax.set_title(f'Sample {i+1} - PCK: {sample["pck"]:.2f}%, MPJPE: {sample["mpjpe"]:.2f}px')
                ax.set_xlabel('X coordinate (pixels)')
                ax.set_ylabel('Y coordinate (pixels)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.invert_yaxis()  # Match image coordinates
                
                plt.tight_layout()
                plt.savefig(f'{viz_dir}/sample_{i+1}.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"Sample {i+1}: PCK={sample['pck']:.2f}%, MPJPE={sample['mpjpe']:.2f}px")
            
            print(f"\nVisualization saved to: {viz_dir}/")
        else:
            print(f"\nPrediction data file not found: {pred_file}")
            
    except Exception as e:
        print(f"Could not load prediction data: {e}")
    
    print("\n" + "="*60)
    print("COMPARISON WITH ORIGINAL MODEL:")
    print("Original model: PCK ~53%, MPJPE ~99 pixels (mode collapse)")
    print(f"Improved model: PCK {results['mean_pck']:.1f}%, MPJPE {results['mean_mpjpe']:.1f} pixels")
    print(f"Improvement: +{results['mean_pck']-53:.1f}% PCK, -{99-results['mean_mpjpe']:.1f} pixels MPJPE")
    print("="*60)

if __name__ == "__main__":
    quick_visualize()
