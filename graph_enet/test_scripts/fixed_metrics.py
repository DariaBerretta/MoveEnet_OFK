import numpy as np
import torch


def pck_error(input_tensor, target_tensor, threshold, multipler=0.6):
    """
    Calculate the Percentage of Correct Keypoints (PCK) error between target and prediction tensors.

    Args:
    - input_tensor (torch.Tensor): Predicted tensor of shape (batch_size, num_joints * 2).
    - target_tensor (torch.Tensor): Ground truth tensor of shape (batch_size, num_joints * 2).
    - threshold (torch.Tensor): Threshold distance tensor of shape (batch_size,) or (batch_size, 1).

    Returns:
    - pck (torch.Tensor): PCK error tensor (scalar).
    """
    # Check if the tensors have the same size
    if input_tensor.size() != target_tensor.size():
        raise ValueError(f"Input and target tensors must have the same size. Input size: {input_tensor.size()}, target size: {target_tensor.size()}")

    # Get batch size and number of joints
    batch_size = input_tensor.size(0)
    num_joints = input_tensor.size(1) // 2
    
    # Reshape tensors to (batch_size, num_joints, 2)
    input_reshaped = input_tensor.view(batch_size, num_joints, 2)
    target_reshaped = target_tensor.view(batch_size, num_joints, 2)

    # Calculate Euclidean distance between corresponding joints
    squared_diff = torch.pow(input_reshaped - target_reshaped, 2)
    euclidean_distance = torch.sqrt(torch.sum(squared_diff, dim=2))  # Shape: (batch_size, num_joints)
    
    # Ensure threshold has the correct shape for broadcasting
    if threshold.dim() == 1:
        threshold = threshold.unsqueeze(1)  # Shape: (batch_size, 1)
    
    # Apply threshold and multiplier
    threshold_expanded = threshold * multipler  # Shape: (batch_size, 1)
    
    # Calculate PCK by counting the number of keypoints within the threshold
    correct_keypoints = (euclidean_distance <= threshold_expanded).float()  # Shape: (batch_size, num_joints)
    
    # Calculate PCK percentage
    pck_per_sample = torch.mean(correct_keypoints, dim=1) * 100.0  # Shape: (batch_size,)
    pck = torch.mean(pck_per_sample)  # Scalar
    
    return pck


def mpjpe_error(input_tensor, target_tensor):
    """
    Calculate the mean per joint position error (MPJPE) between target and prediction tensors.

    Args:
    - input_tensor (torch.Tensor): Predicted tensor of shape (batch_size, num_joints * 2).
    - target_tensor (torch.Tensor): Ground truth tensor of shape (batch_size, num_joints * 2).

    Returns:
    - mpjpe (torch.Tensor): MPJPE error tensor (scalar).
    """
    # Check if the tensors have the same size
    if input_tensor.size() != target_tensor.size():
        raise ValueError(f"Input and target tensors must have the same size. Input size: {input_tensor.size()}, target size: {target_tensor.size()}")

    # Get batch size and number of joints
    batch_size = input_tensor.size(0)
    num_joints = input_tensor.size(1) // 2
    
    # Reshape tensors to (batch_size, num_joints, 2)
    input_reshaped = input_tensor.view(batch_size, num_joints, 2)
    target_reshaped = target_tensor.view(batch_size, num_joints, 2)

    # Calculate Euclidean distance between corresponding joints
    squared_diff = torch.pow(input_reshaped - target_reshaped, 2)
    euclidean_distance = torch.sqrt(torch.sum(squared_diff, dim=2))  # Shape: (batch_size, num_joints)

    # Calculate MPJPE by taking the mean across all joints and samples
    mpjpe = torch.mean(euclidean_distance)
    
    return mpjpe
