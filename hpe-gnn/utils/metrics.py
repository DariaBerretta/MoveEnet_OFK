import numpy as np
import torch


def pck_error(input_tensor, target_tensor, threshold, multipler=0.6) -> torch.float:
    """
    Calculate the Percentage of Correct Keypoints (PCK) error between target and prediction tensors.

    Args:
    - input_tensor (torch.Tensor): Predicted tensor of shape (batch_size, num_joints * 2).
    - target_tensor (torch.Tensor): Ground truth tensor of shape (batch_size, num_joints * 2).
    - threshold (float): Threshold distance for a predicted keypoint to be considered correct.

    Returns:
    - pck (torch.Tensor): PCK error tensor of shape (batch_size,).
    """
    # Check if the tensors have the same size
    if input_tensor.size() != target_tensor.size():
        raise ValueError("Input and target tensors must have the same size\n input size:", input_tensor.size(), ' but target tensor size:', target_tensor.size())

    # Reshape the input and target tensors to (batch_size, num_joints, 2)
    input_reshaped = input_tensor.view(-1, 2)
    target_reshaped = target_tensor.view(-1, 2)
    num_samples = threshold.shape[0]

    # Calculate Euclidean distance between corresponding joints
    squared_diff = torch.pow(input_reshaped - target_reshaped, 2)
    euclidean_distance = torch.sqrt(torch.sum(squared_diff, dim=1)).reshape(num_samples,-1)
    # Calculate PCK by counting the number of keypoints within the threshold
    correct_keypoints = torch.sum(euclidean_distance <= torch.broadcast_to((threshold*multipler), euclidean_distance.shape)).float()
    total_keypoints = torch.tensor(input_reshaped.size()[0]).float()  # Total number of keypoints
    # Calculate PCK error
    pck = torch.mean((correct_keypoints / total_keypoints) * 100.0)
    return pck


def mpjpe_error(input_tensor, target_tensor) -> torch.float:
    """
    Calculate the mean per joint position error (MPJPE) between target and prediction tensors.

    Args:
    - input_tensor (torch.Tensor): Predicted tensor of shape (batch_size, num_joints * 2).
    - target_tensor (torch.Tensor): Ground truth tensor of shape (batch_size, num_joints * 2).

    Returns:
    - mpjpe (torch.Tensor): MPJPE error tensor of shape (batch_size,).
    """
    # Check if the tensors have the same size
    if input_tensor.size() != target_tensor.size():
        raise ValueError("Input and target tensors must have the same size")

    # Reshape the input and target tensors to (num_joints, 2)
    input_reshaped = input_tensor.view(-1, 2)
    target_reshaped = target_tensor.view(-1, 2)

    # Calculate Euclidean distance between corresponding joints
    squared_diff = torch.pow(input_reshaped - target_reshaped, 2)
    euclidean_distance = torch.sqrt(torch.sum(squared_diff, dim=1))

    # Calculate MPJPE by taking the mean along the joints axis
    mpjpe = torch.mean(euclidean_distance, dim=0)
    return mpjpe


