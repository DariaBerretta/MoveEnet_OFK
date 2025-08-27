import numpy as np
import torch


# def pck_error(input_tensor, target_tensor, threshold, multipler=0.6):
#     """
#     Calculate the Percentage of Correct Keypoints (PCK) error between target and prediction tensors.

#     Args:
#     - input_tensor (torch.Tensor): Predicted tensor of shape (batch_size, num_joints * 2).
#     - target_tensor (torch.Tensor): Ground truth tensor of shape (batch_size, num_joints * 2).
#     - threshold (torch.Tensor): Threshold distance tensor of shape (batch_size,) or (batch_size, 1).
#     - multipler (float): Multiplier for the threshold.

#     Returns:
#     - pck (torch.Tensor): PCK error tensor (scalar).
#     """
#     # Check if the tensors have the same size
#     if input_tensor.size() != target_tensor.size():
#         raise ValueError(f"Input and target tensors must have the same size. Input size: {input_tensor.size()}, target size: {target_tensor.size()}")

#     # Get batch size and number of joints
#     batch_size = input_tensor.size(0)
#     num_joints = input_tensor.size(1) // 2
    
#     # Reshape tensors to (batch_size, num_joints, 2)
#     input_reshaped = input_tensor.view(batch_size, num_joints, 2)
#     target_reshaped = target_tensor.view(batch_size, num_joints, 2)

#     # Calculate Euclidean distance between corresponding joints
#     squared_diff = torch.pow(input_reshaped - target_reshaped, 2)
#     euclidean_distance = torch.sqrt(torch.sum(squared_diff, dim=2))  # Shape: (batch_size, num_joints)
    
#     # Ensure threshold has the correct shape for broadcasting
#     if threshold.dim() == 1:
#         threshold = threshold.unsqueeze(1)  # Shape: (batch_size, 1)
    
#     # Apply threshold and multiplier
#     threshold_expanded = threshold * multipler  # Shape: (batch_size, 1)
    
#     # Calculate PCK by counting the number of keypoints within the threshold
#     correct_keypoints = (euclidean_distance <= threshold_expanded).float()  # Shape: (batch_size, num_joints)
    
#     # Calculate PCK percentage
#     pck_per_sample = torch.mean(correct_keypoints, dim=1) * 100.0  # Shape: (batch_size,)
#     pck = torch.mean(pck_per_sample)  # Scalar
    
#     return pck


# def mpjpe_error(input_tensor, target_tensor):
#     """
#     Calculate the mean per joint position error (MPJPE) between target and prediction tensors.

#     Args:
#     - input_tensor (torch.Tensor): Predicted tensor of shape (batch_size, num_joints * 2).
#     - target_tensor (torch.Tensor): Ground truth tensor of shape (batch_size, num_joints * 2).

#     Returns:
#     - mpjpe (torch.Tensor): MPJPE error tensor (scalar).
#     """
#     # Check if the tensors have the same size
#     if input_tensor.size() != target_tensor.size():
#         raise ValueError(f"Input and target tensors must have the same size. Input size: {input_tensor.size()}, target size: {target_tensor.size()}")

#     # Get batch size and number of joints
#     batch_size = input_tensor.size(0)
#     num_joints = input_tensor.size(1) // 2
    
#     # Reshape tensors to (batch_size, num_joints, 2)
#     input_reshaped = input_tensor.view(batch_size, num_joints, 2)
#     target_reshaped = target_tensor.view(batch_size, num_joints, 2)

#     # Calculate Euclidean distance between corresponding joints
#     squared_diff = torch.pow(input_reshaped - target_reshaped, 2)
#     euclidean_distance = torch.sqrt(torch.sum(squared_diff, dim=2))  # Shape: (batch_size, num_joints)

#     # Calculate MPJPE by taking the mean across all joints and samples
#     mpjpe = torch.mean(euclidean_distance)
    
#     return mpjpe


def pck_error(predicted_xy_flat, target_xy_flat, threshold, multipler=0.6):
    """
    Percentage of Correct Keypoints (PCK).
    Global reduction: counts correct keypoints over the whole batch.

    Args:
        predicted_xy_flat: (B, J*2) predicted 2D keypoints [x1,y1,...,xJ,yJ]
        target_xy_flat:    (B, J*2) ground-truth 2D keypoints
        threshold:         float, (B,), (B,1), or (B,J) distance thresholds
        multipler:         scalar multiplier applied to threshold
    Returns:
        Scalar tensor with PCK in percent.
    """
    if predicted_xy_flat.size() != target_xy_flat.size():
        raise ValueError(f"Size mismatch: {predicted_xy_flat.size()} vs {target_xy_flat.size()}")

    batch_size = predicted_xy_flat.size(0)
    num_joints = predicted_xy_flat.size(1) // 2

    # Reshape flat vectors to (B, J, 2)
    predicted_xy = predicted_xy_flat.view(batch_size, num_joints, 2)
    target_xy    = target_xy_flat.view(batch_size, num_joints, 2)

    # Euclidean distance per joint: (B, J)
    diff_xy = predicted_xy - target_xy
    distances_per_joint = torch.linalg.norm(diff_xy, dim=2)

    # Make threshold tensor device/dtype-safe and broadcastable
    threshold_tensor = torch.as_tensor(threshold, dtype=distances_per_joint.dtype, device=distances_per_joint.device)
    if threshold_tensor.ndim == 0:                # scalar → all samples, all joints
        threshold_broadcast = threshold_tensor.view(1, 1)
    elif threshold_tensor.ndim == 1:              # (B,) → per-sample
        if threshold_tensor.shape[0] != batch_size:
            raise ValueError("Threshold length must equal batch size.")
        threshold_broadcast = threshold_tensor.view(batch_size, 1)
    elif threshold_tensor.ndim == 2:              # (B,1) or (B,J)
        if threshold_tensor.shape[0] != batch_size:
            raise ValueError("Threshold first dim must equal batch size.")
        if threshold_tensor.shape[1] not in (1, num_joints):
            raise ValueError("Threshold second dim must be 1 or num_joints.")
        threshold_broadcast = threshold_tensor
    else:
        raise ValueError("Threshold must be scalar, (B,), (B,1), or (B,J).")

    # Correct keypoints boolean mask
    within_threshold = distances_per_joint <= (multipler * threshold_broadcast)  # (B, J)

    # Global reduction over all keypoints in the batch
    num_correct_keypoints = within_threshold.sum().float()
    total_keypoints = within_threshold.numel()
    pck_percent = num_correct_keypoints / total_keypoints * 100.0
    return pck_percent


def mpjpe_error(predicted_xy_flat, target_xy_flat):
    """
    Mean Per-Joint Position Error (MPJPE).
    Global mean over all joints and samples.

    Args:
        predicted_xy_flat: (B, J*2) predicted 2D keypoints
        target_xy_flat:    (B, J*2) ground-truth 2D keypoints
    Returns:
        Scalar tensor with mean Euclidean error.
    """
    if predicted_xy_flat.size() != target_xy_flat.size():
        raise ValueError(f"Size mismatch: {predicted_xy_flat.size()} vs {target_xy_flat.size()}")

    batch_size = predicted_xy_flat.size(0)
    num_joints = predicted_xy_flat.size(1) // 2

    predicted_xy = predicted_xy_flat.view(batch_size, num_joints, 2)
    target_xy    = target_xy_flat.view(batch_size, num_joints, 2)

    distances_per_joint = torch.linalg.norm(predicted_xy - target_xy, dim=2)  # (B, J)
    mean_error = distances_per_joint.mean()
    return mean_error