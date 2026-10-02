def map_roi_to_feature_map(roi, spatial_scale):
    """
    Maps the ROI coordinates from the original image space to the feature map space.

    Args:
        roi (tuple): A tuple of (x1, y1, x2, y2) representing the ROI coordinates in the original image.
        spatial_scale (float): The scaling factor between the original image and the feature map.

    Returns:
        tuple: A tuple of (x1', y1', x2', y2') representing the ROI coordinates in the feature map space.
    """
    x1, y1, x2, y2 = roi
    x1 = int(x1 * spatial_scale)
    y1 = int(y1 * spatial_scale)
    x2 = int(x2 * spatial_scale)
    y2 = int(y2 * spatial_scale)
    return (x1, y1, x2, y2)
