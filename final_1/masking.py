import cv2
import numpy as np

def mask(image, top_percent=0.93, top_margin=5, apply_closing=True):

    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / float(cdf[-1])

    threshold_idx = np.where(cdf_normalized >= top_percent)[0]
    threshold_value = threshold_idx[0] if threshold_idx.size > 0 else 255

    mask = (image >= threshold_value).astype(np.uint8) * 255

    if apply_closing:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for i in range(1, num_labels):
        if stats[i][1] < top_margin:
            mask[labels == i] = 0

    height, width = mask.shape
    contour_mask = np.zeros_like(mask)
    for col in range(width):
        white_indices = np.where(mask[:, col] == 255)[0]
        if white_indices.size > 0:
            contour_mask[white_indices[0], col] = 255

    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated_mask = cv2.dilate(contour_mask, dilation_kernel, iterations=1)
    smooth_mask = cv2.GaussianBlur(dilated_mask, (5,5), 0)
    smooth_mask[dilated_mask == 255] = 255
    smooth_mask_normalized = smooth_mask.astype(np.float32) / 255.0
    masked_image = (image.astype(np.float32) * smooth_mask_normalized).astype(np.uint8)

    return masked_image
