import cv2
import numpy as np

def remove_top_noise_and_keep_first_white(image_input, top_percent=0.90, top_margin=5, apply_closing=True):
    """
    image_input: lehet fájlútvonal (str) vagy már betöltött kép (np.ndarray)
    """

    # Input típusa alapján eldöntjük, mit csináljunk
    if isinstance(image_input, str):
        image = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Failed to load the image: {image_input}")
    elif isinstance(image_input, np.ndarray):
        image = image_input
    else:
        raise ValueError("Az image_input legyen fájlútvonal (str) vagy betöltött kép (np.ndarray)")

    # Histogram + küszöb
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / float(cdf[-1])
    threshold_idx = np.where(cdf_normalized >= top_percent)[0]
    threshold_value = threshold_idx[0] if threshold_idx.size > 0 else 255

    # Maszk készítés
    mask = (image >= threshold_value).astype(np.uint8) * 255

    if apply_closing:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Komponensek szűrése
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i][1] < top_margin:
            mask[labels == i] = 0

    # Kontúrvonal kiemelése
    height, width = mask.shape
    contour_mask = np.zeros_like(mask)
    for col in range(width):
        white_indices = np.where(mask[:, col] == 255)[0]
        if white_indices.size > 0:
            contour_mask[white_indices[0], col] = 255

    # Duzzasztás és simítás
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated_mask = cv2.dilate(contour_mask, dilation_kernel, iterations=1)
    smooth_mask = cv2.GaussianBlur(dilated_mask, (5,5), 0)
    smooth_mask[dilated_mask == 255] = 255
    smooth_mask_normalized = smooth_mask.astype(np.float32) / 255.0

    # Eredmény maszkolás
    masked_image = (image.astype(np.float32) * smooth_mask_normalized).astype(np.uint8)

    return image, threshold_value, smooth_mask, masked_image
