from transform_back_project import interp_img
from transform_project import transform
from masking import mask
import numpy as np
import matplotlib.pyplot as plt


path = "input\\cropped_frame_idx_50.png"
alpha_deg = 35
resolution = 50





alpha_rad = alpha_deg/180*np.pi

transformed_image, depth, offset = transform(path, alpha_deg, resolution)
print(depth, offset)
plt.imshow(transformed_image, cmap="gray")
plt.show()

masked_image = mask(transformed_image)
plt.imshow(masked_image, cmap="gray")
plt.show()

final_image = interp_img(masked_image, depth, alpha_rad, offset)
plt.imshow(final_image, cmap="gray")
plt.show()
