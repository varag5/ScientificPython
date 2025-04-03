import ultrasound_processing_package as upp
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

image_path = "C:\\Users\\buvr_\\Documents\\BUVR 2025.1\\Korea\\SouthKorea2025\\data\\sample_curved_cropped_01.png"

# Íves kép átalakítása
flat_image = upp.transform_curved_to_flat(image_path)

# Megjelenítés
plt.imshow(flat_image, cmap='gray')
plt.title("Transformed Curved Image")
plt.axis('off')
plt.show()

# Kontúrkeresés és zaj eltávolítás
image, threshold_value, smooth_mask, masked_image = upp.remove_top_noise_and_keep_first_white(flat_image)
print("ok")
plt.imshow(masked_image, cmap='gray')
plt.show()

# Kép kimentése
Image.fromarray(np.uint8(masked_image)).save("C:\\Users\\buvr_\\Documents\\BUVR 2025.1\\Korea\\SouthKorea2025\\output\\masked_image.png")
