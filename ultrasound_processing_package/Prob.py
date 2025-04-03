import ultrasound_processing_package as upp
import matplotlib.pyplot as plt

image_path = "ultrasound_processing_package/data/sample_curved_cropped_01.png"

# Íves kép átalakítása
flat_image = upp.transform_curved_to_flat(image_path)

# Megjelenítés
plt.imshow(flat_image, cmap='gray')
plt.title("Transformed Curved Image")
plt.axis('off')
plt.show()
