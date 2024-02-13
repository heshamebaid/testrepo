import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img_path = "E:\\becca-pixels.jpg" 
img = cv2.imread(img_path)

# Define the new size for the image
new_size = (800, 600)

# Resize using nearest neighbor interpolation
nn_img = cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)

# Resize using bilinear interpolation
bl_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

# Resize using spline interpolation
spl_img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)

# Resize using bicubic interpolation
bic_img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)

# Display the original and resized images side by side
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title('Original')
axs[0, 1].imshow(cv2.cvtColor(nn_img, cv2.COLOR_BGR2RGB))
axs[0, 1].set_title('Nearest Neighbor')
axs[1, 0].imshow(cv2.cvtColor(bl_img, cv2.COLOR_BGR2RGB))
axs[1, 0].set_title('Bilinear')
axs[1, 1].imshow(cv2.cvtColor(bic_img, cv2.COLOR_BGR2RGB))
axs[1, 1].set_title('Bicubic')

plt.show()
