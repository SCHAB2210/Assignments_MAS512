import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Read the image from file and convert it to grayscale
image = cv2.imread('porche_panamera_blue.jpg')
image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
plt.imshow(image_grayscale, cmap='gray')
plt.title('Grayscale Image')
plt.show()

# Add Gaussian noise to the grayscale image
mean = 0
stddev = 25
noise = np.random.normal(mean, stddev, image_grayscale.shape)
noisy_image = image_grayscale + noise

# Clip pixel values to be between 0 and 255
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

# Display the noisy image
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')
plt.show()

# Apply Gaussian blur to remove noise (Standard method)
kernel_size = (5, 5)  # (a, b) kernel size
noiseremove_image_standard = cv2.GaussianBlur(noisy_image, kernel_size, 0)

# Display the Gaussian blur result
plt.imshow(noiseremove_image_standard, cmap='gray')
plt.title('Gaussian Blur Noise Removal')
plt.show()

# Custom kernel blur function
def custom_kernel_blur(image, kernel_size):
    kernel = np.ones(kernel_size, np.float32) / (kernel_size[0] * kernel_size[1])
    return cv2.filter2D(image, -1, kernel)

# Apply custom kernel blur
noiseremove_image_kernel = custom_kernel_blur(noisy_image, kernel_size)

# Display the custom kernel blur result
plt.imshow(noiseremove_image_kernel, cmap='gray')
plt.title('Custom Kernel Noise Removal')
plt.show()

# Compute SSIM for Gaussian blur and custom kernel methods
ssim_standard = ssim(image_grayscale, noiseremove_image_standard)
ssim_kernel = ssim(image_grayscale, noiseremove_image_kernel)

print(f'SSIM (Gaussian Blur): {ssim_standard:.4f}')
print(f'SSIM (Custom Kernel): {ssim_kernel:.4f}')

# Plot the comparison
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(noiseremove_image_standard, cmap='gray')
ax[0].set_title(f'SSIM Gaussian: {ssim_standard:.4f}')

ax[1].imshow(noiseremove_image_kernel, cmap='gray')
ax[1].set_title(f'SSIM Kernel: {ssim_kernel:.4f}')

plt.show()
