import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
image = cv2.imread("brain_mri.jpg", 0)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(image, (5,5), 0)

# Apply Thresholding
_, segmented = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)

# Display results
plt.figure()
plt.imshow(segmented, cmap='gray')
plt.title("Segmented Image")
plt.axis("off")
plt.show()
