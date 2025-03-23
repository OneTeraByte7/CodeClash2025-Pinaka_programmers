import cv2

image = cv2.imread("foggy_image.jpg", 0)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
enhanced = clahe.apply(image)
cv2.imshow("Enhanced", enhanced)
cv2.waitKey(0)