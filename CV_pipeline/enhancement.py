import cv2
import numpy as np

def enhance_image(img):

    #Grayscale black and whit img for ocr
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #median blur
    bg = cv2.medianBlur(gray, 21)
    norm = cv2.divide(gray, bg, scale=255)

    # (helps shadows slightly) i tried
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    norm = clahe.apply(gray)

    # 3. Sharpening kernel (مسروق من السكاشن لما فقدن الامل)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(norm, -1, kernel)


    return sharp