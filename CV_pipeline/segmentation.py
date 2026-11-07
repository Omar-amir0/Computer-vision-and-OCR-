import cv2
import os
import numpy as np


# PATHS CHANGED: all outputs now save local instead of colab
# =========================================================

# Dataset folder
input_folder = r"C:\Users\Omar\Desktop\omar\cv\Captured_Images\Samsung_phone\Images" #change path

# Base output folder
base_output = r"C:\Users\Omar\Desktop\omar\cv" #change path

# Output folders
contour_folder = os.path.join(base_output, "contours")

warped_folder = os.path.join(base_output, "warped")

enhanced_folder = os.path.join(base_output, "enhanced")

segmented_folder = os.path.join(base_output, "segmented")

edges_folder = os.path.join(base_output, "edges")

# Create folders automatically
os.makedirs(contour_folder, exist_ok=True)
os.makedirs(warped_folder, exist_ok=True)
os.makedirs(enhanced_folder, exist_ok=True)
os.makedirs(segmented_folder, exist_ok=True)
os.makedirs(edges_folder, exist_ok=True)


# =========================================================
# ORDER POINTS
# =========================================================

def order_points(pts):

    pts = np.array(pts, dtype="float32")

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)

    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


# =========================================================
# PERSPECTIVE TRANSFORM
# =========================================================

def four_point_transform(image, pts):

    rect = order_points(pts)

    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)

    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(
        image,
        M,
        (maxWidth, maxHeight)
    )

    return warped


# =========================================================
# DOCUMENT CONTOUR DETECTION
# =========================================================

def find_document_contour(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 30, 120)

    # Strengthen edges
    kernel = np.ones((5, 5), np.uint8)

    edges = cv2.dilate(edges, kernel, iterations=1)

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        edges.copy(),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    document_contour = None

    image_area = image.shape[0] * image.shape[1]

    # Try exact 4-point contour
    for c in contours[:10]:

        area = cv2.contourArea(c)

        if area < 0.10 * image_area:
            continue

        peri = cv2.arcLength(c, True)

        approx = cv2.approxPolyDP(
            c,
            0.02 * peri,
            True
        )

        if len(approx) == 4:

            document_contour = approx

            print("Used approxPolyDP contour")

            break

    # Fallback
    if document_contour is None:

        for c in contours[:10]:

            area = cv2.contourArea(c)

            if area < 0.10 * image_area:
                continue

            rect = cv2.minAreaRect(c)

            box = cv2.boxPoints(rect)

            box = np.intp(box)

            document_contour = box.reshape(4, 1, 2)

            print("Used minAreaRect fallback")

            break

    return document_contour, edges


# =========================================================
# SEGMENTATION FOR OCR
# =========================================================

def segment_for_ocr(gray_img):

    blurred = cv2.GaussianBlur(
        gray_img,
        (3, 3),
        0
    )

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        15
    )

    # Ensure white background
    if np.mean(binary) < 127:

        binary = 255 - binary

    # Invert image
    inv = 255 - binary

    # Connected component cleanup
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inv,
        connectivity=8
    )

    cleaned = np.zeros_like(inv)

    for i in range(1, num_labels):

        x, y, w, h, area = stats[i]

        if area >= 15 and w >= 2 and h >= 2:

            cleaned[labels == i] = 255

    final = 255 - cleaned

    return final