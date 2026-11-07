import cv2
import os
from enhancement import enhance_image #imports from our projects
from segmentation import (
    find_document_contour,
    four_point_transform,
    segment_for_ocr,
    contour_folder,
    warped_folder,
    enhanced_folder,
    segmented_folder,
    edges_folder
)

from ocr import perform_ocr


# =========================================================
# OUTPUT FOLDER FOR OCR RESULTS
# CHANGED: now saves inside project folder instead of colab 

ocr_folder = r"C:\Users\Omar\Desktop\omar\cv\ocr" #change path

os.makedirs(ocr_folder, exist_ok=True)


# =========================================================
# FULL PROCESSING PIPELINE
# =========================================================

def process_document(image_path, lang="eng", visualize=True):

    """
    Run full document scanner pipeline
    """

    filename = os.path.basename(image_path)

    name_no_ext = os.path.splitext(filename)[0]

    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}")

    #LOAD IMAGE

    image = cv2.imread(image_path)

    if image is None:
        print(f"ERROR: Could not read {image_path}")
        return None

    original = image.copy()

    #EDGE DETECTION aND CONTOUR DETECTION
    print("Step 1: Edge detection...")

    document_contour, edges = find_document_contour(image)

    # Save edges image
    cv2.imwrite(os.path.join(edges_folder, filename), edges)

    if document_contour is None:

        print(f"No document found in {filename}")
        return None

    # Draw contour visualization
    contour_img = original.copy()

    cv2.drawContours(
        contour_img,
        [document_contour],
        -1,
        (0, 255, 0),
        3
    )

    cv2.imwrite(
        os.path.join(contour_folder, filename),
        contour_img
    )

    #PERSPECTIVE TRANSFORMAtioN

    print("Step 2: Perspective transform...")

    warped = four_point_transform(
        image,
        document_contour.reshape(4, 2)
    )

    # Make document vertical
    if warped.shape[1] > warped.shape[0]:

        warped = cv2.rotate(
            warped,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )

    # Dataset-specific correction
    warped = cv2.rotate(
        warped,
        cv2.ROTATE_180
    )

    # Save warped image
    cv2.imwrite(
        os.path.join(warped_folder, filename),
        warped
    )

    # STEP 3: IMAGE ENHANCEMENT

    print("Step 3: Enhancement...")

    enhanced = enhance_image(warped)

    cv2.imwrite(
        os.path.join(enhanced_folder, filename),
        enhanced
    )

    #SEGMENTATION

    print("Step 4: Segmentation...")

    segmented = segment_for_ocr(enhanced)

    cv2.imwrite(
        os.path.join(segmented_folder, filename),
        segmented
    )

    # OCR
    print("Step 5: OCR...")

    text, details, annotated = perform_ocr(
        segmented,
        lang=lang
    )

    # Save OCR annotated image
    cv2.imwrite(
        os.path.join(ocr_folder, filename),
        annotated
    )

    # Save extracted text
    txt_path = os.path.join(
        ocr_folder,
        name_no_ext + ".txt"
    )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved OCR text -> {txt_path}")

    print("-" * 60)

    return text


# SINGLE IMAGE TESTING

#test_image = r"C:\Users\Omar\Desktop\omar\cv\Captured_Images\Samsung_phone\Images\M_Img_Android_D1_L2_r35_a-5_b10.jpg" #change path as needed

#extracted_text = process_document(
#    test_image,
#    lang="eng",
#    visualize=False
#)


# PROCESS ENTIRE DATASET TTHIS IS PREFERED SO USE THIS 

#change path as needed
#=====================================
input_folder = r"C:\Users\Omar\Desktop\omar\cv\Captured_Images\Samsung_phone\Images" 

valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

for filename in os.listdir(input_folder):

    if filename.lower().endswith(valid_ext):

        image_path = os.path.join(input_folder, filename)

        process_document(
            image_path,
            lang="eng",
            visualize=False
        )

print("Processing completed successfully.")