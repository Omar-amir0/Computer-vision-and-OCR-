import pytesseract
import cv2
import os                     
from PIL import Image         #معرفش مين ديه متسألش

#to isntall tesseract u need to 
# !pip install pytesseract 
#use this link to also install in the pc too u need both then configure the path below
#https://github.com/UB-Mannheim/tesseract/wiki 
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Omar\AppData\Local\Programs\Tesseract-OCR\tesseract.exe" #change to ur tesseract path

def perform_ocr(segmented_img, lang="eng"):

    pil_img = Image.fromarray(segmented_img)

    custom_config = r"--oem 3 --psm 6"

    text = pytesseract.image_to_string(
        pil_img,
        lang=lang,
        config=custom_config
    )

    details = pytesseract.image_to_data(
        pil_img,
        lang=lang,
        config=custom_config,
        output_type=pytesseract.Output.DICT
    )

    annotated = cv2.cvtColor(
        segmented_img.copy(),
        cv2.COLOR_GRAY2BGR
    )

    n_boxes = len(details['text'])

    for i in range(n_boxes):

        conf = int(details['conf'][i])

        word = details['text'][i].strip()

        if conf > 30 and word:

            x = details['left'][i]
            y = details['top'][i]
            w = details['width'][i]
            h = details['height'][i]

            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                (0, 200, 0),
                2
            )

            cv2.putText(
                annotated,
                f"{conf}%",
                (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 0, 0),
                1
            )

    return text, details, annotated


# ======================================================
# SINGLE IMAGE TEST
#removed Colab path now local path

#test_image = r"C:\Users\Omar\Desktop\omar\cv\Captured_Images\Samsung_phone\Images\IMAGE.jpg"

#extracted_text = process_document(
    test_image,
    lang="eng",
    visualize=True
#)