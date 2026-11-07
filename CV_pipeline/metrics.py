
import os
import cv2
import numpy as np
import pytesseract
from PIL import Image

from segmentation import find_document_contour, four_point_transform
from enhancement import enhance_image
from ocr import perform_ocr



INPUT_FOLDER = r"C:\Users\Omar\Desktop\omar\cv\Captured_Images\Samsung_phone\Images"

SEGMENTED_FOLDER = r"C:\Users\Omar\Desktop\omar\cv\segmented"

GT_OCR_FOLDER = r"C:\Users\Omar\Desktop\omar\cv\Truth_texts" #handtyped ocr results

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Omar\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")




# IoU = TP / (TP + FP + FN)
#   TP = pixels both masks agree are text
#   FP = your pipeline said text, Tesseract disagrees
#   FN = Tesseract found text, pipeline missed it
# compares the tesseracr extracted images vs the ones in our segmented pipeline
#it is not very accurate due to different sizes and stuff but what matters is ocr accuracy more
def compute_iou(pred_mask, gt_mask):

    pred = (pred_mask > 127).astype(np.uint8)
    gt   = (gt_mask   > 127).astype(np.uint8)

    # resize gt to match pred if shapes differ
    if pred.shape != gt.shape:
        gt = cv2.resize(
            gt,
            (pred.shape[1], pred.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    TP = int(np.sum((pred == 1) & (gt == 1))) 
    FP = int(np.sum((pred == 1) & (gt == 0)))  
    FN = int(np.sum((pred == 0) & (gt == 1)))  

    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    return iou, TP, FP, FN


def evaluate_segmentation():
  #compares them with tesseract text boxes
    ious = []
    results = []

    seg_files = [f for f in os.listdir(SEGMENTED_FOLDER) #in segmented folder only
                 if f.lower().endswith(VALID_EXT)]

    print(f"  Found {len(seg_files)} images in segmented folder...\n")

    for filename in seg_files:

        # load the original image to build the ground truth mask
        image_path = os.path.join(INPUT_FOLDER, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"  Cannot find original image for {filename}")
            continue

        # warp the document flat like pipeline
        contour, _ = find_document_contour(image)
        if contour is None:
            print(f"  [SKIP] No contour for {filename}")
            continue

        warped = four_point_transform(image, contour.reshape(4, 2))
        if warped.shape[1] > warped.shape[0]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        warped = cv2.rotate(warped, cv2.ROTATE_180)

        # run Tesseract on the raw image
        # to get word  boxes ground truth automatically
        gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        details = pytesseract.image_to_data(
            Image.fromarray(gray_warped),
            config="--oem 3 --psm 6",
            output_type=pytesseract.Output.DICT
        )

        # start with a fully black canvas 
        h, w = gray_warped.shape
        gt_mask = np.zeros((h, w), dtype=np.uint8)

        # paint each detected word box white on the canvas
        # only use detections with confidence above 30%
        for i in range(len(details["text"])):
            conf = int(details["conf"][i])
            word = details["text"][i].strip()
            if conf > 30 and word:
                x  = details["left"][i]
                y  = details["top"][i]
                bw = details["width"][i]
                bh = details["height"][i]
                gt_mask[y : y + bh, x : x + bw] = 255  # white = text

        segmented = cv2.imread(
            os.path.join(SEGMENTED_FOLDER, filename),
            cv2.IMREAD_GRAYSCALE
        )

        if segmented is None:
            print(f"  [SKIP] Cannot read segmented image for {filename}")
            continue

        # compare the two masks pixel by pixel
        iou, TP, FP, FN = compute_iou(segmented, gt_mask)
        ious.append(iou)
        results.append({"file": filename, "iou": iou, "TP": TP, "FP": FP, "FN": FN})

        print(f"  {filename}")
        print(f"    IoU={iou:.4f}  TP={TP}  FP={FP}  FN={FN}")

    mean_iou = float(np.mean(ious)) if ious else 0.0
    return {"mean_iou": mean_iou, "per_image": results}


#OCR MATCHING ACCURACY (CER)
# CER = Character Error Rate
#
#   example: ground truth is "Hello World" (11 chars)
#            Tesseract output is "He1lo Wor1d"
#            2 characters are wrong → CER = 2/11 = 0.18
#
# OCR Matching Accuracy = 1 - CER

def levenshtein(a, b):

    m, n = len(a), len(b)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev               # characters match, no cost
            else:
                dp[j] = 1 + min(prev,      # substitution
                                dp[j],     # deletion
                                dp[j - 1]) # insertion
            prev = temp
    return dp[n]

    
def compute_cer(predicted_text, gt_text):
    
    #Character Error Rate — compares letter by letter.
    #Both texts are normalized first so newlines, extra spaces,
    #and indentation don't count as errors. AI added
    #the lower the better here 
    
    import re
    pred = re.sub(r'\s+', ' ', predicted_text).strip().lower()
    gt   = re.sub(r'\s+', ' ', gt_text).strip().lower()

    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0

    return levenshtein(list(pred), list(gt)) / len(gt)


def evaluate_ocr(): #reads.txt files

    cers  = []
    confs = []
    results = []

    # get all ground truth text files you typed
    gt_files = [f for f in os.listdir(GT_OCR_FOLDER)
                if f.lower().endswith(".txt")]

    if not gt_files:
        print("  [WARNING] No .txt files found in Truth_texts folder!")
        print(f"  Folder checked: {GT_OCR_FOLDER}")
        print("  Add your ground truth .txt files and re-run.")
        return {"mean_cer": 0.0, "mean_conf": 0.0, "per_image": []}

    print(f"  Found {len(gt_files)} ground truth text files...\n")

    for gt_filename in gt_files:

        # find the matching image by stem name has to be same name
        stem = os.path.splitext(gt_filename)[0]
        image_path = None

        for ext in VALID_EXT:
            candidate = os.path.join(INPUT_FOLDER, stem + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break

        if image_path is None:
            print(f"  [SKIP] No matching image found for {gt_filename}")
            continue

        # reads manually made text
        with open(os.path.join(GT_OCR_FOLDER, gt_filename), encoding="utf-8") as f:
            gt_text = f.read()

        image = cv2.imread(image_path)
        if image is None:
            print(f"  [SKIP] Cannot read image for {stem}")
            continue

        # run the full pipeline on this image
        contour, _ = find_document_contour(image)
        if contour is None:
            print(f"  [SKIP] No contour found for {stem}")
            continue

        warped = four_point_transform(image, contour.reshape(4, 2))
        if warped.shape[1] > warped.shape[0]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        warped = cv2.rotate(warped, cv2.ROTATE_180)

        enhanced  = enhance_image(warped)

        from segmentation import segment_for_ocr
        segmented = segment_for_ocr(enhanced)

        # get Tesseract's extracted text and per-word confidence scores
        pred_text, details, _ = perform_ocr(segmented, lang="eng")

        # compute CER between Tesseract output and ground truth
        cer = compute_cer(pred_text, gt_text)

        # mean Tesseract confidence across all detected words
        # Tesseract returns -1 for non-word rows
        valid_confs = [
            int(c) for c in details["conf"]
            if str(c).lstrip("-").isdigit() and int(c) > 0
        ]
        mean_conf = float(np.mean(valid_confs)) if valid_confs else 0.0

        cers.append(cer)
        confs.append(mean_conf)
        results.append({"file": stem, "cer": cer, "mean_conf": mean_conf})

        print(f"  {stem}")
        print(f"    CER={cer:.4f}  Tesseract confidence={mean_conf:.1f}%")

    return {
        "mean_cer":  float(np.mean(cers))  if cers  else 0.0,
        "mean_conf": float(np.mean(confs)) if confs else 0.0,
        "per_image": results
    }


# =========================================================
# MAIN




if __name__ == "__main__":

    # iou segmentation
    print("Segmentation IoU (automatic)")

    seg = evaluate_segmentation()

    print(f"\n  ── Summary ──────────────────────────────────")
    print(f"  Images evaluated : {len(seg['per_image'])}")
    print(f"  Mean IoU         : {seg['mean_iou']:.4f} "
          f"({seg['mean_iou'] * 100:.2f}%)")

    #OCR ACCURACY
    print("\n\nOCR Matching Accuracy\n\n")

    if not os.path.isdir(GT_OCR_FOLDER):
        print(f"  Ground truth folder not found: {GT_OCR_FOLDER}")
    else:
        ocr = evaluate_ocr()

        print(f"\n  ── Summary ──────────────────────────────────")
        print(f"  Images evaluated      : {len(ocr['per_image'])}")
        print(f"  Mean CER              : {ocr['mean_cer'] * 100:.2f}%") #the lower the better eror/total characters
        print(f"  Mean Tesseract conf   : {ocr['mean_conf']:.1f}%")
        print(f"  OCR Matching Accuracy : {(1 - ocr['mean_cer']) * 100:.2f}%  (= 1 - CER)") #calculated by 1-CER

    print("  Done.")
  