# Document Scanner and OCR Pipeline

A computer vision project that takes photos of printed documents and processes them into perspective-corrected, enhanced, binarized images, then extracts their text with Tesseract OCR.

## Dataset

The project uses the [SmartDocIQA D1 to D8 dataset on Kaggle](https://www.kaggle.com/datasets/grobibib/smartdociqa-d1-to-d8). Download the dataset from Kaggle and place the source images in the input folder configured in the scripts.

The project also contains `Truth_texts/`, which holds small and manually prepared reference transcriptions used for OCR evaluation.

## Pipeline

For each input image, `CV_pipeline/full_pipeline.py` runs these stages:

1. **Document detection:** convert the image to grayscale, blur it, detect edges with Canny, and find a large four-corner contour. A minimum-area rectangle is used as a fallback.
2. **Perspective correction:** order the document corners and apply a four-point perspective transform. The output is rotated to a consistent orientation.
3. **Image enhancement:** convert to grayscale, apply CLAHE contrast enhancement, and sharpen the result.
4. **Text segmentation:** use adaptive Gaussian thresholding and connected-component filtering to produce a binary image for OCR.
5. **OCR:** use Tesseract (`--oem 3 --psm 6`) to extract text and word confidence values. The pipeline saves the text and an image annotated with high-confidence word boxes.

The modules are in `CV_pipeline/`:

- `full_pipeline.py` — runs the end-to-end batch pipeline.
- `segmentation.py` — document contour detection, perspective transform, and text binarization.
- `enhancement.py` — grayscale contrast enhancement and sharpening.
- `ocr.py` — Tesseract text extraction and confidence-box visualization.
- `metrics.py` — evaluates segmentation overlap and OCR text against reference data.
- `Edge Detection.py` — standalone document edge detection utility.

## Output folders

The pipeline writes intermediate and final results under the configured project output directory:

| Folder | Contents |
| --- | --- |
| `edges/` | Edge maps from document detection |
| `contours/` | Input images with the detected document boundary drawn on them |
| `warped/` | Perspective-corrected document images |
| `enhanced/` | Contrast-enhanced and sharpened images |
| `segmented/` | Binary images prepared for OCR |
| `ocr/` | OCR text files and annotated OCR images |

Python package dependencies are listed in [`requirements.txt`](requirements.txt).

## Configure and run

Before running, update the path constants in `CV_pipeline/segmentation.py`, `CV_pipeline/full_pipeline.py`, `CV_pipeline/ocr.py`, and `CV_pipeline/metrics.py` to match your local project, dataset, and Tesseract installation. In particular, set the input image folder, output folder, ground-truth transcription folder, and `pytesseract.pytesseract.tesseract_cmd`. Tesseract OCR must be installed separately.

From the `CV_pipeline/` directory, run the full_pipleine.py 


The script processes image files in the configured input folder (`.jpg`, `.jpeg`, `.png`, and `.bmp`). It skips images where it cannot detect a document contour.

To calculate metrics after processing, run:

```bash
python metrics.py
```

`metrics.py` reports mean OCR character error rate (CER), OCR matching accuracy (`1 - CER`), mean Tesseract confidence, and segmentation IoU. The OCR comparison uses matching `.txt` files in `Truth_texts/`. The segmentation IoU uses word boxes detected by Tesseract on the warped source image as an automatically generated comparison mask, so it is an approximate measure rather than comparison against manually annotated pixel masks.

## Notes

- Dataset download and use are subject to the terms shown on its Kaggle page.
- The scripts currently use local path constants, so those paths must be updated on another machine before running.
