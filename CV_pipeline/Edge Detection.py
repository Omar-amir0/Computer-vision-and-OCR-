import cv2
import os
import numpy as np
from    enhancement import enhance_image


# Input folders
#change before using
input_folders = [
    r"C:\Users\Omar\Desktop\omar\cv\Captured_Images\Samsung_phone\Images"
]

# Output folder
#for enhanced folders
enhanced_folder = r"C:\Users\Omar\Desktop\omar\cv\enhancement"
os.makedirs(enhanced_folder, exist_ok=True)
# Output folder (rotated images)
output_folder = r"C:\Users\Omar\Desktop\omar\cv\test"
os.makedirs(output_folder, exist_ok=True)

# for contour visualization
contour_folder = r"C:\Users\Omar\Desktop\omar\cv\contours"
os.makedirs(contour_folder, exist_ok=True)

def order_points(document_contour):
    document_contour = document_contour.reshape(4, 2).astype('float32')
    ordered = np.zeros((4, 2), dtype='float32')
    
    sum_all = document_contour.sum(axis=1)
    ordered[0] = document_contour[np.argmin(sum_all)]   # Top-left
    ordered[2] = document_contour[np.argmax(sum_all)]   # Bottom-right
    sub = np.diff(document_contour, axis=1)
    ordered[1] = document_contour[np.argmin(sub)]   # Top-right
    ordered[3] = document_contour[np.argmax(sub)]   # Bottom-left
    
    
    return ordered



    
    

def fix_rotation(image, ordered):
    
    w_top = np.linalg.norm(ordered[1] - ordered[0])
    h_left = np.linalg.norm(ordered[3] - ordered[0])
    
    fix = image
    if w_top > h_left:
        fix = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return fix
    
    



def detect_document(image_path, save_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read: {image_path}")
        return

    # ======================================================
    # STEP 1: IMAGE ENHANCEMENT (FROM IMPORTED MODULE)
    # ======================================================
    enhanced = enhance_image(image)
    original = image.copy()

    # save enhanced image
    enhanced_name = os.path.basename(save_path).replace(".jpg", "_enhanced.jpg")
    enhanced_path = os.path.join(enhanced_folder, enhanced_name)
    cv2.imwrite(enhanced_path, enhanced)
    print(f"Saved enhanced: {enhanced_path}")

    # Step 3: Edge Detection
    edges = cv2.Canny(image, 75, 200)

    # save edges image
    #edges_path = save_path.replace(".jpg", "_edges.jpg")
    #cv2.imwrite(edges_path, edges)
    #print(f"Saved edges: {edges_path}")

    #Find Contours
    contours, _ = cv2.findContours(
        edges.copy(),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    document_contour = None

    # Find 4-point contour
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            document_contour = approx
            break

    #Draw contour
    if document_contour is not None:
        cv2.drawContours(original, [document_contour], -1, (0, 255, 0), 3)
    else:
        print(f"No document found in: {image_path}")
        return

    #Save contour img
    contour_path = os.path.join(contour_folder, os.path.basename(save_path))
    cv2.imwrite(contour_path, original)
    print(f"Saved contour: {contour_path}")

    ordered = order_points(document_contour)

    #Rotation
    fix = fix_rotation(enhanced, ordered)

    # Save final rotated image
    cv2.imwrite(save_path, fix)
    print(f"Saved rotated: {save_path}")


# Process all folders
for folder in input_folders:
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder, filename)

            save_name = f"{os.path.basename(folder)}_{filename}"
            save_path = os.path.join(output_folder, save_name)

            detect_document(image_path, save_path)

print("Processing completed successfully.")

#SINGLE FILE INSTEAD ALL THAT LOOP saves in test folder 
#image_path = r"C:\Users\Omar\Desktop\omar\cv\Captured_Images\Samsung_phone\Images\IMAGE.jpg"
#save_name = "output.jpg"
#save_path = os.path.join(output_folder, save_name)
#detect_document(image_path, save_path)
#print("Single image processed successfully.")