import cv2
import numpy as np
import os
from typing import List, Tuple  
import matplotlib.pyplot as plt

def preprocess_image(img: np.ndarray) -> np.ndarray:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(blurred)
    
    return equalized

def detect_buildings_color(img: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_color = np.array([0, 0, 100])  
    upper_color = np.array([180, 50, 255])
    
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = 100
    filtered_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
            
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        width, height = rect[1]
        if width == 0 or height == 0:
            continue
            
        aspect_ratio = max(width, height) / (min(width, height) + 1e-5)
        if 0.2 < aspect_ratio < 5.0:  
            filtered_contours.append(box)
    
    return mask, filtered_contours

def display_results(original: np.ndarray, mask: np.ndarray, contours: List[np.ndarray]):
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(mask, cmap='gray')
    plt.title('Filtered Image (Mask)')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    contour_img = np.zeros_like(original)
    if contours:
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    plt.title('Vectorized Contours')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    result = original.copy()
    if contours:
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        cv2.putText(result, f"Buildings: {len(contours)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Detected Buildings')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    INPUT_IMAGE = "image bing2.png"
    OUTPUT_DIR = "outputs"
    
    def load_image(path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        return img

    def save_results(img: np.ndarray, mask: np.ndarray, contours: List[np.ndarray], out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, "mask.png"), mask)
        
        result = img.copy()
        if contours:
            cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
            cv2.putText(result, f"Objects: {len(contours)}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imwrite(os.path.join(out_dir, "preview.png"), result)
        print(f"Objects detected: {len(contours)}")
        print(f"Results saved to: {os.path.abspath(out_dir)}")

    try:
        img = load_image(INPUT_IMAGE)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    print("Processing image...")
    mask, buildings = detect_buildings_color(img)
    save_results(img, mask, buildings, OUTPUT_DIR)
    display_results(img, mask, buildings)

if __name__ == "__main__":
    main()