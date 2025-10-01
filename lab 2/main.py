import cv2
import numpy as np
import os
from typing import List, Tuple

def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """Enhance image contrast using CLAHE in LAB color space"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def extract_building_features(img: np.ndarray) -> np.ndarray:
    """Extract features that help identify buildings while excluding trees and roads"""
    # Convert to HSV color space for better color segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for non-building areas (trees, grass, roads)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    lower_gray = np.array([0, 0, 100])
    upper_gray = np.array([180, 30, 220])
    
    # Create masks for non-building areas
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
    
    # Combine masks to exclude non-building areas
    non_building_mask = cv2.bitwise_or(mask_green, mask_gray)
    non_building_mask = cv2.bitwise_not(non_building_mask)
    
    # Convert to grayscale and apply thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Combine with the non-building mask
    result = cv2.bitwise_and(thresh, thresh, mask=non_building_mask)
    
    # Apply morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return result

def detect_buildings(img: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Building detection optimized for KPI satellite image"""
    # 1. Enhance contrast
    enhanced = enhance_contrast(img)
    
    # 2. Extract building features
    features = extract_building_features(enhanced)
    
    # 3. Find contours
    contours, _ = cv2.findContours(
        features, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # 4. Filter contours
    min_area = 50  # Minimum area for buildings
    max_area = 10000  # Maximum area for buildings
    filtered_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
            
        # Get the minimum area rectangle
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Calculate aspect ratio and solidity
        width, height = rect[1]
        if width == 0 or height == 0:
            continue
            
        aspect_ratio = max(width, height) / (min(width, height) + 1e-5)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / (hull_area + 1e-5)
        
        # Filter based on shape properties
        if (0.2 < aspect_ratio < 5.0 and  # Flexible aspect ratio
            0.4 < solidity < 1.1 and      # Check for solid shapes
            0.3 < (area / (width * height + 1e-5)) < 1.5):  # Area ratio
            filtered_contours.append(box)
    
    # Create final mask
    final_mask = np.zeros_like(features)
    cv2.drawContours(final_mask, filtered_contours, -1, 255, -1)
    
    return final_mask, filtered_contours

def main():
    # Configuration
    DEFAULT_BING_PATH = "image bing.png"
    DEFAULT_OUTPUT_DIR = "outputs"
    
    def load_image(path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return img

    def save_results(img: np.ndarray, mask: np.ndarray, contours: List[np.ndarray], out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        # Save binary mask
        cv2.imwrite(os.path.join(out_dir, "mask.png"), mask)
        
        # Draw contours on original image
        vis = img.copy()
        if contours:
            cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(out_dir, "preview.png"), vis)

    # Load the Bing image
    try:
        img = load_image(DEFAULT_BING_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please ensure the Bing image is at: {os.path.abspath(DEFAULT_BING_PATH)}")
        return
    
    print("Processing KPI satellite image...")
    
    # Detect buildings
    mask, buildings = detect_buildings(img)
    
    # Save results
    save_results(img, mask, buildings, DEFAULT_OUTPUT_DIR)
    
    print(f"Detected {len(buildings)} buildings")
    print(f"Results saved to: {os.path.abspath(DEFAULT_OUTPUT_DIR)}")

if __name__ == "__main__":
    main()