import cv2
import numpy as np
import os
from typing import Any
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

def enhance_brightness(pil_img: Image.Image) -> Image.Image:
    enhancer = ImageEnhance.Brightness(pil_img)
    return enhancer.enhance(1.2)  

def detect_buildings(img: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_color = np.array([0, 0, 100])  
    upper_color = np.array([180, 50, 255])
    
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = 100
    min_aspect_ratio = 0.2
    max_aspect_ratio = 5.0
    filtered_contours = []
    
    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        width, height = rect[1]

        if width == 0 or height == 0:
            continue

        aspect_ratio = max(width, height) / (min(width, height) + 1e-5)
        
        if min_aspect_ratio < aspect_ratio < max_aspect_ratio:
            filtered_contours.append(box)
    
    return mask, filtered_contours

def process_image(img: np.ndarray) -> dict[str, Any]:
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    enhanced_img = enhance_brightness(pil_img)
    img_cv = cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_RGB2BGR)
    
    mask, buildings = detect_buildings(img_cv)
    
        
    result_img = img_cv.copy()
    if len(buildings) > 0:
        cv2.drawContours(result_img, buildings, -1, (0, 255, 0), 2)
        cv2.putText(result_img, f"Buildings: {len(buildings)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 0, 255), 2)
    
    return {
        'image': img_cv,
        'mask': mask,
        'buildings': buildings,
        'result': result_img
    }

def display_result(result: dict[str, Any]):
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(result['result'], cv2.COLOR_BGR2RGB))
    plt.title(f"Brightness Enhanced - {len(result['buildings'])} buildings detected")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def save_result(result: dict[str, Any], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
        
    cv2.imwrite(os.path.join(output_dir, "enhanced.png"), result['image'])
    cv2.imwrite(os.path.join(output_dir, "enhanced_result.png"), result['result'])
    cv2.imwrite(os.path.join(output_dir, "enhanced_mask.png"), result['mask'])

def main():
    INPUT_IMAGE = "image bing2.png"
    OUTPUT_DIR = "enhanced_results"
    
    img = cv2.imread(INPUT_IMAGE)
    
    print("Processing image with brightness enhancement...")
    result = process_image(img)
    
    print("Displaying result...")
    display_result(result)
    
    print("Saving results...")
    save_result(result, OUTPUT_DIR)
    print(f"Results saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("Done!")

if __name__ == "__main__":
    main()