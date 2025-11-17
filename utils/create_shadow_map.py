import cv2
import numpy as np
from pathlib import Path

def create_shadow_map_ratio(shadow_img, shadow_free_img, ratio_threshold):

    shadow_gray = cv2.cvtColor(shadow_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    shadow_free_gray = cv2.cvtColor(shadow_free_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    shadow_gray = np.maximum(shadow_gray, 1.0)
    
    ratio = shadow_free_gray / shadow_gray
    
    shadow_mask = ratio > ratio_threshold
    
    
    shadow_map = shadow_mask.astype(np.uint8) * 255
    

    kernel = np.ones((3, 3), np.uint8)
    shadow_map = cv2.morphologyEx(shadow_map, cv2.MORPH_OPEN, kernel)  # remove noise
    shadow_map = cv2.morphologyEx(shadow_map, cv2.MORPH_CLOSE, kernel) # fill small holes
    
    return shadow_map

def process_all_images_ratio(ratio_threshold):

    shadow_dir = Path('./dataset/test/test_A')
    shadow_free_dir = Path('./dataset/test/test_C')
    output_dir = Path('./dataset/test/test_B')   
    output_dir.mkdir(parents=True, exist_ok=True)
    shadow_images = list(shadow_dir.glob('*.png'))



    processed_count = 0
    
    for shadow_path in shadow_images:

        shadow_free_path = shadow_free_dir / shadow_path.name
        
        shadow_img = cv2.imread(str(shadow_path))
        shadow_free_img = cv2.imread(str(shadow_free_path))

        shadow_map = create_shadow_map_ratio(shadow_img, shadow_free_img, ratio_threshold)
        
        output_path = output_dir / shadow_path.name
        cv2.imwrite(str(output_path), shadow_map)
        
        processed_count += 1
        if processed_count % 10 == 0:  # Print progress every 10 images
            print(f"processed {processed_count}/{len(shadow_images)} images")
            
        
    print(f"shadow maps saved to: {output_dir}")



if __name__ == "__main__":
    process_all_images_ratio(1.5)