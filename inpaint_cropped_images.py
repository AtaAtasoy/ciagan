import os
import numpy as np
import cv2
'''
We use the aligned version where each image is centered on a point in- between person’s eyes, 
and then padded and resized to have 178 × 218 resolution, while maintaining original face proportions.
'''
if __name__ == "__main__":
    # Anonymized path
    root = os.path.join('cityscapes_test')
    anonymized_root = os.path.join(root, 'anonymised')
    original_images_root = os.path.join(root, 'orig')
    cropped_images_root = os.path.join(root, 'clr')
    output_images_root = os.path.join(root, 'inpainted')
    landmarks_root = os.path.join(root, 'lndm')
    
    categories = os.listdir(anonymized_root)
    
    for category in categories:
        anonymized_image_category = os.path.join(anonymized_root, category)
        original_images_category = os.path.join(original_images_root, category)
        cropped_images_category = os.path.join(cropped_images_root, category)
        landmarks_category = os.path.join(landmarks_root, category)
        
        if not os.path.exists(output_images_root):
            os.makedirs(output_images_root)
        
        if not os.path.exists(os.path.join(output_images_root, category)):
            os.makedirs(os.path.join(output_images_root, category))
            
        output_images_category = os.path.join(output_images_root, category)
        
        print(f"Anonymizing images in {anonymized_image_category}")
        print(f"Original images in {original_images_category}")
        print(f"Cropped images in {cropped_images_category}")
        
        crop_coordinates = np.load(original_images_category + '/crop_coord.npy')        
        # Group the crop coordinates by the original image name
        crop_coordinates_grouped = {}
        for crop in crop_coordinates:
            original_img_name = crop[5].split('__')[0]
            if original_img_name not in crop_coordinates_grouped:
                crop_coordinates_grouped[original_img_name] = []
            crop_coordinates_grouped[original_img_name].append(crop)
            
        # Iterate through the original images
        for image in crop_coordinates_grouped.keys():
            
            crops = crop_coordinates_grouped[image]
            for crop in crops:
                # Crop is in the form of (y1, y2, x1, x2, padding_size, original_image_name)
                y_min, y_max, x_min, x_max, padding, anonymized_img_name = crop
                
                y_min, y_max, x_min, x_max, padding = int(y_min), int(y_max), int(x_min), int(x_max), int(padding)
                # Skip if the crop is out of bounds
                if y_min < 0 or y_max < 0 or x_min < 0 or x_max < 0:
                    continue

                crop_w, crop_h = x_max - x_min, y_max - y_min
                

                original_img_name = anonymized_img_name.split('__')[0]
                print(f"original_img_name: {original_img_name}")
                # If there is an inpainted version of the original image, load it
                if os.path.exists(os.path.join(output_images_category, original_img_name + ".jpg")):
                    original_img = cv2.imread(os.path.join(output_images_category, original_img_name + '.jpg'))
                else:
                    original_img = cv2.imread(os.path.join(original_images_category, original_img_name + '.jpg'))
                    
                # Load the anonymized image
                anonymized_img = cv2.imread(os.path.join(anonymized_image_category, anonymized_img_name))
                print(f"anonymized_img.shape: {anonymized_img.shape}")
                
                # Load the crop from the original image
                cropped_img = cv2.imread(os.path.join(cropped_images_category, anonymized_img_name))
                resized_anonymized_img = cv2.resize(anonymized_img, (cropped_img.shape[1], cropped_img.shape[0]), interpolation = cv2.INTER_CUBIC)
                print(f"resized_anonymized_img.shape: {resized_anonymized_img.shape}")
                print(f"padding: {padding}")
                
                # Resize the anonymized image to the same size as the crop
                resized_anonymized_img = cv2.resize(anonymized_img, (crop_w, crop_h), interpolation = cv2.INTER_CUBIC)
                
                # Blend the anonymized image with the crop
                mask = np.zeros(cropped_img.shape[:2], np.uint8)
                mask[padding:y_max-y_min-padding, padding:x_max-x_min-padding] = 255
                dst = cv2.inpaint(cropped_img, mask, 3, cv2.INPAINT_TELEA)
                # Make sure resized_anonymized_img is the same size as dst
                dst = cv2.resize(dst, (crop_w, crop_h))
                
                print(f"resized_anonymized_img.shape: {resized_anonymized_img.shape}")
                print(f"dst.shape: {dst.shape}")
                print(f"Cropped image shape: {cropped_img.shape}")
                print(f"Crop coord shape: {(crop_w, crop_h)}")
                
                # Blend the inpainted image with the anonymized image
                blended_img = cv2.addWeighted(resized_anonymized_img, 0.5, dst, 0.5, 0)
                blended_img = cv2.resize(blended_img, 
                                         (original_img[y_min:y_max, x_min:x_max].shape[1], original_img[y_min:y_max, x_min:x_max].shape[0]), 
                                         interpolation = cv2.INTER_CUBIC)
                
                # Write the blended image to the original image
                original_img[y_min:y_max, x_min:x_max] = blended_img
                
                # Save the inpainted image
                cv2.imwrite(os.path.join(output_images_category, original_img_name + ".jpg"), original_img)
                break
            break
        break