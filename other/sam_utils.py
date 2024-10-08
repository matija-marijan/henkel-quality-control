import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.transform = transform
        self.image_paths = []

        for filename in os.listdir(dataset_dir):
            if os.path.isfile(os.path.join(dataset_dir, filename)):
                self.image_paths.append(os.path.join(dataset_dir, filename))

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)

        image_name = os.path.basename(image_path)

        return image, image_name

def show_anns(anns, alpha=0.5):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    # img = np.ones((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 4)) 
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [alpha]])
        img[m] = color_mask
    ax.imshow(img)

def show_best_ball(image, best_ball):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    show_anns([best_ball])
    plt.title('Best Ball')
    plt.axis('off')
    plt.waitforbuttonpress()
    plt.close()

def is_centered(bbox, image_width, image_height, tolerance=0.25):
    x, y, w, h = bbox
    bbox_center_x = x + w / 2
    bbox_center_y = y + h / 2
    image_center_x = image_width / 2
    image_center_y = image_height / 2
    x_tolerance = tolerance * image_width
    y_tolerance = tolerance * image_height

    flag = (abs(bbox_center_x - image_center_x) <= x_tolerance) and (abs(bbox_center_y - image_center_y) <= y_tolerance)
    
    distance_from_center = np.sqrt((bbox_center_x - image_center_x) ** 2 + (bbox_center_y - image_center_y) ** 2)
    score = 1 - (distance_from_center / np.sqrt(image_width ** 2 + image_height ** 2))  # Normalized score (closer = 1)

    return flag, score

def circularity(segmentation, tolerance=0.15):
    contours, _ = cv2.findContours(segmentation.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        perimeter = cv2.arcLength(contours[0], True)
        area = cv2.contourArea(contours[0])
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if ((1 - tolerance) <= circularity <= (1 + tolerance)):
            circular_flag = True
            circularity_score = min(1, max(0, (1 - abs(circularity - 1))))
        else:
            circular_flag = False
            circularity_score = 0
        return circular_flag, circularity_score

def select_best_ball(sam_masks, image_width, image_height):
    
    # loptice su uglavnom R = 200px
    # A = pi * R^2 = 125663.70614359172
    MIN_AREA_THRESHOLD = 100000
    MAX_AREA_THRESHOLD = 150000
    ASPECT_RATIO_TOLERANCE = 0.1
    CENTEREDNESS_TOLERANCE = 0.25
    CIRCULARITY_TOLERANCE = 0.15

    best_candidate = None
    best_score = -float('inf')

    for mask in sam_masks:

        segmentation = mask['segmentation']
        area = mask['area']
        bbox = mask['bbox']
        predicted_iou = mask['predicted_iou']
        stability_score = mask['stability_score']

        if not (MIN_AREA_THRESHOLD <= area <= MAX_AREA_THRESHOLD):
            continue

        _, _, w, h = bbox
        aspect_ratio = w / h
        if abs(aspect_ratio - 1) > ASPECT_RATIO_TOLERANCE:
            continue
        
        center_flag, centering_score = is_centered(bbox, image_width, image_height, CENTEREDNESS_TOLERANCE)
        if not center_flag:
            continue

        circular_flag, circularity_score = circularity(segmentation, CIRCULARITY_TOLERANCE)
        if not circular_flag:
            continue

        # hull = cv2.convexHull(contours[0], returnPoints=False)
        # defects = cv2.convexityDefects(contours[0], hull)
        # print(len(defects))
        # if defects is not None and len(defects) > 100:
        #     print('def')
        #     continue

        score = (
            0.4 * area / MAX_AREA_THRESHOLD +  # Normalize area score to [0, 1]
            0.3 * centering_score +  # Centering score
            0.1 * circularity_score +  # Circularity score
            0.1 * predicted_iou +  # Confidence from SAM
            0.1 * stability_score  # Stability from SAM
        )

        if score > best_score:
            best_score = score
            best_candidate = mask

    return best_candidate

def adaptive_canny(image, sigma=0.33):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the median of the pixel intensities
    v = np.median(gray)

    # Apply adaptive thresholds using the median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    # Apply Canny edge detection
    edges = cv2.Canny(gray, lower, upper)

    return edges

def detect_cracks(mask, image):
    # Use adaptive Canny edge detection to highlight potential cracks
    edges = adaptive_canny(image)
    
    # Mask the edges to the ball region only
    masked_edges = cv2.bitwise_and(edges, edges, mask=mask.astype(np.uint8))
    
    # Optionally, find contours of the edges
    contours, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cracks = []
    # You can now analyze contours for long, thin shapes
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 250:  # You can adjust this threshold
            # A long edge might indicate a crack
            cracks.append(contour)
    
    return cracks

def check_cracks(mask, image):
    edges = adaptive_canny(image)
    masked_edges = cv2.bitwise_and(edges, edges, mask=mask.astype(np.uint8))
    contours, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 250:
            return False
    return True

def show_cracks(image, cracks):

    if cracks:
        img_overlay = np.copy(image)

        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ax.imshow(image)
        ax.set_title("Cracks (Red)")
        
        for crack in cracks:
            cv2.drawContours(img_overlay, [crack], -1, (255, 0, 0), 2)  # Red for cracks
            
        ax.imshow(img_overlay)
        plt.axis('off')
        plt.waitforbuttonpress()
        plt.close()

def detect_contained_masks(best_mask, other_masks):
    """
    Check if any of the other masks are fully contained within the best_mask.
    """
    contained_masks = []
    best_segmentation = best_mask['segmentation']
    min_area = 0.015 * best_mask['area']
    
    for other_mask in other_masks:
        if np.array_equal(other_mask['segmentation'], best_mask['segmentation']):
            continue
        else:
            other_segmentation = other_mask['segmentation']

            # Check if all pixels in the other_mask are also in the main_mask
            containment_check = np.all((other_segmentation == 0) | (best_segmentation == other_segmentation))
            if containment_check:
                # Check size of other_mask relative to best_mask:

                if other_mask['area'] > min_area:
                    contained_masks.append(other_mask)
    
    return contained_masks

def check_contained_masks(best_mask, other_masks):
    best_segmentation = best_mask['segmentation']
    min_area = 0.015 * best_mask['area']
    for other_mask in other_masks:
        if np.array_equal(other_mask['segmentation'], best_mask['segmentation']):
            continue
        else:
            other_segmentation = other_mask['segmentation']
            containment_check = np.all((other_segmentation == 0) | (best_segmentation == other_segmentation))
            if containment_check:
                if other_mask['area'] > min_area:
                    return False
    return True

def show_contained_masks(image, contained_masks):
    """
    Plot the best ball mask and any contained masks on the original image.
    """
    if contained_masks:
        # Create a copy of the image with alpha channel for overlays
        img_overlay = np.concatenate([image, np.ones_like(image[:, :, 0:1])], axis=2)  # Add alpha channel

        # Prepare the figure
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ax.imshow(image)
        # print('plotted img')
        
        # show_anns([best_ball], alpha = 0.25)
        show_anns(contained_masks, alpha = 1)

        # Display the overlay
        ax.imshow(img_overlay)
        # print('plotted mask')
        plt.axis('off')
        ax.set_title("Contained Masks")
        plt.waitforbuttonpress()
        plt.close()

def detect_deformities(mask):
    # Calculate convex hull and compare it to the contour
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        hull = cv2.convexHull(contours[0])
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contours[0])
        
        # Compare convex hull area to contour area (a big difference might indicate a deformity)
        if hull_area / contour_area > 1.25:  # Adjust threshold as needed
            return True  # Detected a potential deformity
    
    return False  # No deformity detected

def check_deformities(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        hull = cv2.convexHull(contours[0])
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contours[0])
        if hull_area / contour_area > 1.25:
            return False
    return True

def show_deformities(image, mask, deformities):
    """
    Show the deformities detected in the image.
    """
    if deformities:
        img_overlay = np.copy(image)

        # Prepare the figure
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ax.imshow(image)
        ax.set_title("Deformities (Blue)")

        # Draw the ball mask contour (white)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_overlay, contours, -1, (255, 255, 255), 2)

        # Highlight deformities in blue
        
        hull = cv2.convexHull(contours[0])
        cv2.drawContours(img_overlay, [hull], -1, (0, 0, 255), 2)  # Blue for deformities

        # Display the updated image
        ax.imshow(img_overlay)
        plt.axis('off')
        plt.waitforbuttonpress()
        plt.close()

def detect_convexity(segmentation):

    contours, _ = cv2.findContours(segmentation.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        
        # Draw the convexity defects
        hull_indices = cv2.convexHull(contours[0], returnPoints=False)
        defects = cv2.convexityDefects(contours[0], hull_indices)

        depth_threshold = 10  # Minimum depth to consider a defect as large
        length_threshold = 10  # Minimum length of defect to consider

        if defects is not None:
            defects_new = []
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contours[0][s][0])
                end = tuple(contours[0][e][0])
                far = tuple(contours[0][f][0])

                # Calculate the length of the defect
                length = np.linalg.norm(np.array(start) - np.array(end))
                
                # Filter based on depth and length
                if (d / 256.0) > depth_threshold and length > length_threshold:
                    # print(f"Defect: depth={d / 256.0}, length={length}")
                    defects_new.append(defects[i, 0])
    
    return defects_new

def check_convexity(segmentation):
    contours, _ = cv2.findContours(segmentation.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        hull_indices = cv2.convexHull(contours[0], returnPoints=False)
        defects = cv2.convexityDefects(contours[0], hull_indices)
        depth_threshold = 10
        length_threshold = 10
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contours[0][s][0])
                end = tuple(contours[0][e][0])
                far = tuple(contours[0][f][0])
                length = np.linalg.norm(np.array(start) - np.array(end))
                if (d / 256.0) > depth_threshold and length > length_threshold:
                    return False
    return True

def show_convexity(segmentation, defects):
    # Create an empty image to draw everything on (uint8 type, with 3 channels for color)
    combined_img = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
    
    contours, _ = cv2.findContours(segmentation.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contour
    cv2.drawContours(combined_img, contours, 0, (0, 255, 0), 2)  # Green for contours

    # Draw the convex hull
    hull = cv2.convexHull(contours[0], returnPoints=True)
    cv2.drawContours(combined_img, [hull], 0, (255, 0, 0), 2)  # Red for hull
    
    if defects:
        for defect in defects:
                s, e, f, d = defect
                start = tuple(contours[0][s][0])
                end = tuple(contours[0][e][0])
                far = tuple(contours[0][f][0])
                cv2.line(combined_img, start, end, (0, 0, 255), 2)  # Red for large defects lines
                cv2.circle(combined_img, far, 5, (255, 0, 0), -1)  # Blue for large defects points

        plt.figure(figsize=(8, 8))
        plt.title('Contours, Convex Hull, and Convexity Defects')
        plt.imshow(combined_img)
        plt.axis('off')  # Hide the axis
        plt.waitforbuttonpress()
        plt.close()

def analyze_ball(best_candidate, image, masks):
    segmentation = best_candidate['segmentation']
    
    cracks = detect_cracks(segmentation, image)
    contained_masks = detect_contained_masks(best_candidate, masks)
    deformities = detect_deformities(segmentation)
    convexity = detect_convexity(segmentation)

    show_best_ball(image, best_candidate)
    show_cracks(image, cracks)
    show_contained_masks(image, contained_masks)
    show_deformities(image, segmentation, deformities)
    show_convexity(segmentation, convexity)
   
    return {
        "cracks": bool(cracks),
        "contained_masks": bool(contained_masks),
        "deformities": bool(deformities),
        "convexity": bool(convexity)
    }

def analyze_masks(masks, image):
    best_ball = select_best_ball(masks, image.shape[1], image.shape[0])

    if best_ball:
        print(f"Best ball found: area: {best_ball['area']}, bbox: {best_ball['bbox']}, iou: {best_ball['predicted_iou']}, stability: {best_ball['stability_score']}")
        
        results = analyze_ball(best_ball, image, masks)
        print(results)
        
        if results['cracks'] or results['deformities'] or results['contained_masks'] or results['convexity']:
            print("Defects found.")
            return 0
        else:
            print("No defects found.")
            return 1
    else:
        print("No ball found in the image.")
        return 0

def check_ball(best_candidate, image, masks):
    segmentation = best_candidate['segmentation']

    if check_contained_masks(best_candidate, masks) is False:
        return False
    if check_deformities(segmentation) is False:
        return False
    if check_convexity(segmentation) is False:
        return False
    if check_cracks(segmentation, image) is False:
        return False
    else:
        return True
    
def check_masks(masks, image):
    best_ball = select_best_ball(masks, image.shape[1], image.shape[0])
    if best_ball:
        if check_ball(best_ball, image, masks):
            print("No defects found.")
            return 1
        else:
            print("Defects found.")
            return 0
    else:
        print("No ball found in the image.")
        return 0