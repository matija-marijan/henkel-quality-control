import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import warnings
warnings.filterwarnings("ignore")
from utils import *

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SAM_CHECKPOINT = "./model/sam_vit_b_01ec64.pth"

sam = sam_model_registry["vit_b"](checkpoint = SAM_CHECKPOINT)
sam.to(device = DEVICE)

# IMAGE_PATH = "./Dataset/pune (25).bmp"
# IMAGE_PATH = "./Dataset/pune (49).bmp"
# IMAGE_PATH = "./Dataset/pune (85).bmp"
# IMAGE_PATH = "./Dataset/pune (93).bmp"
# IMAGE_PATH = "./Dataset/pune (142).bmp"
# IMAGE_PATH = "./Dataset/pune (377).bmp"
IMAGE_PATH = "./Dataset/pune (383).bmp"
# IMAGE_PATH = "./Dataset/prazna (1).bmp"

image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(8, 8))
# plt.imshow(image)
# plt.axis('off')
# plt.show()

mask_generator = SamAutomaticMaskGenerator(
    model = sam,
    min_mask_region_area=250,
    # points_per_side = 4,
    pred_iou_thresh = 0.9,
    stability_score_thresh = 0.9
)   

masks = mask_generator.generate(image)
# print(masks)
# print(len(masks))

# plt.figure(figsize=(8, 8))
# plt.imshow(image)
# show_anns(masks)
# plt.axis('off')
# plt.show() 

best_ball = select_best_ball(masks, image.shape[1], image.shape[0], image)

if best_ball:
    print(f"Best ball found:, area: {best_ball['area']}, bbox: {best_ball['bbox']}, iou: {best_ball['predicted_iou']}, stability: {best_ball['stability_score']}")
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    show_anns([best_ball])
    plt.axis('off')
    plt.show() 

    # Quality (cracks, containment, blemishes, deformities)
    results = analyze_ball(best_ball, image, masks)
    print(results)

else:
    print("No ball found in the image.")

'''
Test if there is a foreground
Extract the mask
Test if a mask is centered?
Test if a mask is a circle
Test if a mask has a crack 
Check if mask has color blemishes
Check if mask has irregularities

'''


# plt.figure(figsize=(16, 8))
# # Display the original image
# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.title('Original Image')
# plt.axis('off')
# # Display the best mask
# plt.subplot(1, 2, 2)
# show_anns(masks)
# plt.title('All masks')
# plt.axis('off')
# plt.show()