import numpy as np
import torch
import matplotlib.pyplot as plt
import csv
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import warnings
warnings.filterwarnings("ignore")
from utils import *

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SAM_CHECKPOINT = "./model/sam_vit_b_01ec64.pth"
DATASET_DIR = "./dataset/"

sam = sam_model_registry["vit_b"](checkpoint = SAM_CHECKPOINT)
sam.to(device = DEVICE)
print("Model loaded successfully.")

mask_generator = SamAutomaticMaskGenerator(
    model = sam,
    min_mask_region_area=250,
    pred_iou_thresh = 0.9,
    stability_score_thresh = 0.9)
print("Mask generator initialized successfully.")

dataset = ImageDataset(DATASET_DIR)
results = []
print(f"Dataset loaded successfully. Processing {len(dataset)} images...")

for idx in range(len(dataset)):

    print(f"Processing image {idx + 1}/{len(dataset)}...")
    image, image_name = dataset[idx]

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(image_name)
    plt.show()
    
    masks = mask_generator.generate(image)
    result = analyze_masks(masks, image)

    results.append((image_name, result))
    status = "PASS" if result == 1 else "FAIL"

    print(f"Processed image {idx + 1}/{len(dataset)}: {status}")

print("All images processed successfully. Saving results to a CSV file...")
# Save results to a CSV file
csv_file = "results.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image Name", "Quality result"])  # Write the header
    writer.writerows(results)  # Write the data
print(f"Results saved to {csv_file}")

'''
Test if there is a foreground
Extract the mask
Test if a mask is centered?
Test if a mask is a circle
Test if a mask has a crack 
Check if mask has color blemishes
Check if mask has irregularities

'''