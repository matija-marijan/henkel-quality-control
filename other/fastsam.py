import numpy as np
import torch
import matplotlib.pyplot as plt
import csv
from fastsam import FastSAM, FastSAMPrompt
import warnings
warnings.filterwarnings("ignore")
from sam_utils import *

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
FASTSAM_CHECKPOINT = "/home/matijamarijan/projects/henkel-quality-control/model/FastSAM-s.pt"
DATASET_DIR = "/home/matijamarijan/projects/henkel-quality-control/dataset/"

model = FastSAM(FASTSAM_CHECKPOINT)
print("Model loaded successfully.")


dataset = ImageDataset(DATASET_DIR)
results = []
print(f"Dataset loaded successfully. Processing {len(dataset)} images...")

for idx in range(len(dataset)):

    print(f"Processing image {idx + 1}/{len(dataset)}...")
    image, image_name= dataset[idx]

    [masks_results] = model(image, device = DEVICE, imgsz = image.shape[0], iou = 0.9)
    masks = masks_results.masks

    new_masks = []
    for mask in masks:
        mask = mask.cpu().numpy()
        mask = mask.data
        mask = mask[0]
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        # print(mask)
        # print(type(mask))
        # print(np.shape(mask))
        new_masks.append(mask)
        # plt.figure()
        # plt.imshow(mask, cmap='gray')  # Plot each mask
        # plt.axis('off')  # Hide axes
        # plt.show()

    masks = new_masks

    # plt.figure(figsize=(8, 8))
    # plt.imshow(image)
    # plt.axis('off')
    # plt.title(image_name)
    # # show_anns(masks)
    # plt.waitforbuttonpress()
    # plt.close()
    
    # result = analyze_masks(masks, image)
    result = check_masks(masks, image)

    results.append((image_name, result))
    status = "PASS" if result == 1 else "FAIL"

    print(f"Image {idx + 1}/{len(dataset)}: {status}")

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