import os
import json
import shutil
import SimpleITK as sitk
import numpy as np

# ----------------------------
# USER PATHS
# ----------------------------
original_dir = r"C:/nnunet/nnunet_raw/Dataset003_Liver/"
new_dir      = r"C:/nnunet/nnunet_raw/Dataset003_Liver_tumor/"
# ----------------------------

print("Preparing new dataset:", new_dir)

# Create folder structure
for sub in ["imagesTr", "labelsTr", "imagesTs", "labelsTs"]:
    os.makedirs(os.path.join(new_dir, sub), exist_ok=True)

# ---------------------------------------------------------
# Helper: resample label to match the corresponding image
# ---------------------------------------------------------
def resample_label_to_image(label_img, ref_img):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(label_img)

# ---------------------------------------------------------
# Process TRAINING labels
# ---------------------------------------------------------
labelsTr_dir = os.path.join(original_dir, "labelsTr")
imagesTr_dir = os.path.join(original_dir, "imagesTr")

train_labels = sorted([f for f in os.listdir(labelsTr_dir) if f.endswith(".nii.gz")])

print(f"\nFound {len(train_labels)} training labels")

for i, label_file in enumerate(train_labels):
    label_path = os.path.join(labelsTr_dir, label_file)

    case_id = label_file.replace(".nii.gz", "")   # liver_0
    image_file = f"{case_id}_0000.nii.gz"         # liver_0_0000.nii.gz
    image_path = os.path.join(imagesTr_dir, image_file)

    if not os.path.exists(image_path):
        print("Missing image for:", label_file)
        continue

    # Load image & label
    img  = sitk.ReadImage(image_path)
    lab  = sitk.ReadImage(label_path)

    # Resample label to image
    lab_resampled = resample_label_to_image(lab, img)

    # Convert label: 2 → 1 (tumor), everything else → 0
    arr = sitk.GetArrayFromImage(lab_resampled)
    arr_binary = (arr == 2).astype(np.uint8)

    lab_bin = sitk.GetImageFromArray(arr_binary)
    lab_bin.CopyInformation(img)

    # Save
    out_label = os.path.join(new_dir, "labelsTr", label_file)
    sitk.WriteImage(lab_bin, out_label)

    # Copy corresponding image
    out_image = os.path.join(new_dir, "imagesTr", image_file)
    shutil.copy2(image_path, out_image)

    if (i + 1) % 5 == 0:
        print(f"Processed {i+1}/{len(train_labels)}")

# ---------------------------------------------------------
# Process TEST set (if exists)
# ---------------------------------------------------------
labelsTs_dir = os.path.join(original_dir, "labelsTs")
imagesTs_dir = os.path.join(original_dir, "imagesTs")

if os.path.exists(labelsTs_dir):
    test_labels = sorted([f for f in os.listdir(labelsTs_dir) if f.endswith(".nii.gz")])
    print(f"\nFound {len(test_labels)} test labels")
else:
    test_labels = []
    print("\nNo test labels found")

for i, label_file in enumerate(test_labels):
    label_path = os.path.join(labelsTs_dir, label_file)
    case_id = label_file.replace(".nii.gz", "")
    image_file = f"{case_id}_0000.nii.gz"
    image_path = os.path.join(imagesTs_dir, image_file)

    if not os.path.exists(image_path):
        print("Missing test image:", label_file)
        continue

    img  = sitk.ReadImage(image_path)
    lab  = sitk.ReadImage(label_path)
    lab_resampled = resample_label_to_image(lab, img)

    arr = sitk.GetArrayFromImage(lab_resampled)
    arr_binary = (arr == 2).astype(np.uint8)

    lab_bin = sitk.GetImageFromArray(arr_binary)
    lab_bin.CopyInformation(img)

    out_label = os.path.join(new_dir, "labelsTs", label_file)
    sitk.WriteImage(lab_bin, out_label)

    out_image = os.path.join(new_dir, "imagesTs", image_file)
    shutil.copy2(image_path, out_image)

# ---------------------------------------------------------
# Create dataset.json
# ---------------------------------------------------------
json_old = os.path.join(original_dir, "dataset.json")
json_new = os.path.join(new_dir, "dataset.json")

with open(json_old, "r") as f:
    data = json.load(f)

# overwrite labels → binary segmentation
data["labels"] = {
    "background": 0,
    "tumor": 1
}

with open(json_new, "w") as f:
    json.dump(data, f, indent=4)

print("\n-------------------------------------")
print("Dataset conversion complete!")
print("New dataset at:", new_dir)
print("-------------------------------------")
