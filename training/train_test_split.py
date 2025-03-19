#!/usr/bin/env python3
import os
import shutil
import argparse
import random

def validate_samples(img_folder, gt_folder):
    """
    Ensure each image sample has a corresponding ground truth sample.
    If an image sample does not have a corresponding gt sample, remove it.
    
    Args:
        img_folder (str): Path to the image folder containing sample subfolders.
        gt_folder (str): Path to the ground truth folder containing sample subfolders.
    
    Returns:
        List of valid sample names that exist in both img_folder and gt_folder.
    """
    img_samples = sorted([d for d in os.listdir(img_folder) if os.path.isdir(os.path.join(img_folder, d))])
    gt_samples = set(os.listdir(gt_folder))  # Convert to set for fast lookup

    valid_samples = []
    
    for sample in img_samples:
        img_path = os.path.join(img_folder, sample)
        gt_path = os.path.join(gt_folder, sample)

        if sample in gt_samples and os.path.isdir(gt_path):
            valid_samples.append(sample)
        else:
            # Remove the image sample if no corresponding ground truth exists
            shutil.rmtree(img_path)
            print(f"Removed {img_path} (No corresponding GT found)")

    return valid_samples

def train_valid_split(root_folder, train_ratio=0.8, seed=42):
    """
    Splits the dataset into training and validation sets after ensuring all images have corresponding GT.

    The original structure is assumed to be:
    
    root_folder/
        img_folder/
            sample1/  (contains one or more JPG images)
            sample2/
            ...
        gt_folder/
            sample1/  (contains one or more PNG images)
            sample2/
            ...
    
    After running this script, the following folders are created:
    
    root_folder/
        train/
            img_folder/
                sample1/
                ...
            gt_folder/
                sample1/
                ...
        valid/
            img_folder/
                sampleX/
                ...
            gt_folder/
                sampleX/
                ...
    """
    # Define the source directories
    img_folder = os.path.join(root_folder, "fold6")
    gt_folder = os.path.join(root_folder, "fold6_annotation")
    
    # Validate samples and remove unmatched ones
    valid_samples = validate_samples(img_folder, gt_folder)

    # Shuffle the sample list with the provided seed
    random.seed(seed)
    random.shuffle(valid_samples)
    
    # Split into train and validation sets
    num_train = int(len(valid_samples) * train_ratio)
    train_samples = valid_samples[:num_train]
    valid_samples = valid_samples[num_train:]
    
    # Define destination directories for train and valid splits
    train_img_dir = os.path.join(root_folder, "train", "img_folder")
    train_gt_dir = os.path.join(root_folder, "train", "gt_folder")
    valid_img_dir = os.path.join(root_folder, "valid", "img_folder")
    valid_gt_dir = os.path.join(root_folder, "valid", "gt_folder")
    
    # Create destination directories
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_gt_dir, exist_ok=True)
    os.makedirs(valid_img_dir, exist_ok=True)
    os.makedirs(valid_gt_dir, exist_ok=True)
    
    # Helper function to copy a sample folder
    def copy_sample(sample, src_root, dst_root, sample_type="img"):
        src_path = os.path.join(src_root, sample)
        dst_path = os.path.join(dst_root, sample)
        if os.path.exists(src_path):
            shutil.copytree(src_path, dst_path)
            print(f"Copied {src_path} -> {dst_path}")
        else:
            print(f"Warning: {src_path} does not exist for {sample_type} data.")

    # Copy training samples
    for sample in train_samples:
        copy_sample(sample, img_folder, train_img_dir, sample_type="img")
        copy_sample(sample, gt_folder, train_gt_dir, sample_type="gt")
    
    # Copy validation samples
    for sample in valid_samples:
        copy_sample(sample, img_folder, valid_img_dir, sample_type="img")
        copy_sample(sample, gt_folder, valid_gt_dir, sample_type="gt")
    
    print("Train samples:", len(train_samples))
    print("Validation samples:", len(valid_samples))

def main():
    parser = argparse.ArgumentParser(
        description="Perform train-validation split for a restructured VOS dataset. "
                    "The script ensures that each image sample has a corresponding ground truth."
    )
    parser.add_argument("--root_folder", type=str, default="./FOLD6/", help="Root folder containing 'img_folder' and 'gt_folder'")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Proportion of data for training (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits")
    
    args = parser.parse_args()
    train_valid_split(args.root_folder, args.train_ratio, args.seed)

if __name__ == "__main__":
    main()
