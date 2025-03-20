import os
import shutil

def remove_unmatched_subfolders(folder1, folder2):
    """Remove subfolders from both folders if they do not appear in both."""
    
    # Get all subfolder names in each directory
    subfolders1 = {name for name in os.listdir(folder1) if os.path.isdir(os.path.join(folder1, name))}
    subfolders2 = {name for name in os.listdir(folder2) if os.path.isdir(os.path.join(folder2, name))}
    
    # Find common subfolders
    common_subfolders = subfolders1 & subfolders2

    # Remove subfolders that are not in both directories
    for folder in subfolders1 - common_subfolders:
        folder_path = os.path.join(folder1, folder)
        shutil.rmtree(folder_path)
        print(f"Deleted: {folder_path}")
    
    for folder in subfolders2 - common_subfolders:
        folder_path = os.path.join(folder2, folder)
        shutil.rmtree(folder_path)
        print(f"Deleted: {folder_path}")

    print("Cleanup complete. Only common subfolders remain.")

# Example usage:
folder_path1 = "./FOLD6/valid/img_folder/"
folder_path2 = "./FOLD6/valid/gt_folder/"

remove_unmatched_subfolders(folder_path1, folder_path2)
