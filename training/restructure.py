import os
import shutil

def restructure_folder(root_folder, img_folder_name="fold6", gt_folder_name="fold6_annotation"):
    """
    Restructure the dataset folders so that each image file in the flat img_folder (JPG)
    and each mask file in the flat gt_folder (PNG) are moved into their own subfolder.
    
    After restructuring, the folder structure will look like:
    
    root_folder/
      img_folder/
        sample1/
          1.jpg
        sample2/
          1.jpg
      gt_folder/
        sample1/
          1.png
        sample2/
          1.png
    """
    # Construct full paths to image and ground truth folders
    img_folder = os.path.join(root_folder, img_folder_name)
    gt_folder = os.path.join(root_folder, gt_folder_name)
    
    # Process image folder (JPG files)
    for filename in os.listdir(img_folder):
        file_path = os.path.join(img_folder, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(".jpg"):
            base_name, _ = os.path.splitext(filename)
            new_dir = os.path.join(img_folder, base_name)
            os.makedirs(new_dir, exist_ok=True)
            
            # Rename the file to "1.jpg" for a single frame per video
            new_file_path = os.path.join(new_dir, "1.jpg")
            shutil.move(file_path, new_file_path)
            print(f"Moved {file_path} -> {new_file_path}")
    
    # Process ground truth folder (PNG files)
    for filename in os.listdir(gt_folder):
        file_path = os.path.join(gt_folder, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(".png"):
            base_name, _ = os.path.splitext(filename)
            new_dir = os.path.join(gt_folder, base_name)
            os.makedirs(new_dir, exist_ok=True)
            
            # Rename the file to "1.png" for a single frame per video
            new_file_path = os.path.join(new_dir, "1.png")
            shutil.move(file_path, new_file_path)
            print(f"Moved {file_path} -> {new_file_path}")

def main():
    restructure_folder("./FOLD6/")

if __name__ == "__main__":
    main()
