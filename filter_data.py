import os
import shutil

source_dir = 'raw_data/PlantVillage'
dest_dir = 'filtered_dataset'
allowed_crops = ['Tomato', 'Potato']

print("Filtering dataset for Tomato and Potato...")

for split in ['train', 'val']:
    split_source = os.path.join(source_dir, split)
    split_dest = os.path.join(dest_dir, split)
    os.makedirs(split_dest, exist_ok=True)
    
    if os.path.exists(split_source):
        for folder_name in os.listdir(split_source):
            # Copy folder if it starts with 'Tomato' or 'Potato'
            if any(folder_name.startswith(crop) for crop in allowed_crops):
                src_path = os.path.join(split_source, folder_name)
                dest_path = os.path.join(split_dest, folder_name)
                
                if not os.path.exists(dest_path):
                    shutil.copytree(src_path, dest_path)
                    print(f"Copied {split}: {folder_name}")

print("\nDataset successfully filtered into 'filtered_dataset/'.")
