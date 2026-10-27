import splitfolders

splitfolders.ratio(
    'raw_data/PlantVillage', 
    output='dataset', 
    seed=42, 
    ratio=(0.8, 0.2)
)
print("Dataset split complete!")
