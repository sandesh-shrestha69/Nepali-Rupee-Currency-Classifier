from sklearn.model_selection import train_test_split
import splitfolders
import os
input_folder = "./data/raw"
output_folder = "./data/processed"

# splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.7,.15,.15))


train_path ="./data/processed/train"
folders = os.listdir(train_path)
class_counts = {}
print("Image counts per class:")
for folder in folders:
    folder_path = os.path.join(train_path, folder)

    if os.path.isdir(folder_path):
        count = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        class_counts[folder] = count
        print(f"{folder}: {count} images")

total_images = sum(class_counts.values())

print(f"{'Class':<15} | {'Count':<6} | {'Ratio (%)':<10}")
print("-" * 35)
for folder, count in class_counts.items():
    ratio = (count / total_images) * 100
    print(f"{folder:<15} | {count:<6} | {ratio:.2f}%")