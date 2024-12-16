import os

# Paths to train and test labels
train_labels_path = r"C:/Users/kalai/OneDrive/Desktop/gun_detection/yolov5/weapon_detection/train/labels"
test_labels_path = r"C:/Users/kalai/OneDrive/Desktop/gun_detection/yolov5/weapon_detection/val/labels"

# Function to update class indices in label files
def update_labels_to_class_zero(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as file:
                lines = file.readlines()

            # Modify the class index to 0 for each line
            updated_lines = []
            for line in lines:
                parts = line.split()
                if len(parts) > 0:
                    parts[0] = "0"  # Set class index to 0
                    updated_lines.append(" ".join(parts) + "\n")

            # Save the updated lines back to the file
            with open(file_path, "w") as file:
                file.writelines(updated_lines)
            print(f"Updated: {file_path}")

# Update labels in train and test folders
update_labels_to_class_zero(train_labels_path)
update_labels_to_class_zero(test_labels_path)

print("Class index update completed.")
