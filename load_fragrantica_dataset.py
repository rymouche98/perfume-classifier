import kagglehub
import zipfile
import os
import shutil

# Download latest version
path = kagglehub.dataset_download("olgagmiufana1/fragrantica-com-fragrance-dataset")

print("Path to dataset files:", path)

# The dataset is already downloaded and extracted to the path above.
# We can now list the files in that directory.
print("Dataset files:", os.listdir(path))

# Define a more convenient local directory to store the dataset
local_dataset_path = os.path.join(os.path.expanduser("~"), "fragrantica_dataset")
os.makedirs(local_dataset_path, exist_ok=True)

print(f"Copying files to: {local_dataset_path}")

# Copy each file from the source to the new local directory
for file_name in os.listdir(path):
    source_file = os.path.join(path, file_name)
    destination_file = os.path.join(local_dataset_path, file_name)
    if os.path.isfile(source_file):
        shutil.copy(source_file, destination_file)
    else:
        shutil.copytree(source_file, destination_file, dirs_exist_ok=True)

print("Files copied successfully!")
print("Your locally stored files are:", os.listdir(local_dataset_path))
