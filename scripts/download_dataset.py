# Create download_dataset.py
import requests
import zipfile
import os

def download_uieb():
    print("Downloading UIEB dataset...")
    
    # Create data folder
    os.makedirs("data", exist_ok=True)
    
    # Download link (use this exact URL)
    url = "https://irvlab.cs.umn.edu/sites/default/files/Underwater%20Image%20Enhancement%20Benchmark.zip"
    
    # Download file
    response = requests.get(url, stream=True)
    with open("data/uieb.zip", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print("Download complete!")

# Run it
download_uieb()

# Add to download_dataset.py
def extract_dataset():
    print("Extracting dataset...")
    
    with zipfile.ZipFile("data/uieb.zip", 'r') as zip_ref:
        zip_ref.extractall("data/")
    
    # Organize files
    raw_folder = "data/raw"
    ref_folder = "data/reference"
    
    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(ref_folder, exist_ok=True)
    
    # Move files to appropriate folders
    # (This will depend on the extracted structure)
    # Usually: raw images in one folder, reference in another
    
    print(f"Dataset extracted!")
    print(f"Raw images: {len(os.listdir(raw_folder))}")
    print(f"Reference images: {len(os.listdir(ref_folder))}")

extract_dataset()