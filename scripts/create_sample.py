# Create sample dataset for faster testing
import shutil
import os

def create_sample_dataset(sample_size=50):
    """Create smaller dataset for quick testing"""
    
    sample_folder = "data/sample"
    os.makedirs(f"{sample_folder}/raw", exist_ok=True)
    os.makedirs(f"{sample_folder}/reference", exist_ok=True)
    
    # Copy first 'sample_size' images
    raw_files = os.listdir("data/raw")[:sample_size]
    ref_files = os.listdir("data/reference")[:sample_size]
    
    for file in raw_files:
        src = os.path.join("data/raw", file)
        dst = os.path.join(f"{sample_folder}/raw", file)
        shutil.copy(src, dst)
    
    for file in ref_files:
        src = os.path.join("data/reference", file)
        dst = os.path.join(f"{sample_folder}/reference", file)
        shutil.copy(src, dst)
    
    print(f"Created sample dataset with {sample_size} images")


if __name__ == "__main__":
    create_sample_dataset(50)