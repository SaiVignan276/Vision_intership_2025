import os
import zipfile
import requests
from tqdm import tqdm
import shutil

# ========= CONFIG =========
BASE_DIR = r"C:\Users\DELL\desktop\yoloproject"
COCO_URL = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
# ==========================


def download_file(url, dest_path):
    """Download a file with progress bar"""
    print(f"\n⬇️ Downloading: {url}")
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as file, tqdm(
        desc=os.path.basename(dest_path),
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def extract_zip(file_path, extract_to):
    """Extract zip files safely"""
    print(f"📦 Extracting: {file_path}")
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Extracted to: {extract_to}")


def prepare_yolo_structure():
    """Ensure correct YOLO folder structure"""
    print("\n🗂️ Setting up YOLO folder structure...")
    folders = [
        "images/train",
        "images/val",
        "labels/train",
        "labels/val"
    ]
    for folder in folders:
        os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)
    print("✅ Folder structure ready!")


def move_coco_data():
    """Move COCO128 data to YOLO folders"""
    coco_path = os.path.join(BASE_DIR, "coco128")
    if not os.path.exists(coco_path):
        print("⚠️ COCO128 folder not found, skipping move.")
        return

    print("\n📁 Moving COCO128 (Detection) images and labels...")
    subsets = [("train2017", "train"), ("val2017", "val")]

    for subset_name, dest in subsets:
        img_src = os.path.join(coco_path, "images", subset_name)
        lbl_src = os.path.join(coco_path, "labels", subset_name)
        img_dest = os.path.join(BASE_DIR, f"images/{dest}")
        lbl_dest = os.path.join(BASE_DIR, f"labels/{dest}")

        for src, dest_folder in [(img_src, img_dest), (lbl_src, lbl_dest)]:
            if os.path.exists(src):
                for f in os.listdir(src):
                    shutil.move(os.path.join(src, f), dest_folder)
    print("✅ COCO128 detection data moved!")


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    os.chdir(BASE_DIR)
    prepare_yolo_structure()

    # === Download COCO128 ===
    coco_zip = os.path.join(BASE_DIR, "coco128.zip")
    if not os.path.exists(coco_zip):
        download_file(COCO_URL, coco_zip)
    extract_zip(coco_zip, BASE_DIR)

    # === Move files to unified structure ===
    move_coco_data()

    print("\n✅ All datasets are ready!")
    print(f"📁 Check inside: {BASE_DIR}\\images and {BASE_DIR}\\labels")


if __name__ == "__main__":
    main()