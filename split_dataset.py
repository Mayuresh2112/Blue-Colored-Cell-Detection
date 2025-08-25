import os
import random
import shutil

# ===========================
# Config
# ===========================
IMG_DIR = "C:/Users/mayur/Downloads/Task/Task/DATA/images_tiled"
LBL_DIR = "C:/Users/mayur/Downloads/Task/Task/DATA/labels_tiled"
OUT_DIR = "C:/Users/mayur/Downloads/Task/Task/DATA/dataset_split"

SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}
EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

# ===========================
def make_dirs():
    for split in SPLIT_RATIOS.keys():
        os.makedirs(os.path.join(OUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, "labels", split), exist_ok=True)

def split_dataset():
    # list all images
    images = [f for f in os.listdir(IMG_DIR) if os.path.splitext(f)[1].lower() in EXTS]
    random.shuffle(images)

    n = len(images)
    n_train = int(SPLIT_RATIOS["train"] * n)
    n_val = int(SPLIT_RATIOS["val"] * n)
    # rest goes to test
    n_test = n - n_train - n_val

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train+n_val],
        "test": images[n_train+n_val:]
    }

    print(f"Total images: {n}")
    print(f" Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

    for split, files in splits.items():
        for img_file in files:
            stem = os.path.splitext(img_file)[0]
            lbl_file = stem + ".txt"

            src_img = os.path.join(IMG_DIR, img_file)
            src_lbl = os.path.join(LBL_DIR, lbl_file)

            dst_img = os.path.join(OUT_DIR, "images", split, img_file)
            dst_lbl = os.path.join(OUT_DIR, "labels", split, lbl_file)

            shutil.copy2(src_img, dst_img)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)
            else:
                # create empty txt if missing
                open(dst_lbl, "w").close()

    print("✅ Split complete. Files saved under:", OUT_DIR)


if __name__ == "__main__":
    make_dirs()
    split_dataset()

