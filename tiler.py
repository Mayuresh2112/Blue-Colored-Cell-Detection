import os
import cv2
import math

# ===========================
# Config
# ===========================
IMG_DIR = "C:/Users/mayur/Downloads/Task/Task/DATA/wdataset"     # your raw images (~5k x 5k)
LBL_DIR = "C:/Users/mayur/Downloads/Task/Task/DATA/wdataset"     # original YOLO labels
OUT_IMG_DIR = "images_tiled"
OUT_LBL_DIR = "labels_tiled"

TILE_SIZE = 1024   # size of each tile
OVERLAP = 256      # overlap between tiles
EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)


def yolo_to_xyxy(line, W, H):
    """YOLO (norm) -> pixel xyxy"""
    class_id, x, y, w, h = map(float, line.strip().split())
    x1 = (x - w / 2) * W
    y1 = (y - h / 2) * H
    x2 = (x + w / 2) * W
    y2 = (y + h / 2) * H
    return int(class_id), x1, y1, x2, y2


def xyxy_to_yolo(class_id, x1, y1, x2, y2, W, H):
    """pixel xyxy -> YOLO (norm)"""
    x = ((x1 + x2) / 2) / W
    y = ((y1 + y2) / 2) / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return f"{int(class_id)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


def get_tiles(W, H, tile=TILE_SIZE, overlap=OVERLAP):
    """Generate (x,y,w,h) for tiles covering the image"""
    xs = list(range(0, max(W - tile, 0) + 1, tile - overlap))
    ys = list(range(0, max(H - tile, 0) + 1, tile - overlap))
    if xs[-1] != W - tile:
        xs.append(W - tile)
    if ys[-1] != H - tile:
        ys.append(H - tile)
    return [(x, y, min(tile, W - x), min(tile, H - y)) for y in ys for x in xs]


def process_image(img_path, lbl_path, out_img_dir, out_lbl_dir):
    fname = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Skipping (cannot read): {img_path}")
        return
    H, W = img.shape[:2]

    # Load labels (if any)
    labels = []
    if os.path.exists(lbl_path):
        with open(lbl_path, "r") as f:
            for line in f.readlines():
                if len(line.strip()) > 0:
                    labels.append(yolo_to_xyxy(line, W, H))

    # Split into tiles
    tiles = get_tiles(W, H)
    for i, (x, y, w, h) in enumerate(tiles):
        tile_img = img[y:y + h, x:x + w]
        tile_name = f"{fname}_{i:03d}"
        out_img_path = os.path.join(out_img_dir, tile_name + ".png")
        out_lbl_path = os.path.join(out_lbl_dir, tile_name + ".txt")

        # Adjust labels for this tile
        new_labels = []
        for cls, x1, y1, x2, y2 in labels:
            # Check overlap with tile
            if x2 <= x or x1 >= x + w or y2 <= y or y1 >= y + h:
                continue
            # Clip box to tile
            nx1 = max(0, x1 - x)
            ny1 = max(0, y1 - y)
            nx2 = min(w, x2 - x)
            ny2 = min(h, y2 - y)
            if nx2 <= nx1 or ny2 <= ny1:
                continue
            new_labels.append(xyxy_to_yolo(cls, nx1, ny1, nx2, ny2, w, h))

        # Save tile + labels
        cv2.imwrite(out_img_path, tile_img)
        with open(out_lbl_path, "w") as f:
            f.write("\n".join(new_labels))


def main():
    imgs = [f for f in os.listdir(IMG_DIR) if os.path.splitext(f)[1].lower() in EXTS]
    print(f"Found {len(imgs)} images to tile...")
    for i, img_file in enumerate(imgs):
        img_path = os.path.join(IMG_DIR, img_file)
        lbl_path = os.path.join(LBL_DIR, os.path.splitext(img_file)[0] + ".txt")
        process_image(img_path, lbl_path, OUT_IMG_DIR, OUT_LBL_DIR)
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(imgs)}")

    print("✅ Done tiling all images.")


if __name__ == "__main__":
    main()
