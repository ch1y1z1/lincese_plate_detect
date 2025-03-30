import os
import json
import shutil

INFO = {}
LICENSES = []
CATEGORIES = [{"id": 1, "name": "license plate", "supercategory": "common-objects"}]
IMAGES = []
ANNOTATIONS = []

CCPD_dir = "./data"
out_dir = "./ccpd_all"
os.makedirs(out_dir, exist_ok=True)

dataset_percentage = 0.2

img_id = 0
annotation_id = 0


def img2coco(file_path):
    global img_id
    global annotation_id

    width, height = 720, 1160

    file_name = os.path.basename(file_path)

    image_info = {
        "id": img_id,
        "file_name": file_name,
        "width": width,
        "height": height,
        "license": 1,
        "coco_url": "",
        "flickr_url": "",
    }

    try:
        file_name_parts = file_name.split("-")
        bbox = file_name_parts[2].split("_")
        x1, y1 = map(int, bbox[0].split("&"))
        x2, y2 = map(int, bbox[1].split("&"))
    except Exception:
        print(f"Invalid file name: {file_name}")
        return

    annotation_info = {
        "id": annotation_id,
        "image_id": img_id,
        "category_id": 1,
        "segmentation": [],
        "area": (x2 - x1) * (y2 - y1),
        "bbox": [x1, y1, x2 - x1, y2 - y1],
        "iscrowd": 0,
    }

    IMAGES.append(image_info)
    img_id += 1

    ANNOTATIONS.append(annotation_info)
    annotation_id += 1

    return


def main():
    for subdir in os.listdir(CCPD_dir):
        subdir_path = os.path.join(CCPD_dir, subdir)
        if os.path.isdir(subdir_path) and subdir.startswith("ccpd"):
            files = [f for f in os.listdir(subdir_path) if f.endswith(".jpg")]
            num = int(len(files) * dataset_percentage)
            for file in files[:num]:
                file_path = os.path.join(subdir_path, file)
                img2coco(file_path)
                shutil.copy(file_path, out_dir)

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": IMAGES,
        "annotations": ANNOTATIONS,
    }

    with open(os.path.join(out_dir, "instances_train.json"), "w") as f:
        json.dump(coco_output, f)


if __name__ == "__main__":
    main()
