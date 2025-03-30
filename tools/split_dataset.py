import supervision as sv

input_dir = "./ccpd_all"
input_json = "./ccpd_all/instances_train.json"
output_dir = "./data"

ds = sv.DetectionDataset.from_coco(
    images_directory_path=input_dir,
    annotations_path=input_json,
)

ds_train, ds_val = ds.split(0.8)

ds_train.as_coco(
    images_directory_path=f"{output_dir}/train",
    annotations_path=f"{output_dir}/train/_annotations.coco.json",
)

ds_val.as_coco(
    images_directory_path=f"{output_dir}/valid",
    annotations_path=f"{output_dir}/valid/_annotations.coco.json",
)