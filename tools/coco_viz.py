import supervision as sv
import matplotlib.pyplot as plt

ds = sv.DetectionDataset.from_coco(
    images_directory_path="../ccpd_coco",
    annotations_path="../ccpd_coco/instances_train.json",
)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_images = []
for i in range(4):
    _, image, annotations = ds[i]

    labels = [ds.classes[class_id] for class_id in annotations.class_id]

    annotated_image = image.copy()
    annotated_image = box_annotator.annotate(annotated_image, annotations)
    annotated_image = label_annotator.annotate(annotated_image, annotations, labels)
    annotated_images.append(annotated_image)

grid = sv.create_tiles(
    annotated_images,
    grid_size=(2, 2),
    single_tile_size=(800, 800),
    tile_padding_color=sv.Color.WHITE,
    tile_margin_color=sv.Color.WHITE,
)

plt.imshow(grid)
