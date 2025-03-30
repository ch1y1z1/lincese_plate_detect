from rfdetr import RFDETRBase
import supervision as sv
import matplotlib.pyplot as plt
import typer
from PIL import Image
from typing import Optional


app = typer.Typer()


@app.command()
def predict_img(
    image_path: str = typer.Argument(..., help="Path to the image to predict"),
    weights: str = typer.Option("./output/checkpoint0007.pth", "-w", "--weights"),
    device: str = typer.Option("mps", "-d", "--device"),
    output_path: Optional[str] = typer.Option(None, "-o", "--output"),
):
    model = RFDETRBase(device=device, pretrain_weights=weights)
    image = Image.open(image_path)

    detections = model.predict(image, threshold=0.5)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    labels = [
        f"license plate: {confidence:.2f}" for confidence in detections.confidence
    ]

    annotated_image = image.copy()
    annotated_image = box_annotator.annotate(annotated_image, detections)
    annotated_image = label_annotator.annotate(annotated_image, detections, labels)

    if output_path:
        plt.imsave(output_path, annotated_image)
    else:
        plt.imshow(annotated_image)
        plt.show()


if __name__ == "__main__":
    app()
