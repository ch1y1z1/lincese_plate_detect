import supervision as sv
from rfdetr import RFDETRBase
import typer

app = typer.Typer()


def predict_video(
    source_path: str = typer.Argument(..., help="Path to the video to predict"),
    target_path: str = typer.Argument(..., help="Path to the output video"),
    weights: str = typer.Option("./output/checkpoint0007.pth", "-w", "--weights"),
    device: str = typer.Option("mps", "-d", "--device"),
):
    model = RFDETRBase(device=device, pretrain_weights=weights)

    def callback(frame, index):
        detections = model.predict(frame, threshold=0.5)

        labels = [
            f"license plate: {confidence:.2f}" for confidence in detections.confidence
        ]
        annotated_frame = frame.copy()
        annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
        annotated_frame = sv.LabelAnnotator().annotate(
            annotated_frame, detections, labels
        )
        return annotated_frame

    sv.process_video(
        source_path=source_path, target_path=target_path, callback=callback
    )
