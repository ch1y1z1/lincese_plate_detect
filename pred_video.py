import supervision as sv
from rfdetr import RFDETRBase
import typer
from paddleocr import PaddleOCR
from PIL import ImageDraw, ImageFont, Image
import numpy as np

app = typer.Typer()


@app.command()
def predict_video(
    source_path: str = typer.Argument(..., help="Path to the video to predict"),
    target_path: str = typer.Argument(..., help="Path to the output video"),
    weights: str = typer.Option("./output/checkpoint0007.pth", "-w", "--weights"),
    device: str = typer.Option("mps", "-d", "--device"),
    threshold: float = typer.Option(0.5, "-t", "--threshold"),
):
    model = RFDETRBase(device=device, pretrain_weights=weights)

    def callback(frame, index):
        detections = model.predict(frame, threshold=threshold)

        detected_text = []
        for xyxy in detections.xyxy:
            x1, y1, x2, y2 = map(int, xyxy)
            if x1 >= x2 or y1 >= y2:
                continue
            cropped_image = frame[y1:y2, x1:x2]
            if cropped_image.size == 0:
                continue

            ocr = PaddleOCR(lang="ch")
            results = ocr.ocr(cropped_image, det=False, cls=False)
            if results and results[0]:
                text, _ = results[0][0]
                detected_text.append(text.strip())
            else:
                detected_text.append("")

        annotated_frame = frame.copy()
        annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
        # annotated_frame = sv.LabelAnnotator().annotate(
        #     annotated_frame, detections, labels
        # )
        # if isinstance(annotated_frame, np.ndarray):
        annotated_frame = Image.fromarray(annotated_frame)
        draw = ImageDraw.Draw(annotated_frame)
        fontStyle = ImageFont.truetype(
            "/Users/chiyizi/Library/Fonts/MapleMono-CN-Medium.ttf",
            48,
            encoding="utf-8",
        )
        for text, confidence, (x1, y1, x2, y2) in zip(
            detected_text, detections.confidence, detections.xyxy
        ):
            draw.text(
                (int(x1), int(y1) - 50),
                f"{text} ({confidence:.2f})",
                font=fontStyle,
                fill=(255, 0, 0),
            )
        return np.array(annotated_frame)

    sv.process_video(
        source_path=source_path, target_path=target_path, callback=callback
    )


if __name__ == "__main__":
    app()
