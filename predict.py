from rfdetr import RFDETRBase
import supervision as sv
import matplotlib.pyplot as plt
import typer
from PIL import Image, ImageDraw, ImageFont
from typing import Optional
import numpy as np
from paddleocr import PaddleOCR
from matplotlib import font_manager

# 设置中文字体，这里使用微软雅黑作为示例
font_path = (
    "/Users/chiyizi/Library/Fonts/MapleMono-CN-Medium.ttf"  # 请根据实际情况修改字体路径
)
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


ocr = PaddleOCR(lang="ch")


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

    detected_text = []
    for xyxy in detections.xyxy:
        x1, y1, x2, y2 = map(int, xyxy)
        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_array = np.array(cropped_image)
        results = ocr.ocr(cropped_array, det=False, cls=False)
        text, _ = results[0][0]
        detected_text.append(text.strip())

    annotated_image = image.copy()
    annotated_image = box_annotator.annotate(annotated_image, detections)

    draw = ImageDraw.Draw(annotated_image)
    fontStyle = ImageFont.truetype(
        "/Users/chiyizi/Library/Fonts/MapleMono-CN-Medium.ttf",
        48,
        encoding="utf-8",
    )
    for text, confidence, (x1, y1, x2, y2) in zip(
        detected_text, detections.confidence, detections.xyxy
    ):
        draw.text(
            (x1, y1 - 50),
            f"{text} ({confidence:.2f})",
            font=fontStyle,
            fill=(255, 0, 0),
        )

    if output_path:
        plt.imsave(output_path, annotated_image)
    else:
        plt.imshow(annotated_image)
        plt.show()


if __name__ == "__main__":
    app()
