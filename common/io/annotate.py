from pathlib import Path

from PIL import Image, ImageDraw


def draw_boxes_on_image(
    image_path: Path, output_path: Path, preds, confidence_threshold=0.8
):
    image = Image.open(image_path).convert("RGB")

    draw = ImageDraw.Draw(image)

    for obj in preds:
        if obj.confidence >= confidence_threshold:
            label_text = f"{obj.label}: {obj.confidence}"
            box = obj.bounding_box
            if obj.label == "kite":
                draw.rectangle(box, outline="yellow", width=2)
            elif obj.label == "person":
                draw.rectangle(box, outline="black", width=2)
            else:
                draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], box[1] - 10), label_text, fill="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
