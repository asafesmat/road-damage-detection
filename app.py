import os
from collections import Counter

import gradio as gr
import pandas as pd
from ultralytics import YOLO

MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
model = YOLO(MODEL_PATH)


def detect(image, conf):
    if image is None:
        raise gr.Error("Upload a road image first.")

    # A PIL image is treated as RGB by Ultralytics (a raw numpy array would be treated as BGR).
    result = model.predict(image, imgsz=640, conf=conf, verbose=False)[0]

    annotated = result.plot()[:, :, ::-1]   # plot() returns BGR, Gradio expects RGB

    detections = [
        (result.names[int(c)], float(s))
        for c, s in zip(result.boxes.cls, result.boxes.conf)
    ]
    table = pd.DataFrame(
        [{"damage type": name, "confidence": round(score, 2)} for name, score in detections],
        columns=["damage type", "confidence"],
    )

    if detections:
        counts = Counter(name for name, _ in detections)
        summary = ", ".join(f"{name}: {n}" for name, n in counts.most_common())
    else:
        summary = "No damage detected at this confidence threshold."

    return annotated, summary, table


demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="pil", label="Road image"),
        gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold"),
    ],
    outputs=[
        gr.Image(label="Detections"),
        gr.Textbox(label="Summary"),
        gr.Dataframe(label="Detected damage"),
    ],
    title="Road Damage Detection (YOLO26s)",
    description=(
        "Upload a road image to localize cracks, potholes and other pavement damage. "
        "Classes: alligator, block, crack, longitudinal, pothole, transverse (the 'edge' class has too few "
        "examples to be reliable). Research demo trained on a public Roboflow dataset; "
        "it is not a safety or inspection tool."
    ),
)

if __name__ == "__main__":
    demo.launch()
