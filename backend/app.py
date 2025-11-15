import base64
import os
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from perceptron import configure, perceive, image, text
from PIL import Image, ImageDraw
from perceptron.pointing.geometry import scale_box_to_pixels

configure(
    provider="perceptron",
    api_key=os.getenv("PERCEPTRON_API_KEY", "<your_API_key>"),
)
app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (images) for frontend
app.mount("/static", StaticFiles(directory="."), name="static")

@app.post("/perceive")
async def perceive_image(
    prompt: str = Form(...),
    file: UploadFile = File(...)
):
    # Save uploaded file temporarily
    contents = await file.read()
    image_path = f"temp_{file.filename}"
    with open(image_path, "wb") as f:
        f.write(contents)

    # Run perception
    @perceive(model="isaac-0.1", expects="box", allow_multiple=True)
    def detect(frame_path):
        scene = image(frame_path)
        return scene + text(prompt)

    result = detect(image_path)

    # Open image and draw boxes
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for idx, box in enumerate(result.points or []):
        scaled_box = scale_box_to_pixels(box, width=img.width, height=img.height)
        top_left, bottom_right = scaled_box.top_left, scaled_box.bottom_right
        draw.rectangle([top_left.x, top_left.y, bottom_right.x, bottom_right.y], outline="lime", width=3)

    # Save annotated image temporarily
    annotated_path = f"annotated_{file.filename}"
    img.save(annotated_path)

    # Convert image to Base64
    with open(annotated_path, "rb") as f:
        encoded_img = base64.b64encode(f.read()).decode("utf-8")

    # Return Base64 image in JSON
    return {
        "text": result.output_text if hasattr(result, "output_text") else str(result),
        "annotated_image": f"data:image/png;base64,{encoded_img}"
    }