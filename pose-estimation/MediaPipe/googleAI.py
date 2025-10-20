import os
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

# api key for authentication
client = genai.Client(api_key="apikey")

# check if the output folder exists, if not create it
output_folder = "AIgenerated"
os.makedirs(output_folder, exist_ok=True)

# get the next available image number
existing_files = [f for f in os.listdir(output_folder) if f.startswith("generated_image_") and f.endswith(".png")]
if existing_files:
    # find the largest number currently used
    existing_numbers = [int(f.split("_")[2].split(".")[0]) for f in existing_files]
    next_index = max(existing_numbers) + 1
else:
    next_index = 1

# put your input image path here
input_image_path = "results/Output_20251020_115344.png"

prompt = """
IMPORTANT: DO NOT ZOOM IN OR ZOOM OUT. DO NOT CROP OR RECENTER. THE OUTPUT MUST MATCH THE INPUT EXACTLY IN FRAMING AND COMPOSITION.

Use the input image as an exact, pixel-level reference for the entire scene. 
The output must be a one-to-one visual replacement — the framing, camera distance, crop, and field of view must remain absolutely identical to the input. 
The output image must perfectly align with the input when overlaid, with no camera movement, zooming, recentering, or reframing of any kind. 
Do NOT add borders, padding, or background extensions. The composition must match at the pixel level.

Follow all Mediapipe-detected skeleton keypoints and joint angles for the main character precisely and invisibly. 
The skeleton and its keypoints are for pose guidance only — they must NEVER appear, be drawn, or be visible in any form in the final image. 

Do NOT alter or shift the position, scale, or orientation of any body parts. 
Every limb angle, joint rotation, and pose detail must exactly match the input image with no deviation. 
Keep all characters, props, and background elements in the exact same position, direction, scale, and perspective as in the input image.

Replace only the main character with Batman in Jim Aparo’s comic book style, maintaining the same body proportions, posture, facing, and silhouette as the original. 
Do NOT add or modify any costume details beyond this description. 
Do NOT include any text, labels, or logos anywhere.

Render the entire scene in a unified, Jim Aparo–style comic aesthetic: bold line art, inked outlines, and expressive shading. 
Maintain the same lighting, shadows, and perspective as the input photo. 
Every object and background element must remain identical in placement and proportion but restyled into comic art. 
The final image must look like a single cohesive comic panel, without any visual overlays, outlines, skeletons, or guides visible.
"""




# load the input image
image = Image.open(input_image_path)

# send the request to the Gemini API
response = client.models.generate_content(
    model="gemini-2.5-flash-image",
    contents=[prompt, image],
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(aspect_ratio="16:9"),
    ),
)

# save the photo to the output folder
for i, part in enumerate(response.parts):
    if part.inline_data:
        generated_image = Image.open(BytesIO(part.inline_data.data))
        output_path = os.path.join(output_folder, f"generated_image_{next_index + i}.png")
        generated_image.save(output_path)
        print(f"✅ Saved: {output_path}")
        generated_image.show()
