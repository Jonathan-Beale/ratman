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
input_image_path = "results/output_simon_gettyimages-1177643064-0169d811744c50e93e38eb67537162652733bb0a.jpg"

prompt = """Use the input image as an exact reference for the entire scene. 
Follow all Mediapipe-detected skeleton keypoints and joint angles for the main character precisely. 
Do NOT change limb positions, angles, or orientation. 
Do NOT reposition any characters, objects, props, or background elements. 
Keep the size, scale, perspective, and relative distances of all elements identical to the input image. 
The skeleton is only for pose guidance — do NOT show it. 

Replace only the main character with Batman in Jim Aparo’s comic book art style, keeping the exact same pose, facing, and position. 
Batman must appear in his **classic, iconic appearance** — the dark gray suit with the black bat symbol on his chest, blue cowl, cape, gloves, boots, and briefs, and a yellow utility belt. 
His face must be fully covered by the Batman cowl except for the lower jaw. 
Do NOT include any text, numbers, names, or logos on his suit, cape, or anywhere in the image. 
Do NOT mix Batman with any other character, version, or human — he must be 100% Batman, not hybrid, mutated, or stylized beyond Jim Aparo’s realistic comic look. 
Do NOT make Batman wear sports uniforms, armor, mechanical enhancements, or alternate versions from other universes. 

Render all other characters, objects, and the background exactly as in the input image, but fully in cartoon/comic style. 
Preserve the lighting, shadows, and perspective as in the input image. 
Ensure the final result looks like a hand-drawn comic panel with bold ink lines, expressive shading, and a fully cartoon environment, while maintaining perfect scene fidelity."""




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
