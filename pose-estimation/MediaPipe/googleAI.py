import os
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

# api key for authentication
client = genai.Client(api_key="putyourapikeyhere") # make sure the api key has access to Gemini API, its somehting weird with google accounts

# check if the output folder exists, if not create it
output_folder = "AIgenerated"
os.makedirs(output_folder, exist_ok=True)

# put your input image path here
input_image_path = "results/output_istockphoto-179137240-612x612.jpg" 

prompt = (
    "Use the human pose skeleton keypoints and joints in the image I sent, "
    "replacing the character with Batman while keeping the same pose. "
    "Stylize it in Jim Aparo’s Batman comic book art style."
)

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
        output_path = os.path.join(output_folder, f"generated_image_{i+1}.png")
        generated_image.save(output_path)
        print(f"✅ Saved: {output_path}")
        generated_image.show()