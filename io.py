# Install required packages
!pip install -q torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
!pip install -q diffusers transformers accelerate ftfy

# Import necessary libraries
import os
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageDraw
from diffusers import StableDiffusionImg2ImgPipeline
from google.colab import auth, drive

# Authenticate and mount Google Drive
auth.authenticate_user()
drive.mount('/content/drive', force_remount=True)

# Define input/output directories
INPUT_DIR = '/content/drive/My Drive/input-logos'
OUTPUT_DIR = '/content/drive/My Drive/output-logos'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Stable Diffusion pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "stabilityai/stable-diffusion-2-1"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=True
).to(device)

# Aesthetic template with enhanced prompts
AESTHETIC_TEMPLATE = """
Minimalist geometric logo design with {points} key elements, 
flat vector style, clean lines, symmetrical composition, 
vibrant colors on neutral background, professional corporate identity
"""

def generate_logo_variations(input_path, output_dir, num_variations=4):
    try:
        # Load and prepare base image
        init_image = Image.open(input_path).convert("RGB")
        init_image = init_image.resize((768, 768))
        filename = os.path.splitext(os.path.basename(input_path))[0]

        for i in range(num_variations):
            try:
                # Generate parameters
                points = int(np.random.choice([3, 4, 5]))  # Convert to native int
                prompt = AESTHETIC_TEMPLATE.format(points=points)
                strength = np.random.uniform(0.3, 0.6)
                guidance = np.random.uniform(7.5, 15.0)

                # Generate image
                result = pipe(
                    prompt=prompt,
                    negative_prompt="text, watermark, low quality, blurry, messy, noisy",
                    image=init_image,
                    strength=strength,
                    guidance_scale=guidance,
                    num_inference_steps=50
                ).images[0]

                # Save result
                output_path = os.path.join(output_dir, f'{filename}_variation_{i+1}.png')
                result.save(output_path)
                print(f'Saved variation {i+1} to {output_path}')

            except Exception as gen_error:
                print(f'Error generating variation {i+1}: {str(gen_error)}')
                continue

    except Exception as e:
        print(f'Error processing file {input_path}: {str(e)}')

# Process all logos
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(INPUT_DIR, filename)
        try:
            print(f'Processing {filename}...')
            generate_logo_variations(input_path, OUTPUT_DIR)
        except Exception as e:
            print(f'Fatal error processing {filename}: {str(e)}')
            continue

print('Processing completed!')