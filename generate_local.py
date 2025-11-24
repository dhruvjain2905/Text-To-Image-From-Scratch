import torch
from conditional_ldm import TextToImage
from PIL import Image
import numpy as np

# Path to your trained model
MODEL_PATH = 'models/cldm_celebA2New_diffusionresumed/unet/88641'

# Device (use MPS for Mac, else 'cuda' or 'cpu')
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_model(model_path):
    print("Loading model from:", model_path)
    model = TextToImage.load(model_path)
    model.to(device)
    model.eval()
    return model

def generate_image(model, prompt, inf_steps=500, n_images=1):
    print(f"Generating image for prompt: '{prompt}'")
    prompts = [prompt for _ in range(n_images)]
    
    # Run the diffusion pipeline once
    for imgs, details in model.run_pipeline(prompts, num_inference_steps=inf_steps, stream_freq=inf_steps):
        pass  # Run until the last output (the final image)

    return imgs

if __name__ == "__main__":
    prompt = input("Enter a text prompt: ")
    model = load_model(MODEL_PATH)
    images = generate_image(model, prompt, inf_steps=500, n_images=1)

    # Take the first image
    img = images[0]
    if isinstance(img, np.ndarray):
        img = Image.fromarray((img * 255).astype(np.uint8))
    
    img.show(title=prompt)  # Opens default image viewer