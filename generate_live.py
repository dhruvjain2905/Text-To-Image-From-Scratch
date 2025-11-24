import torch
from conditional_ldm import TextToImage
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Path to your trained model
MODEL_PATH = 'models/cldm_celebA2New_diffusionresumed/unet/88641'

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_model(model_path):
    print("Loading model from:", model_path)
    model = TextToImage.load(model_path)
    model.to(device)
    model.eval()
    return model

def generate_and_display(model, prompt, inf_steps=500, n_images=1, stream_freq=50):
    print(f"\nGenerating image for prompt: '{prompt}'\n")

    prompts = [prompt for _ in range(n_images)]
    fig, ax = plt.subplots()
    plt.ion()  # interactive mode on
    plt.show()

    latest_img = None

    for imgs, details in model.run_pipeline(
        prompts,
        num_inference_steps=inf_steps,
        stream_freq=stream_freq
    ):
        step = details.get("t", 0) + 1
        total = details.get("total_t", inf_steps)
        img = imgs[0]
        if isinstance(img, np.ndarray):
            img = Image.fromarray((img * 255).astype(np.uint8))

        latest_img = img

        ax.clear()
        ax.imshow(img)
        ax.set_title(f"Step {step} / {total}")
        ax.axis("off")
        plt.pause(0.001)  # tiny pause to update frame

    plt.ioff()
    plt.show()

    return latest_img


if __name__ == "__main__":
    prompt = input("Enter a text prompt: ")
    model = load_model(MODEL_PATH)

    final_img = generate_and_display(model, prompt, inf_steps=500, stream_freq=25)

    # Save final image
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"outputs/{prompt.replace(' ', '_')}_{timestamp}.png"
    final_img.save(filename)

    print(f"\n✅ Saved final image to: {filename}")
