import tkinter as tk
import requests
import base64
from PIL import Image, ImageTk
import time
import threading
import io
import tkinter.ttk as ttk

# API endpoints
INPUT_API_ENDPOINT = "http://127.0.0.1:5000/text_to_image"
IMAGE_API_ENDPOINT = "http://127.0.0.1:5000/jobs/"

recycle_i = None
recycle_j = None

N_ROWS = 2
N_COLS = 2
N_IMAGES = N_ROWS * N_COLS
model_options = [
    
    "models/cldm_celebA2New_diffusionresumed/unet/88641"
    
    ]

def reset_recycle_imagebox():
    global recycle_i, recycle_j
    recycle_i = None
    recycle_j = None

def run_diffusion_api(input_text, model_path, n_images=N_IMAGES):
    response = requests.post(INPUT_API_ENDPOINT, json={"prompt": input_text, "n_inf_steps": 800, "stream_freq": 50, "n_images": n_images, "model_path": model_path})
    return response
 
# Function to make the initial API call
def make_input_api_call(input_text, n_images=N_IMAGES, reset=True):
    if reset:
        reset_recycle_imagebox()
    model_path = dropdown.get() 
    response = run_diffusion_api(input_text, model_path, n_images)
    textbox_value.set("")
    print(response)
    if response.ok:
        job_id = response.json()["job_id"]
        poll_images(job_id)
    else:
        print("Error making input API call.")

# Function to poll for images
def poll_images(job_id):
    job_endpoint = IMAGE_API_ENDPOINT + job_id
    def poll_loop():
        while True:
            response = requests.get(job_endpoint)
            if response.ok:
                response = response.json()
                finished = response.get("finished", False)
                if "response" in response:
                    try:
                        images = [r["img"] for r in response["response"]]
                        # disc_scores = [r.get("d", None) for r in response["response"]]
                        textbox_value.set(f'Reverse Diffusion Steps: {response["response"][0]["t"]} / {response["response"][0]["total_t"]}')
                        # Use the main thread to update the images
                        root.after(0, update_images, images)
                    except Exception as e:
                        print(f"Exception in poll_loop: {e}")
                        print(response.keys())
                        continue
                if finished:
                    break
                
            else:
                print("Error polling for images.")
            time.sleep(1)

    thread = threading.Thread(target=poll_loop)
    thread.daemon = True  # Ensure the thread exits when the main program does
    thread.start()

def update_image(row_i, row_j, image):
    photo = ImageTk.PhotoImage(image)
    image_labels[row_i][row_j].configure(image=photo)
    image_labels[row_i][row_j].image = photo  
    

def update_images(images):
    if recycle_i is None:
        for i in range(N_ROWS):
            for j in range(N_COLS):
                idx = i * 2 + j
                if idx < len(images):
                    image_data = base64.b64decode(images[idx])
                    image_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
    
                    update_image(i, j, image_pil)
                    
                else:
                    image_labels[i][j].configure(image=None)
    else:
        update_image(recycle_i, recycle_j, images[0])
    root.update_idletasks()

def recycle_image(i, j):
    global recycle_i, recycle_j
    recycle_i = i
    recycle_j = j
    make_input_api_call(input_text.get(), 1, False)

g1 = [f"jobs/old_man_gray_hair_cfaf/00_{(i + 25):04d}.png" for i in range(0, 500, 25)]
g2 = [f"jobs/old_man_gray_hair_cfaf/01_{(i + 25):04d}.png" for i in range(0, 500, 25)]
g3 = [f"jobs/old_man_gray_hair_cfaf/05_{(i + 25):04d}.png" for i in range(0, 500, 25)]
g4 = [f"jobs/old_man_gray_hair_cfaf/07_{(i + 25):04d}.png" for i in range(0, 500, 25)]
index = 0

def folder_imgs():

    def display_folder_imgs():
        index = 0
        while (index < 21):
            i1 = Image.open(g1[index])
            i2 = Image.open(g2[index])
            i3 = Image.open(g3[index])
            i4 = Image.open(g3[index])
            update_image(0, 0, i1)
            update_image(0, 1, i2)
            update_image(1, 0, i3)
            update_image(1, 1, i4)
            index += 1
            textbox_value.set(f'Reverse Diffusion Steps: {index*25} / 500')
            time.sleep(0.5)

   

    thread = threading.Thread(target=display_folder_imgs)
    thread.daemon = True  # Ensure the thread exits when the main program does
    thread.start()
                        



if __name__ == "__main__":
    import sys

    prompt = None
    if len(sys.argv) > 1:
        prompt = sys.argv[1]

    if prompt is None:
        # Create the main window
        root = tk.Tk()
        root.title("Diffusion!")

        dropdown_var = tk.StringVar()
        dropdown_var.set(model_options[0])  # Set the default value
        max_length = max(len(option) for option in model_options)

        dropdown = ttk.Combobox(root, textvariable=dropdown_var, values=model_options, state="readonly")
        dropdown.grid(row=0, column=0, padx=5, pady=5, sticky="w")


        # Create the input text box
        input_text = tk.StringVar()
        input_entry = tk.Entry(root, textvariable=input_text)

        input_entry.grid(row=1, column=0, padx=5, pady=5)

        # Create the button
        # submit_button = tk.Button(root, text="Submit", command=lambda: make_input_api_call(input_text.get()))
        submit_button = tk.Button(root, text="Submit", command=lambda: folder_imgs())
        submit_button.grid(row=1, column=1, padx=5, pady=5)

        # Create the image grid
        image_labels = [[tk.Label(root) for _ in range(N_COLS)] for _ in range(N_ROWS)]
        for i in range(N_ROWS):
            for j in range(N_COLS):
                image_labels[i][j].grid(row=i+2, column=j, padx=5, pady=5)
                image_labels[i][j].bind("<Button-1>", lambda event, i=i, j=j : recycle_image(i, j))
        textbox_value = tk.StringVar(value="")
        textbox = tk.Label(root, textvariable=textbox_value)
        textbox.grid(row=N_ROWS + 2, column=N_COLS//2, columnspan=2, padx=5, pady=5)
        # Start the main event loop
        root.mainloop()

    else:
        # Run API
        response = run_diffusion_api(prompt, model_options[0], 16)
        job_id = response.json()["job_id"]
        print(job_id)