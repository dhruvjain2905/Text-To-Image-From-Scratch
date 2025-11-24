# Text-To-Image-From-Scratch
A full training and inference pipeline for generating images from text using a conditional latent diffusion model. 


To run, make sure you cloned the repository with the "models" directory containing the pre-trained models. After that, install all packages from requirements.txt and simply run

`python generate_live.py`

This is a full training pipeline to implement a conditional latent diffusion model from scratch. It is trained on the CelebA dataset, with 200,000 images of faces combined with a set of captions generated from the columns of the CelebA dataset. 

Essentially, we first train a variational autoencoder (VAE), that is able to take our images and encode them into a latent space. Then we use a transformer based model called CLIP that takes the text and image pairs, and basically encode them in a shared embedding space maximizing the cosine similarity between the correct pairs, and minimizing it with the ones that don't match. 

Then finally, we use a conditional diffusion model, that uses a UNet is trained to take out noise and reconstruct the orignal image. But we are also able to guide that diffusion using the embeddings from CLIP and the text prompt. 

Here, for example are the images generated, when you type in "man", versus when you type "woman". Obviously, it is not perfect 


(Need to show the images here)

![alt text](project/outputs/smiling_woman_20251111_220155.png)
