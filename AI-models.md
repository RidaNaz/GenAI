# Generative AI Models and Applications:

#### 1. Generative Adversarial Networks (GANs):

- Two networks ***(generator and discriminator)*** compete to generate realistic data.
- **Applications:**   Image generation, deepfakes, video synthesis, style transfer

#### 2. Variational Autoencoders (VAEs):

- Compress and reconstruct data, generating new samples from latent space.
- **Applications:**   Image reconstruction, anomaly detection, data compression

#### 3. [Diffusion Models:](/AI-models.md#diffusion-models)

-  Iteratively refine random noise to create high-quality images.
- **Applications:**   High-quality image generation, art creation, video synthesis

#### 4. Large Language Models (LLMs):

- Large language models like GPT that generate coherent text.
- **Applications:**   Text generation, chatbots, summarization, translation (e.g., GPT-4)

#### 5. Autoregressive Models:

- Generate data by predicting the next element (e.g., pixels, words) based on previous ones.
- **Applications:**   Audio generation, pixel-level image generation (e.g., PixelCNN)

#### 6. Neural Style Transfer:

- Combines content and style from different images to create new visuals.
- **Applications:**   Art generation by blending content and style from different images


## Diffusion Models

### Stable Diffusion Models:

- Stable Diffusion builds on the Latent Diffusion Models (LDMs), a more efficient type of diffusion model that operates in a compressed latent space rather than directly on pixel data.
- **Applications:** Art Creation, Image Editing, Design, AI assisted creativity
- **How It Works:**
    - ***Encoding:*** Converts images into a lower-dimensional latent space.
    - ***Diffusion Process:*** Refines random noise into coherent images within the latent space.
    - ***Decoding:*** Converts the refined latent representation back into high-resolution images.


### Latent Diffusion Models (LDMs):

- A type of diffusion model that operates in a latent space, a compressed version of the original data, which makes the model more efficient and faster.
- **Applications:**  Image generation, Data Compression
- **How It Works:**
    - ***Encoding:*** Compresses data into a latent space.
    - ***Diffusion Process:*** Adds and removes noise within this compressed space.
    - ***Decoding:*** Transforms the latent data back into the original or high-resolution form.

### Denoising Diffusion Probabilistic Models (DDPMs):

- A class of generative models that gradually adds noise to data during training and then learns to reverse the process to generate new data.
- **Applications:** Image generation, data synthesis, Audio and Video Synthesis
- **How It Works:**
    - ***Forward Process:*** Data is progressively corrupted by adding Gaussian noise over several steps.
    - ***Reverse Process:*** A neural network learns to reverse this noise addition, progressively refining the data from noise to a coherent sample.

### Score-Based Generative Models (SDEs):

- A type of generative model that uses Stochastic Differential Equations (SDEs) to generate data by modeling a diffusion process.
- **Applications:** Image generation, high-quality data augmentation
- **How It Works:**
    - ***Forward Process:*** Adds noise to data using SDEs, converting it into a noise distribution.
    - ***Reverse Process:*** The model estimates the score function to iteratively denoise and generate new data samples.

