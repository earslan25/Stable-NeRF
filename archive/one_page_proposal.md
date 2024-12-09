## One Page Proposal

What is the exact problem you are trying to solve? 
- We want to address 3D reconstruction using a combination of stable diffusion and NeRF. By inserting a model capable of a 3D representation within stable diffusion, we hope to create a more robust 3D reconstruction. Currently, most 3D reconstruction techniques fail when generalizing to out of distribution data. Methods like NeRF and 3D Gaussian Splatting primarily aim to reconstruct a single scene with a single model, rather than many scenes with a single model. Other approaches to generalizable one-shot 3D reconstruction either do not have any form of 3D representation which makes them purely hallucinate and lose 3D consistency, or they cannot generate robust views. With our method, we want to achieve 3D consistent novel view synthesis that works on out of distribution data. For this, our approach will train a neural field in the latent space of pretrained Stable Diffusion. We will use the neural field's outputs as 3D feature maps to condition our U-Net when performing diffusion. Across scenes, this will combine the powerful diffusion network with a 3D forward map that is volume rendering.

What prior works have tried to address it? 
- [Reconstructive Latent-Space Neural Radiance Fields for Efficient 3D Scene Representations](https://arxiv.org/pdf/2310.17880)
- [Diffusion with Forward Models: Solving Stochastic Inverse Problems Without Direct Supervision](https://diffusion-with-forward-models.github.io/diffusion-forward-paper.pdf)
- [Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/pdf/2303.11328)

Reconstructive Latent-Space Neural Radiance Fields for Efficient 3D Scene Representations uses the pretrained encoder and decoder components of Stable Diffusion to utilize learned 2D priors when reconstructing 3D views. Similar to our approach of utilizing a non-RGB domain neural field, they train a neural field in the latent space of the pretrained autoencoder between the encoder and decoder parts to reconstruct a 3D scene.

Diffusion with Forward Models: Solving Stochastic Inverse Problems Without Direct Supervision introduces a forward map to diffusion tasks, combining both. Especially when integrated with pixelNeRF, it can achieve generalizable 3D reconstruction by using an explicit volume rendering forward map as well as diffusion.

Zero-1-to-3: Zero-shot One Image to 3D Object conditions a pretrained Stable Diffusion network to achieve novel view synthesis without any 3D reconstruction. It utilizies the 2D priors of Stable Diffusion to hallucinate an object from a different view given the baseline image, the baseline camera parameters, and the target camera parameters for the novel view. 

How is your approach different? 
- Reconstructive Latent-Space Neural Radiance Fields paper only uses the autoencoder part from Stable Diffusion, whereas we will be utizing the U-Net too. This means that their approach does not benefit from diffusion. Furthermore, this paper does not aim to generalize, it focuses on single scene reconstruction and rendering efficiency for NeRFs.
- Diffusion with Forward Models does not utilize pretrained Stable Diffusion weights (encoder, decoder, U-Net) for the 2D priors, which we believe will be a powerful tool. 
- Zero-1-to-3 does not have an explicit forward map or any kind of 3D representation. We will be training a latent space neural field to combine our 3D forward map with Stable Diffusion, providing 3D conditions as well rather than purely 2D ones.

What data will you use? 
- We will start with the dataset Zero-1-to-3 uses, which is a syntethic dataset of digitally modeled objects. Then, we will move onto real world datasets which Diffusion with Forward Models uses to demonstrate the performance of its integration with pixelNeRF. These datasets include real world fire hydrants and rooms.

What compute will you use? 
- Oscar (Chia)
- Possibly Kaggle, dual T4's

What existing codebases will you use?
- Stable diffusion for pretrained model/architecture
- tiny-cuda-nn/instant-ngp for fused MLP operations and hash encoding