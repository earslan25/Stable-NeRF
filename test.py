
# stable diffusion-nerf pipeline

# NOTE
    # encode using stable diffusion encoder
    # train on the nerf
    # run through the u-net
        # somehow incorporate pose encodings
    # decode

    # get this to work any way possible

# NOTE
    # end goals
        # get a 3d visualization of the encoded nerf

# NOTE
    # intermediate steps

    # get stable diffusion to generate a simple image
    # train a simple nerf to give a novel view



# NOTE: get stable diffusion to generate a simple image

import torch

def test_stable_diffusion():
    """
    Generate a simple image.
    """

    input_image = torch.rand((3,512,512))

    print(input_image.shape)


if __name__ == "__main__":
    test_stable_diffusion()
