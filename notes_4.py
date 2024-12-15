
import torch
import matplotlib.pyplot as plt

device = "cpu"

def load_image(path):
    pred = torch.load(path, map_location=torch.device(device))
    latents_pred = pred.view(1, 64, 64, 4).detach().numpy()
    return latents_pred

# bad image
images = [
    # load_image(f"visualizations/notes_9/pred_0400_0000.pt"),
    # load_image(f"visualizations/notes_9/pred_0400_0001.pt"),
    # load_image(f"visualizations/notes_9/pred_0400_0002.pt"),

    load_image(f"visualizations/notes_10/pred_0500_0000.pt"),
    load_image(f"visualizations/notes_10/pred_0500_0001.pt"),
]
# # good image
# image_1 = load_image(f"visualizations/notes_10/pred_0500_0000.pt")

fig, axes = plt.subplots(4, len(images), figsize=(5, 10))

def image_column(col, image):
    axes[0,col].imshow(image[0][:,:,0])
    axes[1,col].imshow(image[0][:,:,1])
    axes[2,col].imshow(image[0][:,:,2])
    axes[3,col].imshow(image[0][:,:,3])

for i, image in enumerate(images):
    image_column(i, image)

plt.show()
