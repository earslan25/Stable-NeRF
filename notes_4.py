
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


device = "cpu"

def load_image(path):
    pred = torch.load(path, map_location=torch.device(device))
    latents_pred = pred.view(1, 64, 64, 4).detach().numpy()
    return latents_pred

def load_normal_image(path):
    img = mpimg.imread(path)
    return img

# bad image
images = [
    # load_image(f"visualizations/notes_9/pred_0010_0000.pt"),
    # load_image(f"visualizations/notes_9/pred_0020_0000.pt"),
    # load_image(f"visualizations/notes_9/pred_0050_0000.pt"),
    # load_image(f"visualizations/notes_9/pred_0100_0000.pt"),

    # load_image(f"visualizations/notes_9/pred_0400_0001.pt"),
    # load_image(f"visualizations/notes_9/pred_0400_0002.pt"),


    # load_image(f"visualizations/notes_10/pred_0500_0000.pt"),
    # load_image(f"visualizations/notes_10/pred_0500_0001.pt"),

    # load_image(f"visualizations/notes_10/pred_0010_0000.pt"),
    # load_image(f"visualizations/notes_10/pred_0020_0000.pt"),
    # load_image(f"visualizations/notes_10/pred_0050_0000.pt"),
    # load_image(f"visualizations/notes_10/pred_0100_0000.pt"),


    # load_image(f"visualizations/notes_11/pred_0500_0000.pt"),
    # load_image(f"visualizations/notes_11/pred_0500_0001.pt"),
    # load_image(f"visualizations/notes_11/pred_0500_0002.pt"),


    # load_image(f"visualizations/notes_13/target_pred_0000.pt"),
    # load_image(f"visualizations/notes_13/target_pred_0001.pt"),
    # load_image(f"visualizations/notes_13/target_pred_0002.pt"),
    # load_image(f"visualizations/notes_13/target_pred_0003.pt"),
    # load_image(f"visualizations/notes_13/target_pred_0004.pt"),

    # load_image(f"visualizations/notes_7/pred_0010_0000.pt"),
    # load_image(f"visualizations/notes_7/pred_0020_0000.pt"),
    # load_image(f"visualizations/notes_7/pred_0050_0000.pt"),
    # load_image(f"visualizations/notes_7/pred_0070_0000.pt"),
    # load_image(f"visualizations/notes_7/pred_0100_0000.pt"),

    # load_image(f"visualizations/notes_7/target_pred_0001.pt"),
    # load_image(f"visualizations/notes_7/target_pred_0002.pt"),
    # load_image(f"visualizations/notes_7/target_pred_0003.pt"),
    # load_image(f"visualizations/notes_7/target_pred_0004.pt"),

    load_normal_image(f"final_0018.png"),
    load_normal_image(f"final_0017.png"),
    load_normal_image(f"final_0016.png"),
    load_normal_image(f"final_0015.png"),
    load_normal_image(f"final_0014.png"),
]
# # good image
# image_1 = load_image(f"visualizations/notes_10/pred_0500_0000.pt")

# fig, axes = plt.subplots(4, len(images), figsize=(5, 10))

# def image_column(col, image):
#     axes[0,col].imshow(image[0][:,:,0])
#     axes[1,col].imshow(image[0][:,:,1])
#     axes[2,col].imshow(image[0][:,:,2])
#     axes[3,col].imshow(image[0][:,:,3])

fig, axes = plt.subplots(1, len(images), figsize=(10, 5))

def image_column(col, image):
    # axes[col,0].imshow(image[0][:,:,0])
    # axes[col,1].imshow(image[0][:,:,1])
    # axes[col,2].imshow(image[0][:,:,2])
    # axes[col,3].imshow(image[0][:,:,3])

    # axes[0].imshow(image[0][:,:,0])
    # axes[1].imshow(image[0][:,:,1])
    # axes[2].imshow(image[0][:,:,2])
    # axes[3].imshow(image[0][:,:,3])

    # axes[col].imshow(image[0][:,:,0])
    # axes[1].imshow(image[0][:,:,1])
    # axes[2].imshow(image[0][:,:,2])
    # axes[3].imshow(image[0][:,:,3])

    axes[col].imshow(image)

for i, image in enumerate(images):
    image_column(i, image)

plt.show()
