
import torch

device = "cpu"

pred = torch.load(f"visualizations/notes_5/pred_0500_0001.pt", map_location=torch.device(device))
latents_pred = pred.view(1, 64, 64, 4).permute(0, 3, 1, 2)

print(torch.max(latents_pred))
print(torch.min(latents_pred))

pred = torch.load(f"visualizations/notes_8/target_pred_0500_0002.pt", map_location=torch.device(device))
latents_pred = pred.view(1, 64, 64, 4).permute(0, 3, 1, 2)

print(torch.max(latents_pred))
print(torch.min(latents_pred))

# NOTE:
    # the ranges are indeed about the same...