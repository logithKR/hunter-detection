import torch

# Load the handgun detection model
model_handgun = torch.load(r"runs/train/exp/weights/best.pt", map_location='cpu')['model'].float()

# Load the short gun detection model
model_shortgun = torch.load(r"runs/train/exp4/weights/best.pt", map_location='cpu')['model'].float()
# Create a new combined model based on the handgun model
combined_model = model_handgun

# Update the combined model to have 2 classes (handgun + short gun)
combined_model.nc = 2  # 2 classes: handgun and short gun

# Update the detection layer to output 2 classes
combined_model.model[-1] = torch.nn.Conv2d(in_channels=256, out_channels=2 * (combined_model.anchor_grid.shape[0]), kernel_size=1)
# Merge weights for common layers between the models
with torch.no_grad():
    for layer1, layer2 in zip(model_handgun.model, model_shortgun.model):
        if isinstance(layer1, torch.nn.Conv2d):  # Assuming Conv2d layers are shared
            layer1.weight.data = (layer1.weight.data + layer2.weight.data) / 2
            layer1.bias.data = (layer1.bias.data + layer2.bias.data) / 2
# Save the combined model
torch.save(combined_model.state_dict(), 'combined_model.pt')
# Assuming the combined model is now ready for training with both handguns and short guns
from yolov5 import train

train.run(data='dataset.yaml',  # Point to your dataset with both handguns and short guns
          cfg='yolov5s.yaml',  # You can choose your model architecture
          weights='combined_model.pt',  # Start training from the combined model
          epochs=50)  # Train for a specific number of epochs
