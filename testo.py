import torch
from torchvision import transforms
from PIL import Image
import numpy as np

class UpscalerModel(torch.nn.Module):
    def __init__(self):
        super(UpscalerModel, self).__init__()
        # Define your layers here, based on the architecture you expect
        # This is just a placeholder for demonstration purposes
        pass
    
    def forward(self, x):
        # Define the forward pass of your model
        return x

def upscale_image(input_image_path, model_path, output_image_path):
    # Load the image
    image = Image.open(input_image_path).convert("RGB")

    # Define transformation (resize the image for processing)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image.height * 2, image.width * 2)),  # Upscale 2x
        transforms.Lambda(lambda x: x.unsqueeze(0))  # Add batch dimension
    ])

    # Apply transformation
    image_tensor = transform(image).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Initialize the model
    model = UpscalerModel().to(image_tensor.device)

    # Load the state dictionary from the checkpoint
    state_dict = torch.load(model_path)

    # Optionally, remove a prefix from keys if needed (as before)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("model.", "")  # Adjust the prefix removal based on the error message
        new_state_dict[new_key] = value

    # Load the state dictionary with strict=False to allow mismatched keys
    try:
        model.load_state_dict(new_state_dict, strict=False)  # Ignore missing/unexpected keys
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")

    # Set the model to evaluation mode
    model.eval()

    # Perform inference to upscale the image
    with torch.no_grad():
        upscaled_tensor = model(image_tensor)

    # Convert the tensor back to an image
    upscaled_image = upscaled_tensor.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255
    upscaled_image = Image.fromarray(upscaled_image.astype(np.uint8))

    # Save the upscaled image
    upscaled_image.save(output_image_path)
    print(f"Upscaled image saved to {output_image_path}")

# Example usage
upscale_image("/home/trithemius/Downloads/replicate-prediction-t6jn0dje6drm80cmt33tz1rk8m.webp", "/gorgon/ia/ComfyUI/models/upscale_models/4x_NMKD-Siax_200k.pth", "./upscaled_image.jpg")
