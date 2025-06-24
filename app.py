# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import os

app = Flask(__name__)
# Enable CORS for all origins, which is necessary for frontend to communicate with backend
CORS(app)

# --- Configuration for your PyTorch model ---
# UPDATED: Path to your fine-tuned ResNet18 model file
MODEL_PATH = 'resnet18_covid_ct_scans_classification.pth' # This path now matches your file name

# Define the device to run the model on (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformations applied to the input images during prediction.
# These should match the transformations used during your model's training.
# Common transforms for ResNet18 trained on ImageNet are included.
prediction_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet18 typically expects 224x224 input
    transforms.ToTensor(),          # Convert PIL Image to PyTorch Tensor
    # Normalize with mean and standard deviation from ImageNet (common for pre-trained models)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Global variable to hold the loaded model
model = None

# --- Model Loading Function ---
def load_model():
    global model
    try:
        # NOTE: You MUST define your model architecture (e.g., a ResNet18 model)
        # exactly as it was when you saved it. This is a placeholder.
        # If you saved the entire model (including architecture), you might
        # just need torch.load(MODEL_PATH, map_location=device).
        # If you only saved the state_dict, you need to first instantiate
        # the model architecture and then load the state_dict.

        # Example: Instantiating a pre-trained ResNet18 and modifying its final layer
        # if your fine-tuning changed the head.
        import torchvision.models as models
        model = models.resnet18(pretrained=False) # Start with an untrained ResNet18
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2) # Assuming 2 classes: COVID Positive/Negative

        # Load the state dictionary
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval() # Set the model to evaluation mode (important for inference)
        print(f"Model loaded successfully from {MODEL_PATH} on {device}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. Please check the path.")
        model = None
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure your model architecture matches the saved model and it's compatible with torch.load().")
        model = None

# Load the model when the Flask application starts
load_model()

# --- Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    # Check if a file was uploaded in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Read the image file from the request
            img_bytes = file.read()
            # Open the image using PIL (Pillow)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB') # Ensure image is RGB

            # Apply the defined transformations
            input_tensor = prediction_transforms(img)
            # Add a batch dimension (B, C, H, W)
            input_batch = input_tensor.unsqueeze(0)

            # Move the input tensor to the same device as the model
            input_batch = input_batch.to(device)

            # Perform inference with no gradient calculation (saves memory and speeds up)
            with torch.no_grad():
                output = model(input_batch)

            # Get probabilities (optional, but good for understanding confidence)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            # Get the predicted class index
            _, predicted_class_idx = torch.max(output, 1)

            # Map the predicted index to class labels
            # IMPORTANT: Adjust these labels based on your model's output
            # For example, if 0 is Negative and 1 is Positive
            class_labels = ["COVID Negative", "COVID Positive"]
            prediction_label = class_labels[predicted_class_idx.item()]

            # Return the prediction as a JSON response
            return jsonify({
                'prediction': prediction_label,
                'probabilities': probabilities.tolist()[0] # Convert tensor to list
            })

        except Exception as e:
            # Catch any errors during image processing or model inference
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# --- Main entry point for Flask app ---
if __name__ == '__main__':
    # Ensure the model file exists before running the app
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model file '{MODEL_PATH}' not found. Please place your PyTorch model there.")
        print("The server will start, but predictions will fail until the model is available.")

    # Run the Flask app on host 0.0.0.0 to make it accessible from other devices on the network,
    # and port 5000 (common for Flask dev servers).
    app.run(host='0.0.0.0', port=5000, debug=True)
