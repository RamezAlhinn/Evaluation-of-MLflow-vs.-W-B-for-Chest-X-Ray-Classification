import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# The number of input channels is set to 3 to simulate standard image loading,
# but can be changed to 1 if the X-ray images are strictly loaded as grayscale.
INPUT_CHANNELS = 3 
# The number of output classes is 3 (COVID-19, Viral Pneumonia, Normal)
NUM_CLASSES = 3

class CustomCXRClassifier(nn.Module):
    """
    A custom Convolutional Neural Network (CNN) architecture designed for 
    3-way Chest X-Ray classification (COVID-19, Viral Pneumonia, Normal).
    
    The architecture parameters (Conv Depth, Filter Size, Dropout) are key 
    hyperparameters to be tracked and compared in the MLflow vs. W&B evaluation.
    """
    def __init__(self, in_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES):
        super(CustomCXRClassifier, self).__init__()
        
        # --- Convolutional Block 1 (Depth/Block 1) ---
        # Conv1: 16 filters, 3x3 kernel, no padding (P=0)
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Convolutional Block 2 (Depth/Block 2) ---
        # Conv2: 64 filters, 3x3 kernel, padding='same' (P=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25) # Dropout after Conv 2

        # --- Convolutional Block 3 (Depth/Block 3) ---
        # Conv3: 128 filters, 3x3 kernel, padding='same' (P=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.3) # Dropout after Conv 3

        # --- Convolutional Block 4 (Depth/Block 4) ---
        # Conv4: 128 filters, 3x3 kernel, padding='same' (P=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.4) # Dropout after Conv 4
        
        # Calculate the size for the Linear layer input dynamically.
        # Based on an input size of 128x128, the output feature map size is 7x7.
        # (128 channels * 7 * 7) = 6272
        self.fc_input_features = 6272

        # --- Fully Connected Layers ---
        self.fc1 = nn.Linear(self.fc_input_features, 128) # Dense 1: 128 neurons, ReLU
        self.dropout_fc1 = nn.Dropout(0.25) # Dropout after Dense 1

        self.fc2 = nn.Linear(128, 64) # Dense 2: 64 neurons, ReLU
        
        # Output layer: 3 classes, Softmax
        self.output_layer = nn.Linear(64, num_classes)

    def forward(self, x):
        # Block 1
        x = self.pool1(F.relu(self.conv1(x)))
        
        # Block 2
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout1(x)

        # Block 3
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.dropout2(x)

        # Block 4
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.dropout3(x)

        # Flatten
        x = torch.flatten(x, 1) # Retain batch dimension (dim 0)
        
        # Dense Layers
        x = F.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        
        x = F.relu(self.fc2(x))

        # Output Layer (Softmax applied in loss function, but here we include it 
        # for a complete architecture definition per the user's request)
        x = self.output_layer(x)
        # We use LogSoftmax if using CrossEntropyLoss in PyTorch, 
        # but to strictly match the request ("softmax activation") we use Softmax
        # If running a typical PyTorch training loop, use nn.CrossEntropyLoss 
        # which implicitly includes the Softmax operation.
        x = F.softmax(x, dim=1) 
        
        return x

def model_summary(model, input_size=(INPUT_CHANNELS, 128, 128)):
    """
    Initializes the model and prints a summary to confirm the layer sizes 
    and parameter counts based on a hypothetical input image size.
    
    Args:
        model (nn.Module): The PyTorch model class.
        input_size (tuple): The expected input tensor shape (C, H, W).
    """
    print("--- Custom CNN Model Summary ---")
    print(f"Input Shape (C, H, W): {input_size}")
    # Note: Requires 'torchsummary' to be installed (pip install torchsummary)
    summary(model, input_size=input_size)
    print("--------------------------------")


if __name__ == '__main__':
    # Initialize the model
    model = CustomCXRClassifier()
    
    # Run the summary using a standard input size of 128x128
    # The image size must be consistent across all experiments.
    model_summary(model, input_size=(INPUT_CHANNELS, 128, 128))

    # Example of a test forward pass
    test_input = torch.randn(1, INPUT_CHANNELS, 128, 128)
    output = model(test_input)
    print(f"\nTest Output Shape: {output.shape} (Batch Size x Num Classes)")
    print("Test Output (Probabilities):", output.data)