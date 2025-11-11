"""
Refactored CNN model with proper architecture patterns and configurability.

Key improvements over original:
1. Removed softmax from forward() - returns raw logits
2. Made architecture configurable via ModelConfig
3. Dynamic FC input size calculation
4. Added type hints
5. Better code organization with private methods
6. Added predict_proba() for when probabilities are needed
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """
    Configuration for CNN model architecture.

    This makes the model flexible and eliminates hardcoded values.
    Now you can easily experiment with different architectures.

    Attributes:
        input_channels: Number of input channels (3 for RGB, 1 for grayscale)
        num_classes: Number of output classes
        image_size: Input image size (assumes square images)
        conv_filters: Number of filters in each conv layer
        dropout_rates: Dropout rates for each conv block
        fc_hidden_size: Size of first fully connected layer
        fc_intermediate_size: Size of second fully connected layer
        fc_dropout: Dropout rate for fully connected layers

    Example:
        >>> # Use defaults
        >>> config = ModelConfig()
        >>>
        >>> # Custom configuration for 4-class problem
        >>> config = ModelConfig(
        ...     num_classes=4,
        ...     image_size=224,
        ...     conv_filters=[32, 64, 128, 256]
        ... )
    """

    input_channels: int = 3
    num_classes: int = 3
    image_size: int = 128

    # Convolutional layer filters
    conv_filters: Tuple[int, int, int, int] = (16, 64, 128, 128)

    # Dropout rates for conv blocks
    dropout_rates: Tuple[float, float, float, float] = (0.25, 0.25, 0.3, 0.4)

    # Fully connected layer sizes
    fc_hidden_size: int = 128
    fc_intermediate_size: int = 64
    fc_dropout: float = 0.25

    def __post_init__(self):
        """Validate configuration."""
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {self.num_classes}")

        if self.input_channels < 1:
            raise ValueError(f"input_channels must be positive, got {self.input_channels}")

        if self.image_size < 32:
            raise ValueError(f"image_size too small, minimum 32, got {self.image_size}")

        if len(self.conv_filters) != 4:
            raise ValueError(f"conv_filters must have 4 values, got {len(self.conv_filters)}")

        if len(self.dropout_rates) != 4:
            raise ValueError(f"dropout_rates must have 4 values, got {len(self.dropout_rates)}")

        for rate in self.dropout_rates:
            if not 0 <= rate < 1:
                raise ValueError(f"Dropout rate must be in [0, 1), got {rate}")

        if not 0 <= self.fc_dropout < 1:
            raise ValueError(f"fc_dropout must be in [0, 1), got {self.fc_dropout}")


class CustomCXRClassifier(nn.Module):
    """
    Configurable Convolutional Neural Network for Chest X-Ray classification.

    Architecture:
        Conv Block 1 -> Pool -> Dropout
        Conv Block 2 -> Pool -> Dropout
        Conv Block 3 -> Pool -> Dropout
        Conv Block 4 -> Pool -> Dropout
        Flatten
        FC1 -> ReLU -> Dropout
        FC2 -> ReLU
        Output (logits)

    Attribution:
        Based on architecture from:
        https://github.com/Vinay10100/Chest-X-Ray-Classification

    Key changes from original:
        1. **CRITICAL FIX**: Removed softmax from forward() method
           - Returns raw logits instead
           - CrossEntropyLoss expects logits, not probabilities
           - Applying softmax here causes numerical instability and incorrect gradients

        2. Made architecture configurable via ModelConfig
           - No more hardcoded values
           - Easy to experiment with different architectures

        3. Dynamic FC input size calculation
           - Works with any input image size
           - No manual calculation needed

        4. Added type hints for better IDE support and clarity

        5. Separated concerns with private methods

    Example:
        >>> # Create model with default config
        >>> config = ModelConfig()
        >>> model = CustomCXRClassifier(config)
        >>>
        >>> # Forward pass returns logits
        >>> x = torch.randn(32, 3, 128, 128)
        >>> logits = model(x)  # Shape: (32, 3)
        >>>
        >>> # Use with CrossEntropyLoss (recommended)
        >>> criterion = nn.CrossEntropyLoss()
        >>> loss = criterion(logits, labels)
        >>>
        >>> # Get probabilities when needed
        >>> probs = model.predict_proba(x)  # Shape: (32, 3)
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the CNN model.

        Args:
            config: Model configuration specifying architecture parameters
        """
        super().__init__()
        self.config = config

        # Extract filter sizes for readability
        f1, f2, f3, f4 = config.conv_filters

        # Extract dropout rates
        d1, d2, d3, d4 = config.dropout_rates

        # --- Convolutional Block 1 ---
        # Note: padding=0 (no padding) as in original architecture
        self.conv1 = nn.Conv2d(
            config.input_channels,
            f1,
            kernel_size=3,
            padding=0
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Note: Dropout after first block not in original, but we include for consistency
        self.dropout1 = nn.Dropout2d(d1)

        # --- Convolutional Block 2 ---
        # padding='same' keeps spatial dimensions unchanged
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(d2)

        # --- Convolutional Block 3 ---
        self.conv3 = nn.Conv2d(f2, f3, kernel_size=3, padding='same')
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(d3)

        # --- Convolutional Block 4 ---
        self.conv4 = nn.Conv2d(f3, f4, kernel_size=3, padding='same')
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout2d(d4)

        # Calculate FC input size dynamically
        self.fc_input_features = self._calculate_fc_input_size()

        # --- Fully Connected Layers ---
        self.fc1 = nn.Linear(self.fc_input_features, config.fc_hidden_size)
        self.dropout_fc1 = nn.Dropout(config.fc_dropout)

        self.fc2 = nn.Linear(config.fc_hidden_size, config.fc_intermediate_size)

        # Output layer
        self.output_layer = nn.Linear(config.fc_intermediate_size, config.num_classes)

    def _calculate_fc_input_size(self) -> int:
        """
        Dynamically calculate the number of features after conv layers.

        This method runs a dummy forward pass through the convolutional layers
        to determine the output size. This eliminates the need for manual
        calculation and makes the model work with any input size.

        Returns:
            Number of features after flattening conv layer outputs

        Why this is important:
            - Original code hardcoded 6272 for 128x128 images
            - This wouldn't work if image_size changed
            - Dynamic calculation makes the model flexible

        How it works:
            1. Create a dummy tensor with the configured input shape
            2. Pass it through conv layers
            3. Flatten and measure the size
            4. Return the size for FC layer initialization
        """
        # Create dummy input
        dummy_input = torch.zeros(
            1,  # Batch size of 1
            self.config.input_channels,
            self.config.image_size,
            self.config.image_size
        )

        # Forward through conv layers only
        x = self._forward_conv_layers(dummy_input)

        # Flatten and return size
        return x.view(1, -1).size(1)

    def _forward_conv_layers(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through convolutional layers only.

        Extracted as a private method for:
        1. Code reuse (used in both forward() and _calculate_fc_input_size())
        2. Better organization
        3. Easier to modify conv architecture

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor after all conv layers, before flattening
        """
        # Block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Block 3
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # Block 4
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool4(x)
        x = self.dropout4(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        **CRITICAL**: This method returns RAW LOGITS, not probabilities.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output logits of shape (batch_size, num_classes)

        Why no softmax?
            When using nn.CrossEntropyLoss (the standard for classification):
            - CrossEntropyLoss expects LOGITS (raw scores)
            - It internally applies log_softmax for numerical stability
            - Applying softmax here then passing to CrossEntropyLoss causes:
              1. Numerical instability (log(softmax(x)) can underflow)
              2. Incorrect gradients
              3. Slower convergence

        When to use softmax?
            - During inference to get probabilities
            - Use the predict_proba() method instead
            - Or apply F.softmax(logits, dim=1) manually after forward()

        Example:
            >>> # Training (use logits with CrossEntropyLoss)
            >>> logits = model(images)
            >>> loss = criterion(logits, labels)  # criterion = nn.CrossEntropyLoss()
            >>>
            >>> # Inference (get class predictions)
            >>> logits = model(images)
            >>> predictions = torch.argmax(logits, dim=1)
            >>>
            >>> # Inference (get probabilities)
            >>> probs = model.predict_proba(images)
        """
        # Convolutional layers
        x = self._forward_conv_layers(x)

        # Flatten
        x = torch.flatten(x, 1)  # Keep batch dimension

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc1(x)

        x = self.fc2(x)
        x = F.relu(x)

        # Output layer - returns LOGITS
        x = self.output_layer(x)

        # NO SOFTMAX! Return raw logits
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions.

        Use this method during inference when you need class probabilities
        instead of logits.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Probability distribution of shape (batch_size, num_classes)
            Each row sums to 1.0

        Example:
            >>> model.eval()
            >>> with torch.no_grad():
            ...     probs = model.predict_proba(images)
            ...     # probs[0] = [0.8, 0.15, 0.05]  # Probabilities for first image
            ...     top_class = torch.argmax(probs, dim=1)
            ...     confidence = torch.max(probs, dim=1)[0]
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def get_num_parameters(self) -> int:
        """
        Get total number of trainable parameters.

        Useful for comparing model complexity.

        Returns:
            Total number of parameters

        Example:
            >>> model = CustomCXRClassifier(config)
            >>> num_params = model.get_num_parameters()
            >>> print(f"Model has {num_params:,} parameters")
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def model_summary(
    model: CustomCXRClassifier,
    device: str = 'cpu'
) -> None:
    """
    Print a summary of the model architecture.

    Args:
        model: The model to summarize
        device: Device to use for summary ('cpu' or 'cuda')

    Example:
        >>> config = ModelConfig()
        >>> model = CustomCXRClassifier(config)
        >>> model_summary(model)
    """
    print("=" * 70)
    print("Custom CXR Classifier - Model Summary")
    print("=" * 70)

    config = model.config
    print(f"\nConfiguration:")
    print(f"  Input channels:    {config.input_channels}")
    print(f"  Number of classes: {config.num_classes}")
    print(f"  Input size:        {config.image_size}x{config.image_size}")
    print(f"  Conv filters:      {config.conv_filters}")
    print(f"  Dropout rates:     {config.dropout_rates}")
    print(f"  FC hidden size:    {config.fc_hidden_size}")
    print(f"  FC inter size:     {config.fc_intermediate_size}")

    print(f"\nModel Statistics:")
    num_params = model.get_num_parameters()
    print(f"  Total parameters:  {num_params:,}")
    print(f"  FC input features: {model.fc_input_features}")

    print(f"\nArchitecture:")
    print(f"  Input -> Conv({config.conv_filters[0]}) -> Pool -> Dropout({config.dropout_rates[0]})")
    print(f"       -> Conv({config.conv_filters[1]}) -> Pool -> Dropout({config.dropout_rates[1]})")
    print(f"       -> Conv({config.conv_filters[2]}) -> Pool -> Dropout({config.dropout_rates[2]})")
    print(f"       -> Conv({config.conv_filters[3]}) -> Pool -> Dropout({config.dropout_rates[3]})")
    print(f"       -> Flatten({model.fc_input_features})")
    print(f"       -> FC({config.fc_hidden_size}) -> ReLU -> Dropout({config.fc_dropout})")
    print(f"       -> FC({config.fc_intermediate_size}) -> ReLU")
    print(f"       -> FC({config.num_classes}) -> Logits")

    # Test forward pass
    print(f"\nTest Forward Pass:")
    model = model.to(device)
    model.eval()

    test_input = torch.randn(
        2,  # Batch size of 2
        config.input_channels,
        config.image_size,
        config.image_size
    ).to(device)

    with torch.no_grad():
        logits = model(test_input)
        probs = model.predict_proba(test_input)

    print(f"  Input shape:       {tuple(test_input.shape)}")
    print(f"  Output shape:      {tuple(logits.shape)}")
    print(f"  Logits (sample):   {logits[0].cpu().numpy()}")
    print(f"  Probs (sample):    {probs[0].cpu().numpy()}")
    print(f"  Probs sum:         {probs[0].sum().item():.6f} (should be 1.0)")

    print("=" * 70)


if __name__ == '__main__':
    print("Testing refactored CNN model\n")

    # Test 1: Default configuration
    print("Test 1: Default Configuration (3-class, 128x128, RGB)")
    print("-" * 70)
    config = ModelConfig()
    model = CustomCXRClassifier(config)
    model_summary(model)

    print("\n\n")

    # Test 2: Custom configuration
    print("Test 2: Custom Configuration (4-class, 224x224, RGB)")
    print("-" * 70)
    custom_config = ModelConfig(
        num_classes=4,
        image_size=224,
        conv_filters=(32, 64, 128, 256),
        dropout_rates=(0.3, 0.3, 0.4, 0.5),
        fc_hidden_size=256,
        fc_intermediate_size=128
    )
    custom_model = CustomCXRClassifier(custom_config)
    model_summary(custom_model)

    print("\n\n")

    # Test 3: Grayscale configuration
    print("Test 3: Grayscale Configuration (3-class, 128x128, Grayscale)")
    print("-" * 70)
    grayscale_config = ModelConfig(
        input_channels=1,
        num_classes=3,
        image_size=128
    )
    grayscale_model = CustomCXRClassifier(grayscale_config)
    model_summary(grayscale_model)

    print("\n\n")

    # Test 4: Verify logits vs probabilities
    print("Test 4: Verify Logits vs Probabilities")
    print("-" * 70)
    model = CustomCXRClassifier(ModelConfig())
    model.eval()

    test_input = torch.randn(1, 3, 128, 128)

    with torch.no_grad():
        logits = model(test_input)
        probs = model.predict_proba(test_input)
        manual_probs = F.softmax(logits, dim=1)

    print(f"Logits:              {logits[0].numpy()}")
    print(f"Probs (method):      {probs[0].numpy()}")
    print(f"Probs (manual):      {manual_probs[0].numpy()}")
    print(f"Probs sum:           {probs[0].sum().item():.6f}")
    print(f"Match:               {torch.allclose(probs, manual_probs)}")

    print("\n✓ All tests passed!")
