"""
Unit tests for model architecture.
"""

import pytest
import torch
import torch.nn as nn
from src.models.cnn_model import CustomCXRClassifier


@pytest.mark.unit
class TestCustomCXRClassifier:
    """Test suite for CustomCXRClassifier model."""

    def test_model_initialization(self):
        """Test that model initializes correctly."""
        model = CustomCXRClassifier(num_classes=3)
        assert isinstance(model, nn.Module)

    def test_forward_pass_output_shape(self, device):
        """Test forward pass produces correct output shape."""
        model = CustomCXRClassifier(num_classes=3).to(device)
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 128, 128).to(device)

        output = model(input_tensor)

        assert output.shape == (batch_size, 3), \
            f"Expected shape ({batch_size}, 3), got {output.shape}"

    def test_forward_pass_output_is_probability(self, device):
        """Test that output is a valid probability distribution."""
        model = CustomCXRClassifier(num_classes=3).to(device)
        input_tensor = torch.randn(1, 3, 128, 128).to(device)

        output = model(input_tensor)

        # Check probabilities sum to 1
        assert torch.allclose(
            output.sum(dim=1),
            torch.tensor([1.0]).to(device),
            atol=1e-5
        ), "Output probabilities should sum to 1"

        # Check probabilities are in [0, 1]
        assert (output >= 0).all() and (output <= 1).all(), \
            "Output probabilities should be between 0 and 1"

    @pytest.mark.parametrize("batch_size", [1, 2, 8, 16, 32])
    def test_model_handles_different_batch_sizes(self, batch_size, device):
        """Test model handles various batch sizes."""
        model = CustomCXRClassifier(num_classes=3).to(device)
        input_tensor = torch.randn(batch_size, 3, 128, 128).to(device)

        output = model(input_tensor)

        assert output.shape[0] == batch_size, \
            f"Failed for batch size {batch_size}"

    def test_model_training_mode(self, device):
        """Test model behavior in training mode."""
        model = CustomCXRClassifier(num_classes=3).to(device)
        model.train()

        assert model.training, "Model should be in training mode"

        # With dropout active, outputs should potentially differ
        input_tensor = torch.randn(1, 3, 128, 128).to(device)
        _ = model(input_tensor)  # Just verify no errors

    def test_model_eval_mode(self, device):
        """Test model behavior in evaluation mode."""
        model = CustomCXRClassifier(num_classes=3).to(device)
        model.eval()

        assert not model.training, "Model should be in evaluation mode"

        # Without dropout, outputs should be deterministic
        input_tensor = torch.randn(1, 3, 128, 128).to(device)
        output1 = model(input_tensor)
        output2 = model(input_tensor)

        assert torch.allclose(output1, output2), \
            "Outputs should be identical in eval mode"

    def test_model_parameters_have_gradients(self):
        """Test that model parameters require gradients."""
        model = CustomCXRClassifier(num_classes=3)

        for name, param in model.named_parameters():
            assert param.requires_grad, \
                f"Parameter {name} should require gradients"

    def test_backward_pass_computes_gradients(self, device):
        """Test that backward pass computes gradients."""
        model = CustomCXRClassifier(num_classes=3).to(device)
        model.train()

        input_tensor = torch.randn(2, 3, 128, 128).to(device)
        target = torch.tensor([0, 1]).to(device)
        criterion = nn.CrossEntropyLoss()

        output = model(input_tensor)
        loss = criterion(output, target)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, \
                f"Parameter {name} should have gradients after backward()"
            assert not torch.isnan(param.grad).any(), \
                f"Parameter {name} has NaN gradients"

    @pytest.mark.parametrize("num_classes", [2, 3, 5, 10])
    def test_model_with_different_num_classes(self, num_classes, device):
        """Test model works with various number of output classes."""
        model = CustomCXRClassifier(num_classes=num_classes).to(device)
        input_tensor = torch.randn(1, 3, 128, 128).to(device)

        output = model(input_tensor)

        assert output.shape[1] == num_classes, \
            f"Output should have {num_classes} classes"

    def test_model_serialization(self, sample_model, tmp_path):
        """Test model can be saved and loaded."""
        save_path = tmp_path / "model.pth"

        # Save model
        torch.save(sample_model.state_dict(), save_path)

        # Load model
        loaded_model = CustomCXRClassifier(num_classes=3)
        loaded_model.load_state_dict(torch.load(save_path))

        # Compare outputs
        input_tensor = torch.randn(1, 3, 128, 128)
        sample_model.eval()
        loaded_model.eval()

        output1 = sample_model(input_tensor)
        output2 = loaded_model(input_tensor)

        assert torch.allclose(output1, output2), \
            "Loaded model should produce identical outputs"

    def test_model_parameter_count(self):
        """Test model has expected number of parameters."""
        model = CustomCXRClassifier(num_classes=3)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        assert total_params > 0, "Model should have parameters"
        assert trainable_params == total_params, \
            "All parameters should be trainable by default"

    def test_model_with_zero_input(self, device):
        """Test model handles zero input."""
        model = CustomCXRClassifier(num_classes=3).to(device)
        input_tensor = torch.zeros(1, 3, 128, 128).to(device)

        output = model(input_tensor)

        assert not torch.isnan(output).any(), \
            "Model should not produce NaN with zero input"
        assert not torch.isinf(output).any(), \
            "Model should not produce Inf with zero input"
