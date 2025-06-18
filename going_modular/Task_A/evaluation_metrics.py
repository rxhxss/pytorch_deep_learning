import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from tqdm.auto import tqdm
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import os

def evaluate_model(model: torch.nn.Module,
                   test_data_path: str,
                   transform: transforms.Compose,
                   batch_size: int,
                   device: torch.device,
                   num_classes: int = None) -> dict:
    """
    Evaluates a PyTorch model using Accuracy, Precision, Recall, and F1-score.
    
    Args:
        model: PyTorch model to evaluate
        test_data_path: Path to the test data directory
        transform: PyTorch transform to apply to the test data
        batch_size: Batch size for the data loader
        device: Device to run evaluation on ("cuda" or "cpu")
        num_classes: Number of classes (needed for multiclass metrics)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Set model to evaluation mode
    test_data=datasets.ImageFolder(test_data_path, transform=transform,target_transform=None)
    test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=os.cpu_count(),
      pin_memory=True,
  )
    model.eval()
    
    # Initialize lists to store true and predicted labels
    y_true = []
    y_pred = []
    
    # Disable gradient calculation
    with torch.inference_mode():
        # Loop through batches
        for X, y in tqdm(test_dataloader, desc="Evaluating"):
            # Send data to device
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            y_logits = model(X)
            
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            
            # Store labels
            y_true.extend(y.cpu().numpy())
            y_pred.extend(y_pred_class.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='macro' if num_classes > 2 else 'binary', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='macro' if num_classes > 2 else 'binary', zero_division=0),
        "f1": f1_score(y_true, y_pred, average='macro' if num_classes > 2 else 'binary', zero_division=0)
    }
    
    return metrics

def print_metrics(metrics: dict):
    """
    Prints the evaluation metrics.
    
    Args:
        metrics: Dictionary containing evaluation metrics
    """
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    
    
