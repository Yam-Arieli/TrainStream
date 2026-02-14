__all__ = ["simple_torch_train"]

from typing import Any

def simple_torch_train(data, model, optimizer, criterion, epochs=1, device="cpu", **kwargs) -> Any:
    """
    A boilerplate PyTorch training loop.
    Expects 'data' to be a standard PyTorch DataLoader or iterable of (x, y).
    """
    import torch
    
    model.train()
    model.to(device)
    
    epoch_losses = []
    
    # If data is a Tensor, wrap it; if it's a DataLoader, iterate it.
    # This is a naive implementation assuming data is iterable batch-style.
    iterator = data if hasattr(data, '__iter__') else [data]

    for _ in range(epochs):
        for batch in iterator:
            # Assuming batch is [inputs, labels]
            inputs, labels = batch[0].to(device), batch[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())

    # Return metadata for the selector
    return {'avg_loss': sum(epoch_losses) / len(epoch_losses)}