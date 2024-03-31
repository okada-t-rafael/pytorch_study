"""
Contains functions for training and testing a PyTorch model.
"""
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple


def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str
        ) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then runs through all the
    required training steps (forward pass, loss calculation, optimizer step).

    Args:
        model: A Pytorch model to be trained.
        dataloader: A Dataloader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss and training accuracy metrics. In the form
        (train_loss, train_accuracy). For example: (0.1112, 0.8743)
    """
    # Put the model into target device
    model.to(device)

    # Pu the model in train mode
    model.train()

    # Setup train loss and train accuracy variables
    train_loss: float = 0.0
    train_acc: float = 0.0

    # Loop through dataloader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to the target device
        X = X.to(device)
        y = y.to(device)

        # 1. Forward pass
        y_pred_logits = model(X)  # output model logits

        # 2. Calculate the loss
        loss = loss_fn(y_pred_logits, y)
        train_loss += loss.item()  # convert this tensor to a standard python number

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate accuracy metric
        y_pred_probs = torch.softmax(y_pred_logits, dim=1)
        y_pred_labels = torch.argmax(y_pred_probs, dim=1)
        train_acc += (y_pred_labels == y).sum().item() / len(y_pred_labels)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc


def test_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: str) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to 'eval' mode and then performs a forward
    pass on a testing dataset.

    Args:
        model: A Pytorch model to be tested.
        dataloader: A Dataloader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of testing loss and testing accuracy metrics. In the form
        (test_loss, test_accuracy). For example: (0.1112, 0.8743)
    """
    # Put model into target device
    model.to(device)

    # Put model in eval mode
    model.eval()

    # Step test loss and test accuracy variables
    test_loss: float = 0.0
    test_acc: float = 0.0

    # Turn on inference mode
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X = X.to(device)
            y = y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate the loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate the accuracy
            test_pred_probs = test_pred_logits.softmax(dim=1)
            test_pred_labels = test_pred_probs.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

        # Adjust metrics to get average loss and accuracy per batch
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)

        return test_loss, test_acc


def train(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        epochs: int,
        device: str,
        ) -> Dict[str, List[float]]:
    """
    Trains and tests a PyTorch model.

    Passes a target model through 'train_step' and 'test_step' functions for a
    number of epochs, training and testing the model in the same epoch loop.
    Calculates, prints and stores evaluation metrics throughout.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for each
        epoch.

        For example if training for epochs = 2: {
            train_loss: [2.0616, 1.0537],
            train_acc: [0.3945, 0.3945],
            test_loss: [1.2641, 1.5706],
            test_acc: [0.3400, 0.2973]
        }
    """
    # Create an empty results dictionary
    results: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []}

    # Loop through training and testing steps for a number o epochs
    for epoch in tqdm(range(epochs)):
        # Training step
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device)

        # Test step
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device)

        # Print out what's happening
        print(
            f" - Epoch: {epoch} | "
            f"Train_loss: {train_loss:.4f}, Train_acc: {train_acc:.3f}% | "
            f"Test_loss: {test_loss:.4f}, Test_acc: {test_acc:.3f}%")

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results
