import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils.logging import logger

from src.models.early_stopping import EarlyStopping


def train_autoencoder(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    patience: int,
    min_delta: float,
    device: str,
    verbose_training: bool = True,
    verbose_validation: bool = False,
):
    model.to(device)
    logger.info(f"Device: {device} | Model on: {next(model.parameters()).device}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=5,
    )

    early_stopping = EarlyStopping(
        patience=patience, min_delta=min_delta, verbose=False
    )

    train_losses = []
    val_losses = []

    # Training
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for x, y in train_loader:
            mask = y == 0
            if mask.sum().item() == 0:
                continue

            x = x[mask].to(device)
            x_reconstructed = model(x)
            loss = loss_fn(x_reconstructed, x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for x, y in val_loader:
                mask = y == 0
                x = x[mask].to(device)
                x_reconstructed = model(x)
                loss = loss_fn(x_reconstructed, x)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        val_losses.append(avg_val_loss)

        if verbose_training:
            logger.info(
                f"[Epoch {epoch + 1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
            )

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Check early stopping
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            if verbose_validation:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Load best model
    early_stopping.load_best_model(model)
    if verbose_validation:
        logger.info(
            f"Loaded best model with validation loss: {early_stopping.best_loss:.4f}"
        )

    return model, train_losses, val_losses
