import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils.logging import logger

from src.models.early_stopping import EarlyStopping


def train_autoencoder_frozen(
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

    if verbose_validation:
        logger.info("STRATEGY: FROZEN ENCODER")
        logger.info("Encoder remains frozen, training decoder + bottleneck only")

    if hasattr(model, "encoder"):
        for p in model.encoder.parameters():
            p.requires_grad = False

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5
    )
    early_stopping = EarlyStopping(
        patience=patience, min_delta=min_delta, verbose=False
    )

    train_losses = []
    val_losses = []

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

        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0
        train_losses.append(avg_train_loss)

        # validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for x, y in val_loader:
                mask = y == 0
                if mask.sum().item() == 0:
                    continue

                x = x[mask].to(device)
                x_reconstructed = model(x)
                loss = loss_fn(x_reconstructed, x)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        val_losses.append(avg_val_loss)

        if verbose_training:
            logger.info(
                f"[Epoch {epoch + 1}/{epochs}] Frozen | "
                f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}"
            )

        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            if verbose_validation:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    early_stopping.load_best_model(model)
    if verbose_validation:
        logger.info(
            f"Loaded best model with validation loss: {early_stopping.best_loss:.4f}"
        )

    return model, train_losses, val_losses


def train_autoencoder_progressive_unfreeze(
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
    unfreeze_epoch: int = 20,
):
    model.to(device)
    logger.info(f"Device: {device} | Model on: {next(model.parameters()).device}")

    if verbose_validation:
        logger.info("STRATEGY: PROGRESSIVE UNFREEZE")
        logger.info(
            f"Encoder frozen until epoch {unfreeze_epoch}, "
            "then unfrozen with reduced LR for encoder"
        )

    if hasattr(model, "encoder"):
        for p in model.encoder.parameters():
            p.requires_grad = False

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5
    )
    early_stopping = EarlyStopping(
        patience=patience, min_delta=min_delta, verbose=False
    )

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Unfreeze encoder after specified epoch
        if epoch == unfreeze_epoch:
            if verbose_validation:
                logger.info(f"Unfreezing encoder at epoch {epoch + 1}")

            if hasattr(model, "unfreeze_encoder"):
                model.unfreeze_encoder()
            elif hasattr(model, "encoder"):
                for p in model.encoder.parameters():
                    p.requires_grad = True

            opt = torch.optim.Adam(
                [
                    {"params": model.encoder.parameters(), "lr": lr * 0.1},
                    {"params": model.to_latent.parameters(), "lr": lr},
                    {"params": model.from_latent.parameters(), "lr": lr},
                    {"params": model.decoder.parameters(), "lr": lr},
                ]
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=5
            )

            if verbose_validation:
                logger.info(f"Encoder LR: {lr * 0.1:.2e} | Decoder LR: {lr:.2e}")

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

        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0
        train_losses.append(avg_train_loss)

        # validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for x, y in val_loader:
                mask = y == 0
                if mask.sum().item() == 0:
                    continue

                x = x[mask].to(device)
                x_reconstructed = model(x)
                loss = loss_fn(x_reconstructed, x)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        val_losses.append(avg_val_loss)

        if verbose_training:
            logger.info(
                f"[Epoch {epoch + 1}/{epochs}] | "
                f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}"
            )

        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            if verbose_validation:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    early_stopping.load_best_model(model)
    if verbose_validation:
        logger.info(
            f"Loaded best model with validation loss: {early_stopping.best_loss:.4f}"
        )

    return model, train_losses, val_losses


def train_autoencoder_differential_lr(
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
    encoder_lr: float | None = None,
    decoder_lr: float | None = None,
):
    model.to(device)
    logger.info(f"Device: {device} | Model on: {next(model.parameters()).device}")

    if encoder_lr is None:
        encoder_lr = lr * 0.1
    if decoder_lr is None:
        decoder_lr = lr

    if verbose_validation:
        logger.info("STRATEGY: DIFFERENTIAL LEARNING RATES")
        logger.info(f"Encoder LR: {encoder_lr:.2e} | Decoder LR: {decoder_lr:.2e}")

    if hasattr(model, "unfreeze_encoder"):
        model.unfreeze_encoder()
    elif hasattr(model, "encoder"):
        for p in model.encoder.parameters():
            p.requires_grad = True

    opt = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.to_latent.parameters(), "lr": decoder_lr},
            {"params": model.from_latent.parameters(), "lr": decoder_lr},
            {"params": model.decoder.parameters(), "lr": decoder_lr},
        ]
    )
    loss_fn = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5
    )
    early_stopping = EarlyStopping(
        patience=patience, min_delta=min_delta, verbose=False
    )

    train_losses = []
    val_losses = []

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

        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0
        train_losses.append(avg_train_loss)

        # validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for x, y in val_loader:
                mask = y == 0
                if mask.sum().item() == 0:
                    continue

                x = x[mask].to(device)
                x_reconstructed = model(x)
                loss = loss_fn(x_reconstructed, x)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        val_losses.append(avg_val_loss)

        if verbose_training:
            logger.info(
                f"[Epoch {epoch + 1}/{epochs}] Diff-LR | "
                f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}"
            )

        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            if verbose_validation:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    early_stopping.load_best_model(model)
    if verbose_validation:
        logger.info(
            f"Loaded best model with validation loss: {early_stopping.best_loss:.4f}"
        )

    return model, train_losses, val_losses
