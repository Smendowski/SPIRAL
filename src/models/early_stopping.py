from src.utils.logging import logger


class EarlyStopping:
    def __init__(
        self, patience: int = 10, min_delta: float = 1e-4, verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            if self.verbose:
                logger.debug(f"Initial validation loss: {val_loss:.4f}")
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.debug(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                logger.debug(
                    f"Validation loss improved: {self.best_loss:.4f} → {val_loss:.4f}"
                )
            self.best_loss = val_loss
            self.best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            self.counter = 0

    def load_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
