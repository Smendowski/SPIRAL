import time
import warnings
import pandas as pd
import json
from src.utils.logging import logger
from pathlib import Path
from src.utils.reproducibility import set_reproducibility
from omegaconf import DictConfig, OmegaConf
from src.images.loaders import get_image_loaders
from src.models.cnn_autoencoder.model import CNNAutoEncoder
from src.models.cnn_autoencoder.training import (
    train_autoencoder as train_cnn_autoencoder,
)
from src.models.resnet18_autoencoder.model import ResNet18Autoencoder
from src.models.resnet18_autoencoder.training import (
    train_autoencoder_frozen,
    train_autoencoder_differential_lr,
    train_autoencoder_progressive_unfreeze,
)
from src.models.evaluation import evaluate_tsb_ad_nonoverlapping
from src.models.pvt_autoencoder.model import PVTv2B1Autoencoder
from src.models.pvt_autoencoder.training import (
    train_pvt_autoencoder_frozen,
    train_pvt_autoencoder_progressive_unfreeze,
    train_pvt_autoencoder_differential_lr,
)
import tempfile
import zipfile
import os
import numpy as np
import torch


def run_pipeline(
    csv_file: str, images_dir: str, output_dir: str, config_file: str, device: str
) -> None:
    warnings.filterwarnings("ignore")

    logger.info(f"Loading configuration from {config_file}")
    config: DictConfig = OmegaConf.load(config_file)
    set_reproducibility(config.get("random_seed", 42))

    file_paths = pd.read_csv(csv_file)["file_name"].to_list()
    for _file_path in file_paths:
        file_stem = Path(_file_path).stem

        ts2i_transformations = [
            "RN",
            "LP",
            "SG",
            "GASF",
            "GADF",
            "MTF",
            "RP",
            "RWT",
            "MWT",
            "SPIRAL",
        ]
        for ts2i_transformation in ts2i_transformations:
            dataset = file_stem.split("_")[1]
            zip_name = f"{file_stem}_ts2i_{ts2i_transformation}.zip"
            zip_path = Path(images_dir) / dataset / zip_name

            if not zip_path.exists():
                logger.warning(
                    f"Zip not found: {zip_path}, skipping {ts2i_transformation}"
                )
                continue

            logger.info(
                f"Using zip: {zip_path} for transformation {ts2i_transformation}"
            )

            storage = Path(os.environ.get("PLG_GROUPS_STORAGE")) / "plggaiforsoc"
            with tempfile.TemporaryDirectory(dir=storage) as tmp_dir:
                tmp_base = Path(tmp_dir)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(tmp_base)

                metadata_file = tmp_base / "metadata.json"
                with open(metadata_file, "r") as f:
                    dataset_metadata = json.load(f)

                models = {
                    "cnn_autoencoder": {
                        "model_cls": CNNAutoEncoder,
                        "model_kwargs": {
                            "latent_dim": config["cnn_autoencoder"].get(
                                "latent_dim", 128
                            ),
                        },
                        "training_fn": train_cnn_autoencoder,
                        "training_kwargs": {},
                    },
                    "resnet18_autoencoder_frozen": {
                        "model_cls": ResNet18Autoencoder,
                        "model_kwargs": {
                            "latent_dim": config["resnet18_autoencoder_frozen"].get(
                                "latent_dim", 128
                            ),
                            "freeze_encoder": True,
                        },
                        "training_fn": train_autoencoder_frozen,
                        "training_kwargs": {},
                    },
                    "resnet18_autoencoder_progressive_unfreeze": {
                        "model_cls": ResNet18Autoencoder,
                        "model_kwargs": {
                            "latent_dim": config[
                                "resnet18_autoencoder_progressive_unfreeze"
                            ].get("latent_dim", 128),
                            "freeze_encoder": True,
                        },
                        "training_fn": train_autoencoder_progressive_unfreeze,
                        "training_kwargs": {
                            "unfreeze_epoch": config[
                                "resnet18_autoencoder_progressive_unfreeze"
                            ].get("unfreeze_epoch", 20),
                        },
                    },
                    "resnet18_autoencoder_differential_lr": {
                        "model_cls": ResNet18Autoencoder,
                        "model_kwargs": {
                            "latent_dim": config[
                                "resnet18_autoencoder_differential_lr"
                            ].get("latent_dim", 128),
                            "freeze_encoder": False,
                        },
                        "training_fn": train_autoencoder_differential_lr,
                        "training_kwargs": {
                            "encoder_lr": config[
                                "resnet18_autoencoder_differential_lr"
                            ].get("encoder_lr", None),
                            "decoder_lr": config[
                                "resnet18_autoencoder_differential_lr"
                            ].get("decoder_lr", None),
                        },
                    },
                    "pvt_autoencoder_frozen": {
                        "model_cls": PVTv2B1Autoencoder,
                        "model_kwargs": {
                            "latent_dim": config["pvt_autoencoder_frozen"].get(
                                "latent_dim", 128
                            ),
                            "freeze_encoder": True,
                        },
                        "training_fn": train_pvt_autoencoder_frozen,
                        "training_kwargs": {},
                    },
                    "pvt_autoencoder_progressive_unfreeze": {
                        "model_cls": PVTv2B1Autoencoder,
                        "model_kwargs": {
                            "latent_dim": config[
                                "pvt_autoencoder_progressive_unfreeze"
                            ].get("latent_dim", 128),
                            "freeze_encoder": True,
                        },
                        "training_fn": train_pvt_autoencoder_progressive_unfreeze,
                        "training_kwargs": {
                            "unfreeze_epoch": config[
                                "pvt_autoencoder_progressive_unfreeze"
                            ].get("unfreeze_epoch", 20),
                        },
                    },
                    "pvt_autoencoder_differential_lr": {
                        "model_cls": PVTv2B1Autoencoder,
                        "model_kwargs": {
                            "latent_dim": config["pvt_autoencoder_differential_lr"].get(
                                "latent_dim", 128
                            ),
                            "freeze_encoder": False,
                        },
                        "training_fn": train_pvt_autoencoder_differential_lr,
                        "training_kwargs": {
                            "encoder_lr": config["pvt_autoencoder_differential_lr"].get(
                                "encoder_lr", None
                            ),
                            "decoder_lr": config["pvt_autoencoder_differential_lr"].get(
                                "decoder_lr", None
                            ),
                        },
                    },
                }

                for model_name, metadata in models.items():
                    cfg = config[model_name]

                    train_loader, val_loader, test_loader = get_image_loaders(
                        str(tmp_base),
                        batch_size=cfg["batch_size"],
                        image_size=config["image_size"],
                        seed=config["random_seed"],
                        val_split=cfg["val_split"],
                    )

                    logger.info(f"Train images: {len(train_loader.dataset)}")
                    logger.info(f"Test images: {len(test_loader.dataset)}")

                    model = metadata["model_cls"](**metadata["model_kwargs"])
                    training_fn = metadata["training_fn"]

                    logger.info(f"Training {model_name} on {ts2i_transformation}")
                    train_start = time.perf_counter()
                    model, train_losses, val_losses = training_fn(
                        model,
                        train_loader,
                        val_loader,
                        epochs=cfg["epochs"],
                        lr=cfg["learning_rate"],
                        patience=cfg["patience"],
                        min_delta=cfg["min_delta"],
                        device=device,
                        verbose_training=False,
                        verbose_validation=False,
                        **metadata["training_kwargs"],
                    )
                    train_end = time.perf_counter()

                    logger.info(f"Evaluating {model_name} on {ts2i_transformation}")

                    evaluation_metrics = evaluate_tsb_ad_nonoverlapping(
                        model,
                        test_loader,
                        y_test=np.array(dataset_metadata["y_test"], dtype=np.int32),
                        window_len=dataset_metadata["window_len"],
                        slidingWindow=dataset_metadata[
                            "slidingWindow_TSB_AD_evaluation"
                        ],
                        device=device,
                        output_dir=output_dir,
                        model_name=model_name,
                    )

                    (Path(output_dir)).mkdir(parents=True, exist_ok=True)
                    csv_path = Path(output_dir) / f"{Path(csv_file).stem}.csv"
                    row = {
                        "dataset": dataset_metadata["dataset"],
                        "file_path": file_stem,
                        "ts2i_transformation_pretty_name": ts2i_transformation,
                        "model": model_name,
                        "latent_dim": metadata["model_kwargs"].get("latent_dim", None),
                        "window_length": dataset_metadata["window_len"],
                        "stride": dataset_metadata["stride"],
                        "image_size": config["image_size"],
                        "epochs": cfg["epochs"],
                        "learning_rate": cfg["learning_rate"],
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "AUC_PR": float(evaluation_metrics["AUC-PR"]),
                        "AUC_ROC": float(evaluation_metrics["AUC-ROC"]),
                        "VUS_PR": float(evaluation_metrics["VUS-PR"]),
                        "VUS_ROC": float(evaluation_metrics["VUS-ROC"]),
                        "Standard_F1": float(evaluation_metrics["Standard-F1"]),
                        "PA_F1": float(evaluation_metrics["PA-F1"]),
                        "Event_F1": float(evaluation_metrics["Event-based-F1"]),
                        "R_F1": float(evaluation_metrics["R-based-F1"]),
                        "Affiliation_F": float(evaluation_metrics["Affiliation-F"]),
                        "training_time_s": train_end - train_start,
                        "device": device,
                    }

                    if not csv_path.exists():
                        pd.DataFrame([row]).to_csv(csv_path, index=False)
                    else:
                        pd.DataFrame([row]).to_csv(
                            csv_path, mode="a", header=False, index=False
                        )

                    # Cleanup
                    del train_loader, val_loader, test_loader, model
                    import gc

                    torch.cuda.empty_cache()
                    gc.collect()
