from src.utils.logging import logger
import typer
from cli import app
import os


@app.command("ts2i")
def full_ts2i_transformation(
    csv_file: str = typer.Option(
        ..., "--csv-file", "-f", help="CSV file with datasets to run"
    ),
    output_dir: str = typer.Option(
        ..., "--output-dir", "-o", help="Output directory to store results"
    ),
    config_file: str = typer.Option(
        ..., "--config-file", "-c", help="Configuration file"
    ),
    device: str = typer.Option(..., "--device", "-d", help="Torch device"),
):
    from src.pipelines.transform_time_series_to_images import run_pipeline

    run_pipeline(csv_file, output_dir, config_file, device)


@app.command("train")
def full_training(
    csv_file: str = typer.Option(
        ..., "--csv-file", "-f", help="CSV file with datasets to run"
    ),
    images_dir: str = typer.Option(
        ..., "--images-dir", "-i", help="Directory with zipped images"
    ),
    output_dir: str = typer.Option(
        ..., "--output-dir", "-o", help="Output directory to store results"
    ),
    config_file: str = typer.Option(
        ..., "--config-file", "-c", help="Configuration file"
    ),
    device: str = typer.Option(..., "--device", "-d", help="Torch device"),
):
    from src.pipelines.train_vision_backbones import run_pipeline

    run_pipeline(csv_file, images_dir, output_dir, config_file, device)


@app.command("verify-gpu")
def verify_gpu():
    """Verify if GPU is available and working."""
    import torch

    print("torch.__version__:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"GPU is available: {torch.cuda.get_device_name(device)}")
        # Test a simple tensor operation
        a = torch.tensor([1.0, 2.0, 3.0], device=device)
        b = torch.tensor([4.0, 5.0, 6.0], device=device)
        c = a + b
        logger.info(f"Tensor addition on GPU successful: {c}")
    else:
        logger.warning("GPU is not available. Please check your CUDA installation.")


if __name__ == "__main__":
    app()
