"""FareMark: Model-Watermark-Driven Free-Rider Detection in Federated Learning."""

from .watermark import WatermarkKey, extract_watermark, watermark_loss, bit_accuracy
from .client import FLClient
from .server import FLServer
from .models import build_model, build_model_for_dataset
from .datasets import load_dataset, split_iid, make_trigger_loader
from .config import FareMarkConfig
from .train import FareMarkTrainer

__all__ = [
    "WatermarkKey", "extract_watermark", "watermark_loss", "bit_accuracy",
    "FLClient", "FLServer",
    "build_model", "build_model_for_dataset",
    "load_dataset", "split_iid", "make_trigger_loader",
    "FareMarkConfig", "FareMarkTrainer",
]