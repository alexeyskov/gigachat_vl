from dataclasses import dataclass
from typing import Optional, Callable, Type

from torch.utils.data import IterableDataset as TorchIterableDataset

@dataclass
class DatasetConfig:
    """Configuration for each Russian VLM dataset"""
    name: str
    total_samples: Optional[int]
    
    # Core functions and classes
    load_raw_func: Callable  # function that returns raw HF iterable (streaming)
    dataset_class: Type[TorchIterableDataset]  # class that converts raw -> {"image": PIL, "question": str, "answer": str}
    download_func: Optional[Callable] = None  # optional download function
    
    # Metadata
    requires_download: bool = False   # True = needs manual download before use (local files required)