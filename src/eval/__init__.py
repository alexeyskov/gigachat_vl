from .mmbench_ru import MMBenchRU
import vlmeval.dataset as vlmeval_dataset # type: ignore

if MMBenchRU not in vlmeval_dataset.DATASET_CLASSES:
    vlmeval_dataset.DATASET_CLASSES.append(MMBenchRU)
    vlmeval_dataset.SUPPORTED_DATASETS.extend(MMBenchRU.supported_datasets())