"""Local evaluation tools; VLMEvalKit registration is optional."""
from importlib.util import find_spec

if find_spec("vlmeval") is not None:
    from .mmbench_ru import MMBenchRU
    import vlmeval.dataset as vlmeval_dataset  # type: ignore

    if MMBenchRU not in vlmeval_dataset.DATASET_CLASSES:
        vlmeval_dataset.DATASET_CLASSES.append(MMBenchRU)
        vlmeval_dataset.SUPPORTED_DATASETS.extend(MMBenchRU.supported_datasets())
