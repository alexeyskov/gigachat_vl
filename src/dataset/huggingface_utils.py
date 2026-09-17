from fnmatch import fnmatch
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, hf_hub_download


def download_parquet_files(
    repo_id: str,
    dataset_root: str,
    force_redownload: bool = False,
    hf_token: Optional[str] = None,
    max_parquet_files: Optional[int] = None,
    parquet_pattern: str = "data/*.parquet",
) -> None:
    dataset_root_path = Path(dataset_root)
    dataset_root_path.mkdir(parents=True, exist_ok=True)

    api = HfApi(token=hf_token)

    repo_files = api.list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
    )

    parquet_files = sorted(
        path
        for path in repo_files
        if fnmatch(path, parquet_pattern)
    )

    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files matching {parquet_pattern!r} "
            f"found in Hugging Face dataset {repo_id}"
        )

    if max_parquet_files is not None:
        if max_parquet_files <= 0:
            raise ValueError(
                f"max_parquet_files must be positive, got {max_parquet_files}"
            )

        parquet_files = parquet_files[:max_parquet_files]

    for repo_path in parquet_files:
        local_path = dataset_root_path / repo_path

        if local_path.exists() and not force_redownload:
            continue

        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=repo_path,
            local_dir=str(dataset_root_path),
            token=hf_token,
            force_download=force_redownload,
        )