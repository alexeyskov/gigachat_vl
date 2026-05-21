import base64
import io
import re
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image
import pandas as pd

from vlmeval.dataset import ImageMCQDataset # type: ignore
from vlmeval.smp.file import LMUDataRoot # type: ignore
from vlmeval.smp import load, dump # type: ignore


RU2LAT = {
    "А": "A", "а": "A",
    "Б": "B", "б": "B",
    "В": "C", "в": "C",
    "Г": "D", "г": "D",
    "Д": "E", "д": "E",
}

LAT2RU = {
    "A": "А",
    "B": "Б",
    "C": "В",
    "D": "Г",
    "E": "Д",
}

def normalize_ru_mcq_prediction(pred):
    if pd.isna(pred):
        return pred

    s = str(pred).strip()

    if s in RU2LAT:
        return RU2LAT[s]

    m = re.search(
        r"(?i)(?:ответ|вариант|выбор)?\s*[:\-–—]?\s*[\(\[]?\s*([АБВГДабвгд])\s*[\)\].,:;!?]?",
        s,
    )
    if m:
        return RU2LAT[m.group(1)]

    m = re.search(r"\b([ABCDE])\b", s.upper())
    if m:
        return m.group(1)

    return s

def encode_image_to_base64(image_path):
    img = Image.open(image_path).convert("RGB")
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

class MMBenchRU(ImageMCQDataset):
    TYPE = "MCQ"
    DATASET_URL = {}
    DATASET_MD5 = {}
    DATASET_NAME = "MMBenchRU"

    @classmethod
    def supported_datasets(cls):
        return [cls.DATASET_NAME]

    @classmethod
    def prepare_dataset(cls):
        lmu_root = Path(LMUDataRoot())
        tsv_path = lmu_root / f"{cls.DATASET_NAME}.tsv"
        image_dir = lmu_root / "images" / cls.DATASET_NAME

        if tsv_path.exists() and image_dir.exists() and len(list(image_dir.glob("*.png"))) > 0:
            print(f"✅ {cls.DATASET_NAME} already prepared")
            return str(tsv_path)

        print("⏳ Downloading mmbench_ru_dev.parquet...")
        parquet_path = hf_hub_download(
            repo_id="deepvk/MMBench-ru",
            filename="mmbench_ru_dev.parquet",
            repo_type="dataset",
        )

        df = pd.read_parquet(parquet_path)

        image_dir.mkdir(parents=True, exist_ok=True)

        print("⏳ Extracting images...")

        image_cache = {}
        image_paths = []

        for _, row in df.iterrows():
            orig_idx = int(row['index']) % 1_000_000
            
            if orig_idx in image_cache:
                rel_path = image_cache[orig_idx]
                image_paths.append(rel_path)
                continue

            img_data = row["image"]
            img = None
            if isinstance(img_data, dict):
                if 'bytes' in img_data and img_data['bytes']:
                    img = Image.open(io.BytesIO(img_data['bytes']))
                elif 'path' in img_data and img_data['path']:
                    img = Image.open(img_data['path'])
            elif isinstance(img_data, str):
                img = Image.open(img_data)

            if img is None:
                image_paths.append("")
                continue

            img_filename = f"{orig_idx}.png"
            img_path = image_dir / img_filename
            img.save(img_path, format="PNG")

            image_cache[orig_idx] = img_path
            image_paths.append(img_path)

        columns_to_keep = [
            "index", "question", "hint", "A", "B", "C", "D",
            "answer", "category", "source", "l2-category",
            "split", "comment"
        ]
        new_df = df[columns_to_keep].copy()
        new_df["index"] = new_df["index"].astype(str)
        new_df["image_path"] = image_paths

        new_df.to_csv(tsv_path, sep="\t", index=False)

        print(f"✅ Dataset prepared: {tsv_path}")
        print(f"   Unique images saved: {len(image_cache)}")
        return str(tsv_path)
    
    def load_data(self, dataset):
        self.prepare_dataset()
        return super().load_data(dataset)
    
    def evaluate(self, eval_file, **judge_kwargs):
        suffix = eval_file.split(".")[-1]
        data = load(eval_file)

        if "prediction" in data:
            data["prediction"] = data["prediction"].apply(normalize_ru_mcq_prediction)

        normalized_eval_file = eval_file.replace(
            f".{suffix}",
            f"_ru_normalized.{suffix}",
        )
        dump(data, normalized_eval_file)

        return super().evaluate(normalized_eval_file, **judge_kwargs)

    def build_prompt(self, line):
        if isinstance(line, dict):
            line = pd.Series(line)

        image_item = None
        image_path = line.get("image_path")
        if image_path and str(image_path).strip():
            try:
                b64 = encode_image_to_base64(str(image_path))
                image_value = f"data:image/png;base64,{b64}"
                image_item = {"type": "image", "value": image_value}
            except Exception as e:
                print(f"⚠️ Error while encoding {image_path}: {e}")
        
        prompt = ""
        if pd.notna(line.get("hint")) and str(line["hint"]).strip():
            prompt += f"Подсказка: {line['hint']}\n\n"
        prompt += f"Вопрос: {line['question']}\n\n"
        prompt += (
            f"А. {line['A']}\n"
            f"Б. {line['B']}\n"
            f"В. {line['C']}\n"
            f"Г. {line['D']}\n\n"
            "Выбери правильный вариант и ответь только одной русской буквой: А, Б, В или Г."
        )

        content = []
        if image_item:
            content.append(image_item)
        content.append({"type": "text", "value": prompt})
        
        return content