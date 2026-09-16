from enum import Enum

from src.dataset.dataset_base import DatasetConfig, DatasetTask, Language


from src.dataset.sources.captioning.llava_pretrain_ru import (
    download_llava_pretrain_ru,
    load_llava_pretrain_ru,
    LLaVAPretrainRuIterableDataset,
)
from src.dataset.sources.captioning.mscoco_caption_ml import (
    download_mscoco_caption_ml,
    load_mscoco_caption_ml,
    MSCOCOCaptionMlIterableDataset,
)
from src.dataset.sources.captioning.pixmo_cap_en import (
    download_pixmo_cap_en,
    load_pixmo_cap_en,
    PixMoCapEnIterableDataset,
)


from src.dataset.sources.ocr.rustitw_ocr_ru import (
    download_rustitw_ocr,
    load_rustitw_ocr,
    RusTitWOCRIterableDataset,
)


from src.dataset.sources.text.openhermes_ru import (
    download_openhermes_ru_text,
    load_openhermes_ru_text,
    OpenHermesRuIterableDataset,
)
from src.dataset.sources.text.smoltalk2_ml import (
    download_smoltalk2,
    load_smoltalk2_sft,
    SmolTalk2IterableDataset,
)


from src.dataset.sources.vqa.gqa_ru import (
    download_gqa_ru, 
    load_gqa_ru, 
    GQARUIterableDataset,
)
from src.dataset.sources.vqa.llava_instruct_ru import (
    download_llava_instruct_ru,
    load_llava_instruct_ru,
    LLaVAInstructRuIterableDataset,
)

from src.dataset.sources.vqa.mws_vision_ru import (
    download_mws_vision,
    load_mws_vision,
    MWSVisionIterableDataset,
)
from src.dataset.sources.vqa.pixmo_ask_model_anything_en import (
    download_pixmo_ask_model_anything_en,
    load_pixmo_ask_model_anything_en,
    PixMoAskModelAnythingEnIterableDataset,
)
from src.dataset.sources.vqa.docvqa_en import (
    download_docvqa_en,
    load_docvqa_en,
    load_infographicvqa_en,
    DocVQAEnIterableDataset,
    InfographicVQAEnIterableDataset,
)
from src.dataset.sources.vqa.chartqa_en import (
    download_chartqa_en,
    load_chartqa_en,
    ChartQAEnIterableDataset,
)
from src.dataset.sources.vqa.vlm_reasoning_sft_ru import (
    download_ru_vlm_reasoning_sft,
    load_ru_vlm_reasoning_sft,
    RuVLMReasoningSFTIterableDataset,
)

class SupportedDatasets(Enum):
    LLAVA_PRETRAIN_RU = DatasetConfig(
        name="maya-multimodal/pretrain",
        total_samples=550_000,
        load_raw_func=load_llava_pretrain_ru,
        dataset_class=LLaVAPretrainRuIterableDataset,
        download_func=download_llava_pretrain_ru,
        languages=frozenset({Language.RU}),
        tasks=frozenset({DatasetTask.CAPTIONING}),
    )

    MSCOCO_CAPTION_ML = DatasetConfig(
        name="piyushsinghpasi/mscoco-multilingual-30k",
        total_samples=30_000,
        load_raw_func=load_mscoco_caption_ml,
        dataset_class=MSCOCOCaptionMlIterableDataset,
        download_func=download_mscoco_caption_ml,
        languages=frozenset({Language.RU}),
        tasks=frozenset({DatasetTask.CAPTIONING}),
    )

    GQA_RU = DatasetConfig(
        name="deepvk/GQA-ru",
        total_samples=52_216,
        load_raw_func=load_gqa_ru,
        dataset_class=GQARUIterableDataset,
        download_func=download_gqa_ru,
        languages=frozenset({Language.RU}),
        tasks=frozenset({DatasetTask.VQA}),
    )

    LLAVA_INSTRUCT_RU = DatasetConfig(
        name="deepvk/LLaVA-Instruct-ru",
        total_samples=143_980,
        load_raw_func=load_llava_instruct_ru,
        dataset_class=LLaVAInstructRuIterableDataset,
        download_func=download_llava_instruct_ru,
        languages=frozenset({Language.RU}),
        tasks=frozenset({DatasetTask.VQA}),
    )

    RUSTITW_OCR = DatasetConfig(
        name="rustitw_ru",
        total_samples=28_000,
        load_raw_func=load_rustitw_ocr,
        dataset_class=RusTitWOCRIterableDataset,
        download_func=download_rustitw_ocr,
        languages=frozenset({Language.RU}),
        tasks=frozenset({DatasetTask.OCR}),
    )

    OPENHERMES_RU_TEXT = DatasetConfig(
        name="d0rj/OpenHermes-2.5-ru",
        total_samples=1_000_000,
        load_raw_func=load_openhermes_ru_text,
        dataset_class=OpenHermesRuIterableDataset,
        download_func=download_openhermes_ru_text,
        languages=frozenset({Language.RU}),
        tasks=frozenset({DatasetTask.TEXT_INSTRUCTION}),
    )

    SMOLTALK2_SFT = DatasetConfig(
        name="HuggingFaceTB/smoltalk2",
        total_samples=None,
        load_raw_func=load_smoltalk2_sft,
        dataset_class=SmolTalk2IterableDataset,
        download_func=download_smoltalk2,
        languages=frozenset({Language.MULTILINGUAL}),
        tasks=frozenset({DatasetTask.TEXT_INSTRUCTION}),
    )

    MWS_VISION = DatasetConfig(
        name="MTSAIR/MWS-Vision-Bench",
        total_samples=1302,
        load_raw_func=load_mws_vision,
        dataset_class=MWSVisionIterableDataset,
        download_func=download_mws_vision,
        languages=frozenset({Language.RU}),
        tasks=frozenset(
            {DatasetTask.VQA, DatasetTask.OCR, DatasetTask.VISUAL_REASONING}
        ),
    )

    PIXMO_CAP_EN = DatasetConfig(
        name="dnth/pixmo-cap-images",
        total_samples=46_000,
        load_raw_func=load_pixmo_cap_en,
        dataset_class=PixMoCapEnIterableDataset,
        download_func=download_pixmo_cap_en,
        languages=frozenset({Language.EN}),
        tasks=frozenset({DatasetTask.CAPTIONING}),
    )

    PIXMO_ASK_MODEL_ANYTHING_EN = DatasetConfig(
        name="dnth/pixmo-ask-model-anything-images",
        total_samples=153_592,
        load_raw_func=load_pixmo_ask_model_anything_en,
        dataset_class=PixMoAskModelAnythingEnIterableDataset,
        download_func=download_pixmo_ask_model_anything_en,
        languages=frozenset({Language.EN}),
        tasks=frozenset({DatasetTask.VQA}),
    )

    # DocVQA and InfographicVQA are downloaded together by download_docvqa_en(...)
    # and share the same dataset_root. They are loaded as separate datasets because
    # their parquet schemas are different.
    DOCVQA_EN = DatasetConfig(
        name="lmms-lab/DocVQA/DocVQA",
        total_samples=10_500,
        load_raw_func=load_docvqa_en,
        dataset_class=DocVQAEnIterableDataset,
        download_func=download_docvqa_en,
        languages=frozenset({Language.EN}),
        tasks=frozenset({DatasetTask.VQA, DatasetTask.OCR}),
    )
    INFOGRAPHICVQA_EN = DatasetConfig(
        name="lmms-lab/DocVQA/InfographicVQA",
        total_samples=6_090,
        load_raw_func=load_infographicvqa_en,
        dataset_class=InfographicVQAEnIterableDataset,
        download_func=download_docvqa_en,
        languages=frozenset({Language.EN}),
        tasks=frozenset({DatasetTask.VQA, DatasetTask.OCR}),
    )

    CHARTQA_EN = DatasetConfig(
        name="lmms-lab/ChartQA",
        total_samples=2_500,
        load_raw_func=load_chartqa_en,
        dataset_class=ChartQAEnIterableDataset,
        download_func=download_chartqa_en,
        languages=frozenset({Language.EN}),
        tasks=frozenset({DatasetTask.VQA, DatasetTask.VISUAL_REASONING}),
    )

    RU_VLM_REASONING_SFT = DatasetConfig(
        name="mnezhinskii/ru-vlm-reasoning-sft",
        total_samples=3_338,
        load_raw_func=load_ru_vlm_reasoning_sft,
        dataset_class=RuVLMReasoningSFTIterableDataset,
        download_func=download_ru_vlm_reasoning_sft,
        languages=frozenset({Language.RU}),
        tasks=frozenset({DatasetTask.VQA, DatasetTask.VISUAL_REASONING}),
    )
