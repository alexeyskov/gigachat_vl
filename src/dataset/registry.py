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
    DocVQAEnIterableDataset,
)

from src.dataset.sources.vqa.infographicvqa_en import (
    download_infographicvqa_en,
    load_infographicvqa_en,
    InfographicVQAEnIterableDataset,
)
from src.dataset.sources.vqa.chartqa_en import (
    download_chartqa_en,
    load_chartqa_en,
    ChartQAEnIterableDataset,
)
from src.dataset.sources.vqa.textvqa_en import (
    download_textvqa_en,
    load_textvqa_en,
    TextVQAEnIterableDataset,
)
from src.dataset.sources.vqa.mme_en import (
    download_mme_en,
    load_mme_en,
    MMEEnIterableDataset,
)
from src.dataset.sources.vqa.mm_vet_v2_en import (
    download_mm_vet_v2_en,
    load_mm_vet_v2_en,
    MMVetV2EnIterableDataset,
)
from src.dataset.sources.vqa.ok_vqa_train_en import (
    download_ok_vqa_train_en,
    load_ok_vqa_train_en,
    OKVQATrainEnIterableDataset,
)
from src.dataset.sources.vqa.scienceqa_img_en import (
    download_scienceqa_img_en,
    load_scienceqa_img_en,
    ScienceQAImgEnIterableDataset,
)
from src.dataset.sources.vqa.seed_bench_en import (
    download_seed_bench_en,
    load_seed_bench_en,
    SeedBenchEnIterableDataset,
)
from src.dataset.sources.vqa.ai2d_en import (
    download_ai2d_en,
    load_ai2d_en,
    AI2DEnIterableDataset,
)
from src.dataset.sources.vqa.a_okvqa_en import (
    download_a_okvqa_en,
    load_a_okvqa_en,
    AOKVQAEnIterableDataset,
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
        download_func=download_infographicvqa_en,
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

    TEXTVQA_EN = DatasetConfig(
        name="lmms-lab-encoder/textvqa",
        total_samples=34_602,
        load_raw_func=load_textvqa_en,
        dataset_class=TextVQAEnIterableDataset,
        download_func=download_textvqa_en,
        languages=frozenset({Language.EN}),
        tasks=frozenset({DatasetTask.VQA, DatasetTask.OCR}),
    )

    SCIENCEQA_IMG_EN = DatasetConfig(
        name="lmms-lab/ScienceQA-IMG",
        total_samples=12_596,
        load_raw_func=load_scienceqa_img_en,
        dataset_class=ScienceQAImgEnIterableDataset,
        download_func=download_scienceqa_img_en,
        languages=frozenset({Language.EN}),
        tasks=frozenset({DatasetTask.VQA, DatasetTask.VISUAL_REASONING}),
    )

    OK_VQA_TRAIN_EN = DatasetConfig(
        name="Multimodal-Fatima/OK-VQA_train",
        total_samples=None,
        load_raw_func=load_ok_vqa_train_en,
        dataset_class=OKVQATrainEnIterableDataset,
        download_func=download_ok_vqa_train_en,
        languages=frozenset({Language.EN}),
        tasks=frozenset({DatasetTask.VQA, DatasetTask.VISUAL_REASONING}),
    )

    SEED_BENCH_EN = DatasetConfig(
        name="lmms-lab/SEED-Bench",
        total_samples=None,
        load_raw_func=load_seed_bench_en,
        dataset_class=SeedBenchEnIterableDataset,
        download_func=download_seed_bench_en,
        languages=frozenset({Language.EN}),
        tasks=frozenset({DatasetTask.VQA, DatasetTask.VISUAL_REASONING}),
    )

    MM_VET_V2_EN = DatasetConfig(
        name="whyu/mm-vet-v2",
        total_samples=None,
        load_raw_func=load_mm_vet_v2_en,
        dataset_class=MMVetV2EnIterableDataset,
        download_func=download_mm_vet_v2_en,
        languages=frozenset({Language.EN}),
        tasks=frozenset({DatasetTask.VQA, DatasetTask.VISUAL_REASONING}),
    )

    MME_EN = DatasetConfig(
        name="lmms-lab/MME",
        total_samples=None,
        load_raw_func=load_mme_en,
        dataset_class=MMEEnIterableDataset,
        download_func=download_mme_en,
        languages=frozenset({Language.EN}),
        tasks=frozenset({DatasetTask.VQA}),
    )

    AI2D_EN = DatasetConfig(
        name="lmms-lab-encoder/ai2d",
        total_samples=3_088,
        load_raw_func=load_ai2d_en,
        dataset_class=AI2DEnIterableDataset,
        download_func=download_ai2d_en,
        languages=frozenset({Language.EN}),
        tasks=frozenset({DatasetTask.VQA, DatasetTask.VISUAL_REASONING}),
    )

    A_OKVQA_EN = DatasetConfig(
        name="HuggingFaceM4/A-OKVQA",
        total_samples=17_056,
        load_raw_func=load_a_okvqa_en,
        dataset_class=AOKVQAEnIterableDataset,
        download_func=download_a_okvqa_en,
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
