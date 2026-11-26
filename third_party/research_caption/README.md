# Caption scoring reference sources

The scorer is an unmodified Python 3 port from
[`salaniz/pycocoevalcap@a24f74c4`](https://github.com/salaniz/pycocoevalcap/tree/a24f74c408c918f1f4ec34e9514bc8a76ce41ffd).
The original Python 2 sources from
[`tylin/coco-caption@3a9afb26`](https://github.com/tylin/coco-caption/tree/3a9afb2682141a03e1cdc02b0df6770d2c884f6f)
are retained for independent parity tests. CIDEr differs only in `xrange`/`range`,
`iteritems`/`items`, and a terminal newline. See the retained BSD license and
`provenance.json` for exact URLs, revisions and SHA256s.

This is the COCO-caption four-gram implementation with clipping, sigma 6
Gaussian length penalty, and native 0–10 scale. A separate explicit ×100 value
may be reported; its range is 0–1000, not VQA accuracy. Single-reference PixMo
evaluation is labeled as such and is not a standard multi-reference COCO score.

`scripts/prepare_research_caption_assets.py` prepares the pinned Stanford
CoreNLP 3.4.1 tokenizer jar and checksum-bound Temurin JRE8 in
`/workspace/.research_eval_assets/caption_v1`. It does not install global Java,
change Python packages, use CUDA or download SPICE models. The asset descriptor
is `configs/research/caption_metric_assets.json`.

The production wrapper uses UTF-8, converts CRLF/CR/LF to spaces before PTB,
retains empty generated captions and checks output line cardinality. It uses
the original punctuation list, including its exact case behavior. Reference
document frequencies come only from the full fixed reference-image corpus;
image bootstrap resamples per-image scores without refitting IDF.

Run the actual JVM/scorer parity checks with:

```bash
CUDA_VISIBLE_DEVICES='' /venv/main/bin/python -m pytest -q tests/test_research_caption_metrics.py
```

This validates a scorer. Checkpoint-bound caption generation and full primary
data/overlap checks remain separate requirements.
