"""Download only the pinned tokenizer required by synthetic integration tests."""
from pathlib import Path
from huggingface_hub import hf_hub_download
hf_hub_download('NghiMe/NghiASR', 'bpe.model',
    revision='6192fa584da79827c6325961521ac742a70befb8',
    local_dir=Path(__file__).resolve().parents[1] / 'assets/tokenizer')
