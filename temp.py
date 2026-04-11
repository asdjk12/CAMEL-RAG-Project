from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="BAAI/bge-base-zh",
    local_dir="D:/hf_models/bge-base-zh"
)