from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
        folder_path="./finetuned_starcoder2_rlstep_500",
        repo_id="vdaita/diff-starcoder-7b-rl",
        repo_type="model"
    )
