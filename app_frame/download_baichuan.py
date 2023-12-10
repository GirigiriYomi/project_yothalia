from huggingface_hub import snapshot_download

snapshot_download(repo_id="baichuan-inc/Baichuan2-13B-Chat-4bits", local_dir="./model_weights/baichuan-inc/Baichuan2-13B-Chat-4bits")
snapshot_download(repo_id="baichuan-inc/Baichuan2-7B-Chat-4bits", local_dir="./model_weights/baichuan-inc/Baichuan2-7B-Chat-4bits")