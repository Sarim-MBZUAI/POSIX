accelerate launch --config_file config.yaml get_llm_responses.py \
    --load_data_from_disk \
    --dataset_name_or_path data/mmlu_zeroshot \
    --model_name_or_path mistralai/Mistral-7B-instruct-v0.1 \
    --max_new_tokens 5 \
    --per_device_batch_size 16 \
    --save_as responses/mmlu_zeroshot_mistral_instruct.json