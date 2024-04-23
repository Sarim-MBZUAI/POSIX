# MODEL_NAME_OR_PATH=meta-llama/Llama-2-7b-hf
# MODEL_NAME_OR_PATH=meta-llama/Llama-2-7b-chat-hf
# MODEL_NAME_OR_PATH=mistralai/Mistral-7B-v0.1
MODEL_NAME_OR_PATH=mistralai/Mistral-7B-Instruct-v0.2

mkdir -p responses/${MODEL_NAME_OR_PATH}
# Datasets without perturbation
for n_shot in 0 1 2 3 4 5
do
    accelerate launch --config_file config.yaml get_llm_responses.py \
        --load_data_from_disk \
        --dataset_name_or_path data/mmlu_${n_shot}_shot_unperturbed \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --max_new_tokens 5 \
        --per_device_batch_size 8 \
        --save_as responses/${MODEL_NAME_OR_PATH}/mmlu_${n_shot}_shot_unperturbed.json
done

perturbation_type="grammatical"
for seed in 0 1 2 4 8
do
    for n_shot in 0 1 2 3 4 5
    do
        for n_error in 1 2 4 8
        do
            for error_type in "add" "del" "sub" "jux" "mix"
            do
                accelerate launch --config_file config.yaml get_llm_responses.py \
                    --load_data_from_disk \
                    --dataset_name_or_path data/mmlu_${n_shot}_shot_${perturbation_type}_${error_type}_${n_error}_errors_${seed}_seed \
                    --model_name_or_path $MODEL_NAME_OR_PATH \
                    --max_new_tokens 5 \
                    --per_device_batch_size 8 \
                    --save_as responses/${MODEL_NAME_OR_PATH}/mmlu_${n_shot}_shot_${perturbation_type}_${error_type}_${n_error}_errors_${seed}_seed.json
            done
        done
    done
done

perturbation_type="synonym"
for seed in 0 1 2 4 8
do
    for n_shot in 0 1 2 3 4 5
    do
        for n_error in 1 2 4 8
        do
            accelerate launch --config_file config.yaml get_llm_responses.py \
                --load_data_from_disk \
                --dataset_name_or_path data/mmlu_${n_shot}_shot_${perturbation_type}_${n_error}_errors \
                --model_name_or_path $MODEL_NAME_OR_PATH \
                --max_new_tokens 5 \
                --per_device_batch_size 4 \
                --save_as responses/${MODEL_NAME_OR_PATH}/mmlu_${n_shot}_shot_${perturbation_type}_${n_error}_errors.json
        done
    done
done