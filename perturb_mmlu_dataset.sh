mkdir -p data
# Datasets without perturbation
for n_shot in 0 1 2 3 4 5
do
    python3 perturb_mmlu_dataset.py \
        --perturbation_type null \
        --n_shot $n_shot \
        --save_as data/mmlu_${n_shot}_shot_unperturbed
done

# Datasets with grammatical perturbation
perturbation_type="grammatical"
for seed in 0 1 2 4 8 
do
    for n_shot in 0 1 2 3 4 5
    do
        for n_error in 1 2 4 8
        do
            for error_type in "add" "del" "sub" "jux" "mix"
            do
                python3 perturb_mmlu_dataset.py \
                    --perturbation_type $perturbation_type \
                    --n_error $n_error \
                    --error_type $error_type \
                    --seed $seed \
                    --n_shot $n_shot \
                    --save_as data/mmlu_${n_shot}_shot_${perturbation_type}_${error_type}_${n_error}_errors_${seed}_seed
            done
        done
    done
done

# Datasets with synonym perturbation
perturbation_type="synonym"
for seed in 0 1 2 4 8
do  
    for n_shot in 0 1 2 3 4 5
    do
        for n_error in 1 2 4 8
        do
            python3 perturb_mmlu_dataset.py \
                --perturbation_type $perturbation_type \
                --n_error $n_error \
                --seed $seed \
                --n_shot $n_shot \
                --save_as data/mmlu_${n_shot}_shot_${perturbation_type}_${n_error}_errors_${seed}_seed
        done
    done
done