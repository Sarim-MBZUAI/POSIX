import argparse
import pandas as pd
import datasets
from datasets import load_dataset
from tqdm.auto import tqdm
from sentence_perturber import NullPerturber, GrammaticalPerturber

def parse_args():
    parser=argparse.ArgumentParser(description="Prepare the MMLU datasets with perturbations")
    parser.add_argument(
        "--perturbation_type",
        type=str,
        choices=["null", "grammatical"],
        required=True,
        help="The type of perturbation to apply"
    )
    parser.add_argument(
        "--n_error",
        type=int,
        help="The number of tokens to perturb"
    )
    parser.add_argument(
        "--error_type",
        type=str,
        help="The type of error to apply"
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        help="The number of few shot examples (including 0) to consider"
    )
    parser.add_argument(
        "--save_as",
        type=str,
        required=True,
        help="The name of the output dataset"
    )
    args=parser.parse_args()
    return args

def main():
    args=parse_args()

    mmlu=load_dataset("cais/mmlu", "all")
    mmlu_dev=mmlu["dev"]
    mmlu_test=mmlu["test"]
    all_subjects=[]
    for subject in mmlu_dev["subject"]:
        if subject not in all_subjects:
            all_subjects.append(subject)

    if args.perturbation_type=="null":
        perturber=NullPerturber()
    elif args.perturbation_type=="grammatical":
        perturber=GrammaticalPerturber(
            n_error=args.n_error, 
            error_type=args.error_type
        )

    df=pd.DataFrame(columns=["id", "task", "prompt", "target"])
    idx2letter={0:"(A)", 1:"(B)", 2:"(C)", 3:"(D)"}
    pbar=tqdm(range(len(all_subjects)))
    N=0
    for subject in all_subjects:
        subject_test_data=mmlu_test.filter(lambda examples: [sub==subject for sub in examples["subject"]], batched=True)
        subject_val_data=mmlu_dev.filter(lambda examples: [sub==subject for sub in examples["subject"]], batched=True)
        description=f"The following are multiple choice questions (with answers) about {subject}.\n\n"
        fewshot_demonstrations=""
        if args.n_shot>0:
            for i in range(len(subject_val_data)):
                question=subject_val_data[i]["question"]
                choices=subject_val_data[i]["choices"]
                target=idx2letter[subject_val_data[i]["answer"]]
                fewshot_demonstrations+=f"Q:{question}\n(A){choices[0]} (B){choices[1]} (C){choices[2]} (D){choices[3]}\nA: {target}\n\n"
                if i==args.n_shot-1:
                    break
        for i in range(len(subject_test_data)):
            question=subject_test_data[i]["question"]
            choices=subject_test_data[i]["choices"]
            target=idx2letter[subject_test_data[i]["answer"]]
            perturbed_question=perturber.perturb_sentence(question)
            prompt=f"Q:{perturbed_question}\n(A){choices[0]} (B){choices[1]} (C){choices[2]} (D){choices[3]}\nA: "
            prompt=f"{description}{fewshot_demonstrations}{prompt}"
            df.loc[N]=[N, f"mmlu_{subject}", prompt, target]
            N+=1
        pbar.update(1)
    dataset=datasets.Dataset.from_pandas(df, preserve_index=False)
    dataset.save_to_disk(f"{args.save_as}")

if __name__=="__main__":
    main()