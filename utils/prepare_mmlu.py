import pandas as pd
import datasets
from datasets import load_dataset
from tqdm.auto import tqdm

mmlu=load_dataset("cais/mmlu", "all")
mmlu_dev=mmlu["dev"]
mmlu_test=mmlu["test"]
all_subjects=list(set(mmlu_dev["subject"]))

df=pd.DataFrame(columns=["task", "prompt", "target"])
idx2letter={0:"(A)", 1:"(B)", 2:"(C)", 3:"(D)"}
pbar=tqdm(range(len(all_subjects)))
for subject in all_subjects:
    subject_test_data=mmlu_test.filter(lambda examples: [sub==subject for sub in examples["subject"]], batched=True)
    description=f"The following are multiple choice questions (with answers) about {subject}.\n\n"
    for i in range(len(subject_test_data)):
        question=subject_test_data[i]["question"]
        choices=subject_test_data[i]["choices"]
        target=idx2letter[subject_test_data[i]["answer"]]
        prompt=f"{description}Q:{question}\n(A){choices[0]} (B){choices[1]} (C){choices[2]} (D){choices[3]}\nA: "
        df.loc[i]=[subject, prompt, target]
    pbar.update(1)
dataset=datasets.Dataset.from_pandas(df, preserve_index=False)
dataset.save_to_disk("./../data/mmlu_zeroshot")