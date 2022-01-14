# Loading the created dataset using datasets
from datasets import load_dataset, load_metric
from transformers import AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
import torchaudio
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "2" # specify gpu for training
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.file_utils import ModelOutput
from src.models import Wav2Vec2ForSpeechClassification
from src.collator import DataCollatorCTCWithPadding
from src.trainer import CTCTrainer
from transformers import TrainingArguments
from utils import compute_metrics
from tensorboardX import SummaryWriter

# dataset files generated by preprocess_data.py
data_files = {
    "train": "dataset/train.csv", 
    "validation": "dataset/test.csv",
}

dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

print('number of sample in training set: ',train_dataset)
print('number of sample in validation set: ', eval_dataset)

# We need to specify the input and output column
input_column = "path"
output_column = "label"

# we need to distinguish the unique labels in our SER dataset
label_list = train_dataset.unique(output_column)
label_list.sort()  # Let's sort it for determinism
num_labels = len(label_list)
print(f"A classification problem with {num_labels} classes: {label_list}")

model_name_or_path = "ckps/wav2vec2-base-100k-classifier/checkpoint-4000" # The path of pretrained model
pooling_mode = "mean"

# config
config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    label2id={label: i for i, label in enumerate(label_list)},
    id2label={i: label for i, label in enumerate(label_list)},
    finetuning_task="wav2vec2_clf",
)
setattr(config, 'pooling_mode', pooling_mode)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path,)
target_sampling_rate = feature_extractor.sampling_rate
print(f"The target sampling rate: {target_sampling_rate}")

def speech_file_to_array_fn(path):
    speech_array, sampling_rate = torchaudio.load(path)
    speech = speech_array.squeeze().numpy()
    return speech

def label_to_id(label, label_list):
    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1
    return label

def preprocess_function(examples):
    speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
    target_list = [label_to_id(label, label_list) for label in examples[output_column]]
    result = feature_extractor(speech_list, sampling_rate=target_sampling_rate)
    result["labels"] = list(target_list)
    return result

# set small dataset for demo training
#max_samples = 1000
#train_dataset = train_dataset.select(range(max_samples))
#eval_dataset = eval_dataset.select(range(max_samples))

print('generating training dataset ...')
train_dataset = train_dataset.map(
    preprocess_function,
    batch_size=16,
    batched=True,
    num_proc=16
)
print('generating validation dataset ...')
eval_dataset = eval_dataset.map(
    preprocess_function,
    batch_size=16,
    batched=True,
    num_proc=16
)

data_collator = DataCollatorCTCWithPadding(feature_extractor=feature_extractor, padding=True)
is_regression = False

model = Wav2Vec2ForSpeechClassification.from_pretrained(
    model_name_or_path,
    config=config,
)
model.freeze_feature_extractor()

training_args = TrainingArguments(
    output_dir="ckps/wav2vec2-base-100k-classifier",
    per_device_train_batch_size=20,
    per_device_eval_batch_size=20,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=12,
    fp16=True,
    save_steps=500,
    eval_steps=500,
    logging_steps=10,
    learning_rate=5e-5,
    save_total_limit=2,
    warmup_ratio=0.1
)



trainer = CTCTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=feature_extractor,
)

trainer.train()
