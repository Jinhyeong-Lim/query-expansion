import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification


tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained('roberta-base')

dataset = load_dataset('conllpp')
metric = load_metric('seqeval')

task = "ner"
label_list = dataset['train'].features['ner_tags'].feature.names
id2label = {idx:label for idx, label in enumerate(label_list)}
label2id = {label:idx for idx, label in id2label.items()}
label_all_token = True


def tokenize_and_align_labels(examples):
    # examples is batch of inputs 
    tokenized_input = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)

    labels = []

    for sent_idx, sent_label in enumerate(examples['ner_tags']):

        word_ids = tokenized_input.word_ids(batch_index=sent_idx)
        previous_word_idx = None

        # aligned label ids for current sent idx
        label_ids = []
        for word_idx in word_ids:
            # for special token: [CLS] [SEP]
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(sent_label[word_idx])
            else:
                label_ids.append(sent_label[word_idx] if label_all_token else -100)

            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_input['labels'] = labels
    return tokenized_input


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


data_collator = DataCollatorForTokenClassification(tokenizer)
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
model = AutoModelForTokenClassification.from_pretrained('roberta-base', num_labels=len(label_list), id2label=id2label, label2id=label2id, finetuning_task="ner")
batch_size = 10


args = TrainingArguments(
    f"tesk-{task}",
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy='epoch',
    logging_strategy='epoch',
    logging_dir='./logs'
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
