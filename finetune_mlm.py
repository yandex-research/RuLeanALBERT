import json
import os
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from shutil import rmtree

import numpy as np
import pandas as pd
import torch
import transformers.utils.logging
import wandb
from datasets import load_dataset, Dataset, DatasetDict
from transformers import RobertaTokenizerFast, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, \
    TrainingArguments, AlbertTokenizerFast, AutoModel
from transformers.trainer_utils import set_seed

from data import TASK_TO_CONFIG, TASK_TO_NAME
from field_collator import FieldDataCollatorWithPadding
from models import SpanClassificationModel, EntityChoiceModel
from src import LeanAlbertConfig, LeanAlbertForSequenceClassification, LeanAlbertForPreTraining


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


MODEL_TO_HUB_NAME = {
    'ruroberta-large': "sberbank-ai/ruRoberta-large",
}

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

N_EPOCHS = 40
LEARNING_RATE = 1e-5
MAX_LENGTH = 512


def main(task, model_name, checkpoint_path, data_dir, batch_size, grad_acc_steps, dropout, weight_decay, num_seeds):
    if checkpoint_path is not None:
        tokenizer = AlbertTokenizerFast.from_pretrained('tokenizer')
        assert model_name is None
        model_name = 'lean_albert'
    else:
        tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_TO_HUB_NAME[model_name],
                                                         cache_dir=data_dir / 'transformers_cache')

    if task == "russe":
        data_collator = FieldDataCollatorWithPadding(tokenizer, fields_to_pad=(("e1_mask", 0, 0), ("e2_mask", 0, 0)),
                                                     pad_to_multiple_of=8)
    elif task == "rucos":
        data_collator = FieldDataCollatorWithPadding(tokenizer=tokenizer,
                                                     fields_to_pad=(
                                                         ("entity_mask", 0, 1),
                                                         ("labels", -1, None)
                                                     ))
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    if task == "rucos":
        # we're using a custom dataset, because the HF hub version has no information about entity indices
        train = Dataset.from_json(["RuCoS/train.jsonl"])
        val = Dataset.from_json(["RuCoS/val.jsonl"])
        test = Dataset.from_json(["RuCoS/test.jsonl"])
        dataset = DatasetDict(train=train, validation=val, test=test)
    elif task == "rucola":
        train_df, in_domain_dev_df, out_of_domain_dev_df, test_df = map(
            pd.read_csv, ("RuCoLA/in_domain_train.csv", "RuCoLA/in_domain_dev.csv", "RuCoLA/out_of_domain_dev.csv",
                          "RuCoLA/test.csv")
        )

        # concatenate datasets to get aggregate metrics
        dev_df = pd.concat((in_domain_dev_df, out_of_domain_dev_df))
        train, dev, test = map(Dataset.from_pandas, (train_df, dev_df, test_df))
        dataset = DatasetDict(train=train, validation=dev, test=test)
    else:
        dataset = load_dataset("russian_super_glue", task)

    config = TASK_TO_CONFIG[task](dataset)

    processed_dataset = dataset.map(partial(config.process_data, tokenizer=tokenizer, max_length=MAX_LENGTH),
                                    num_proc=32, keep_in_memory=True, batched=True)

    if "labels" in processed_dataset["test"].column_names:
        test_without_labels = processed_dataset['test'].remove_columns(['labels'])
    else:
        test_without_labels = processed_dataset["test"]

    transformers.utils.logging.enable_progress_bar()

    model_prefix = f"{model_name}_" \
                   f"{task}_" \
                   f"dr{dropout}_" \
                   f"wd{weight_decay}_" \
                   f"bs{batch_size * grad_acc_steps}"

    dev_metrics_per_run = []
    predictions_per_run = []

    for seed in range(num_seeds):
        set_seed(seed)

        if checkpoint_path is not None:
            model_config = LeanAlbertConfig.from_pretrained('config.json')
            model_config.num_labels = config.num_classes
            model_config.classifier_dropout_prob = dropout

            if task in ("russe", "rucos"):
                model = LeanAlbertForPreTraining(model_config)
            else:
                model = LeanAlbertForSequenceClassification(model_config)

            model.resize_token_embeddings(len(tokenizer))
            checkpoint = torch.load(checkpoint_path, map_location='cpu')['model']
            incompat_keys = model.load_state_dict(checkpoint, strict=False)
            print("missing", incompat_keys.missing_keys)
            print("unexpected", incompat_keys.unexpected_keys)

            if task in ("russe", "rucos"):
                model = model.albert
        else:
            if task in ("russe", "rucos"):
                model = AutoModel.from_pretrained(MODEL_TO_HUB_NAME[model_name],
                                                  attention_probs_dropout_prob=dropout,
                                                  hidden_dropout_prob=dropout,
                                                  cache_dir=data_dir / 'transformers_cache')
            else:
                model = AutoModelForSequenceClassification.from_pretrained(MODEL_TO_HUB_NAME[model_name],
                                                                           num_labels=config.num_classes,
                                                                           attention_probs_dropout_prob=dropout,
                                                                           hidden_dropout_prob=dropout,
                                                                           cache_dir=data_dir / 'transformers_cache')

        if task == "russe":
            model = SpanClassificationModel(model, num_labels=config.num_classes)
        elif task == "rucos":
            model = EntityChoiceModel(model)

        run_base_dir = f"{model_prefix}_{seed}"

        run = wandb.init(project='brbert', name=run_base_dir)
        run.config.update({"task": task, "model": model_name, "checkpoint": str(checkpoint_path)})

        training_args = TrainingArguments(
            output_dir=data_dir / 'checkpoints' / run_base_dir, overwrite_output_dir=True,
            evaluation_strategy='epoch', logging_strategy='epoch', logging_first_step=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size, gradient_accumulation_steps=grad_acc_steps,
            optim="adamw_torch", learning_rate=LEARNING_RATE, weight_decay=weight_decay,
            num_train_epochs=N_EPOCHS, warmup_ratio=0.1, save_strategy='epoch',
            seed=seed, fp16=True, dataloader_num_workers=4, group_by_length=True,
            report_to='wandb', run_name=run_base_dir, save_total_limit=1,
            load_best_model_at_end=True, metric_for_best_model=config.best_metric
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset['train'],
            eval_dataset=processed_dataset['validation'],
            compute_metrics=partial(config.compute_metrics, split="validation",
                                    processed_dataset=processed_dataset["validation"]),
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        train_result = trainer.train()
        print(run_base_dir)
        print('train', train_result.metrics)

        dev_predictions = trainer.predict(test_dataset=processed_dataset['validation'])
        print('dev', dev_predictions.metrics)

        run.summary.update(dev_predictions.metrics)
        wandb.finish()

        dev_metrics_per_run.append(dev_predictions.metrics[f"test_{config.best_metric}"])

        predictions = trainer.predict(test_dataset=test_without_labels)
        predictions_per_run.append(predictions.predictions)

        if task != "terra":
            rmtree(data_dir / 'checkpoints' / run_base_dir)

    best_run = np.argmax(dev_metrics_per_run)
    best_predictions = predictions_per_run[best_run]
    processed_predictions = config.process_predictions(best_predictions, split="test",
                                                       processed_dataset=processed_dataset["test"])

    prefix_without_task = model_prefix.replace(f"{task}_", "")

    os.makedirs(f"preds/{prefix_without_task}", exist_ok=True)

    if task == "rucola":
        result_df = pd.DataFrame.from_records(processed_predictions, index="id")
        result_df.to_csv(f"preds/{prefix_without_task}/{TASK_TO_NAME[task]}.csv")
    else:
        with open(f"preds/{prefix_without_task}/{TASK_TO_NAME[task]}.jsonl", 'w+') as outf:
            for prediction in processed_predictions:
                print(json.dumps(prediction, ensure_ascii=True, cls=NumpyEncoder), file=outf)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-t", '--task', choices=TASK_TO_CONFIG.keys())
    parser.add_argument("-m", '--model-name', choices=MODEL_TO_HUB_NAME.keys())
    parser.add_argument("-c", '--checkpoint', type=Path)
    parser.add_argument("-d", "--data-dir", type=Path)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--grad-acc-steps", required=True, type=int)
    parser.add_argument("--dropout", required=True, type=float)
    parser.add_argument("--weight-decay", required=True, type=float)
    parser.add_argument("--num-seeds", required=True, type=int)
    args = parser.parse_args()
    main(args.task, args.model_name, args.checkpoint, args.data_dir, args.batch_size, args.grad_acc_steps,
         args.dropout, args.weight_decay, args.num_seeds)
