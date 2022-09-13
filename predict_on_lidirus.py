import json
import os
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, \
    TrainingArguments, AlbertTokenizerFast

from data import TASK_TO_CONFIG, TASK_TO_NAME
from src import LeanAlbertForSequenceClassification

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

BATCH_SIZE = 8
MAX_LENGTH = 512

TASK = "lidirus"


def main(model_path: Path, arch):
    run_dirname = model_path.parts[-1]
    assert 'terra' in run_dirname

    if arch == 'lean_albert':
        tokenizer = AlbertTokenizerFast.from_pretrained('tokenizer')
    else:
        tokenizer = AutoTokenizer.from_pretrained(arch)
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    dataset = load_dataset("russian_super_glue", TASK)
    config = TASK_TO_CONFIG[TASK](dataset)

    processed_dataset = dataset.map(partial(config.process_data, tokenizer=tokenizer, max_length=512),
                                    batched=True, remove_columns=["label"])
    test_without_labels = processed_dataset['test'].remove_columns(['labels'])

    last_path = max(Path(model_path).glob("checkpoint*"), default=None, key=os.path.getctime)
    with open(last_path / 'trainer_state.json') as f:
        trainer_state = json.load(f)
        best_path = Path(trainer_state['best_model_checkpoint']).parts[-1]

    if arch == 'lean_albert':
        model = LeanAlbertForSequenceClassification.from_pretrained(model_path / best_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path / best_path)

    training_args = TrainingArguments(
        output_dir=model_path, overwrite_output_dir=True,
        evaluation_strategy='epoch', logging_strategy='epoch', per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE, save_strategy='epoch', save_total_limit=1,
        fp16=True, dataloader_num_workers=4, group_by_length=True,
        report_to='none', load_best_model_at_end=True, metric_for_best_model=config.best_metric
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        compute_metrics=config.compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    predictions = trainer.predict(test_dataset=test_without_labels)
    processed_predictions = config.process_predictions(predictions.predictions, split="test")

    preds_dir = f"preds/{run_dirname.replace('_terra', '')}"

    os.makedirs(preds_dir, exist_ok=True)

    with open(f"{preds_dir}/{TASK_TO_NAME[TASK]}.jsonl", 'w+') as outf:
        for prediction in processed_predictions:
            print(json.dumps(prediction, ensure_ascii=True), file=outf)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m", '--model', type=Path, required=True)
    parser.add_argument("-a", '--arch')
    args = parser.parse_args()
    main(args.model, args.arch)
