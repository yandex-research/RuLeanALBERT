import re
import string
from collections import Counter
from typing import Type, List, Dict

import datasets
import numpy as np
import pandas as pd
import pymorphy2
from Levenshtein import distance
from datasets import load_metric, DatasetDict
from fuzzysearch import find_near_matches
from scipy.special import softmax
from transformers import EvalPrediction


def _quantize_max_length(max_length):
    return max_length - max_length % 8


class DatasetConfig:
    best_metric: str
    num_classes: int

    def __init__(self, d: DatasetDict):
        self.data = d

    @staticmethod
    def process_data(examples, tokenizer, max_length):
        pass

    def compute_metrics(self, p: EvalPrediction, split: str):
        pass

    def process_predictions(self, p: np.ndarray, split: str, **kwargs) -> List[Dict]:
        pass


ACCURACY = load_metric("accuracy")
F1 = load_metric("f1")


class RCBConfig(DatasetConfig):
    best_metric = "f1"
    num_classes = 3

    _index_to_label = {0: "entailment", 1: "contradiction", 2: "neutral"}

    @staticmethod
    def process_data(examples, tokenizer, max_length):
        result = tokenizer(examples['premise'], examples['hypothesis'], truncation='longest_first',
                           return_token_type_ids=True, max_length=_quantize_max_length(max_length), padding=False)

        result["labels"] = examples['label']
        return result

    def compute_metrics(self, p: EvalPrediction, split: str, **kwargs):
        preds = p.predictions
        preds = preds.argmax(axis=1)

        acc_result = ACCURACY.compute(predictions=preds, references=p.label_ids)
        f1_result = F1.compute(predictions=preds, references=p.label_ids, average="macro")

        result = {'accuracy': acc_result['accuracy'], 'f1': f1_result["f1"]}
        return result

    def process_predictions(self, p: np.ndarray, split: str, **kwargs):
        preds_list = p.argmax(axis=1).tolist()

        return [{"idx": idx, "label": self._index_to_label[predicted_class]}
                for idx, predicted_class in enumerate(preds_list)]


class TerraConfig(DatasetConfig):
    best_metric = "accuracy"
    num_classes = 2
    _index_to_label = {0: "entailment", 1: "not_entailment"}

    @staticmethod
    def process_data(examples, tokenizer, max_length):
        result = tokenizer(examples['premise'], examples['hypothesis'], return_token_type_ids=True, padding=False)
        result["labels"] = examples['label']
        return result

    def compute_metrics(self, p: EvalPrediction, split: str, **kwargs):
        preds = p.predictions
        preds = preds.argmax(axis=1)

        acc_result = ACCURACY.compute(predictions=preds, references=p.label_ids)

        result = {'accuracy': acc_result['accuracy']}
        return result

    def process_predictions(self, p: np.ndarray, split: str, **kwargs):
        preds_list = p.argmax(axis=1).tolist()

        return [{"idx": idx, "label": self._index_to_label[predicted_class]}
                for idx, predicted_class in enumerate(preds_list)]


class LiDiRusConfig(TerraConfig):
    @staticmethod
    def process_data(examples, tokenizer, max_length):
        result = tokenizer(examples['sentence1'], examples['sentence2'], return_token_type_ids=True, padding=False)
        result["labels"] = examples['label']
        return result


class DaNetQAConfig(DatasetConfig):
    best_metric = "accuracy"
    num_classes = 2

    @staticmethod
    def process_data(examples, tokenizer, max_length):
        result = tokenizer(examples['passage'], examples['question'], truncation='only_first',
                           return_token_type_ids=True, max_length=_quantize_max_length(max_length), padding=False)
        result["labels"] = examples['label']
        return result

    def compute_metrics(self, p: EvalPrediction, split: str, **kwargs):
        preds = p.predictions
        preds = preds.argmax(axis=1)

        acc_result = ACCURACY.compute(predictions=preds, references=p.label_ids)

        result = {'accuracy': acc_result['accuracy']}
        return result

    def process_predictions(self, p: np.ndarray, split: str, **kwargs):
        preds_list = p.argmax(axis=1).tolist()

        return [{"idx": idx, "label": str(bool(prediction)).lower()} for idx, prediction in enumerate(preds_list)]


class PARusConfig(DatasetConfig):
    best_metric = "accuracy"
    num_classes = 2

    @staticmethod
    def process_data(examples, tokenizer, max_length):
        first_texts = []
        second_texts = []
        labels = []

        for (
            premise, choice1, choice2, question, label
        ) in zip(
            examples["premise"], examples["choice1"], examples["choice2"], examples["question"], examples["label"]
        ):
            if question == "cause":
                first_texts.extend([choice1, choice2])
                second_texts.extend([premise, premise])
            elif question == "effect":
                first_texts.extend([premise, premise])
                second_texts.extend([choice1, choice2])

            if label == 0:
                # could've used [1-label, label], but they mean different things
                labels.extend([1, 0])
            else:  # includes case of -1 for test data, but we remove it later anyway
                labels.extend([0, 1])

        assert len(first_texts) == len(second_texts) == len(labels)

        result = tokenizer(first_texts, second_texts, truncation='longest_first', return_token_type_ids=True,
                           max_length=_quantize_max_length(max_length), padding=False)

        result["labels"] = labels
        return result

    def _get_per_instance_preds(self, flattened_preds):
        true_probs = softmax(flattened_preds, axis=1)[:, 1]
        probs_per_instance = true_probs.reshape((true_probs.shape[0] // 2, 2))
        return np.argmax(probs_per_instance, axis=1)

    def compute_metrics(self, p: EvalPrediction, split: str, **kwargs):
        preds = p.predictions
        per_instance_preds = self._get_per_instance_preds(preds)
        label_ids = p.label_ids.reshape((per_instance_preds.shape[0], 2)).argmax(axis=1)  # [0, 1] -> 1, [1, 0] -> 0

        acc_result = ACCURACY.compute(predictions=per_instance_preds, references=label_ids)
        result = {'accuracy': acc_result['accuracy']}
        return result

    def process_predictions(self, p: np.ndarray, split: str, **kwargs):
        preds = self._get_per_instance_preds(p).tolist()

        return [{"idx": idx, "label": is_same} for idx, is_same in enumerate(preds)]


class MuSeRCConfig(DatasetConfig):
    best_metric = "f1"
    num_classes = 2

    @staticmethod
    def process_data(examples, tokenizer, max_length):
        questions_with_answers = [question + answer for question, answer in
                                  zip(examples['question'], examples['answer'])]

        result = tokenizer(examples['paragraph'], questions_with_answers, truncation='only_first',
                           return_token_type_ids=True, max_length=_quantize_max_length(max_length), padding=False)
        result["labels"] = examples['label']
        return result

    @staticmethod
    def _get_paragraph_metrics(x):
        em = int((x["prediction"] == x["labels"]).all())
        f1_result = F1.compute(predictions=x["prediction"], references=x["labels"], average="binary")
        return pd.Series({"f1": f1_result["f1"], "em": em})

    def compute_metrics(self, p: EvalPrediction, split: str, **kwargs):
        preds = p.predictions.argmax(axis=-1)
        labels = p.label_ids

        split_idx_df = pd.DataFrame(self.data[split]["idx"])

        split_idx_df["prediction"] = preds
        split_idx_df["labels"] = labels

        metrics = split_idx_df.groupby("paragraph").apply(self._get_paragraph_metrics).mean().to_dict()
        return metrics

    @staticmethod
    def _unravel_answers(x):
        x_renamed = x[["answer", "prediction"]].rename(columns={"answer": "idx", "prediction": "label"})
        return {"answers": x_renamed.to_dict("records"), "idx": x.name}

    @staticmethod
    def _unravel_preds(x):
        questions = x.groupby("question").apply(MuSeRCConfig._unravel_answers).to_list()

        result = {"idx": x.name, "passage": {"questions": questions}}
        return result

    def process_predictions(self, p: np.ndarray, split: str, **kwargs):
        split_idx_df = pd.DataFrame(self.data[split]["idx"])
        preds = p.argmax(axis=1)

        split_idx_df["prediction"] = preds
        result = split_idx_df.groupby("paragraph").apply(self._unravel_preds).to_list()
        return result


class RUSSEConfig(DatasetConfig):
    best_metric = "accuracy"
    num_classes = 2
    _index_to_label = {0: "false", 1: "true"}

    PUNCTUATION_TO_REMOVE = ' «»—-…\xa0' + string.punctuation + ''.join(map(str, range(10)))
    morph = pymorphy2.MorphAnalyzer()

    @staticmethod
    def _get_leading_trailing_spaces(x):
        start_len = len(x)
        len_lstrip = len(x.lstrip(RUSSEConfig.PUNCTUATION_TO_REMOVE))
        len_rstrip = len(x.rstrip(RUSSEConfig.PUNCTUATION_TO_REMOVE))

        return start_len - len_lstrip, start_len - len_rstrip

    @staticmethod
    def process_data(examples, tokenizer, max_length):
        examples['sentence1'] = list(map(lambda x: x.replace(u'\xa0', ' '), examples['sentence1']))
        examples['sentence2'] = list(map(lambda x: x.replace(u'\xa0', ' '), examples['sentence2']))

        result = tokenizer(examples['sentence1'], examples['sentence2'], truncation='longest_first',
                           return_token_type_ids=True, max_length=_quantize_max_length(max_length), padding=False)

        e1_masks = []
        e2_masks = []

        for example_index, (word, start1, start2, end1, end2, input_ids) in enumerate(zip(
            examples["word"], examples["start1"], examples["start2"], examples["end1"], examples["end2"],
            result["input_ids"],
        )):

            assert all(char not in word for char in RUSSEConfig.PUNCTUATION_TO_REMOVE)

            e1_mask = np.zeros((len(input_ids),), dtype=bool)
            e2_mask = np.zeros((len(input_ids),), dtype=bool)

            # fill parts of mask corresponding to the target entity
            for seq_index, (mask, start, end) in enumerate(zip((e1_mask, e2_mask), (start1, start2), (end1, end2))):
                sentence_idx = "sentence1" if seq_index == 0 else "sentence2"

                # fix error 1: end is longer than the length of the sentence
                input_sentence = examples[sentence_idx][example_index]
                end = min(end, len(input_sentence))

                # fix error 2: spans include spaces
                input_string = input_sentence[start:end]
                leading_spaces, trailing_spaces = RUSSEConfig._get_leading_trailing_spaces(input_string)
                start += leading_spaces
                end -= trailing_spaces

                # fix error 3: spans are too short
                while start > 0 and input_sentence[start - 1] not in RUSSEConfig.PUNCTUATION_TO_REMOVE:
                    start -= 1
                while end < len(input_sentence) and input_sentence[end] not in RUSSEConfig.PUNCTUATION_TO_REMOVE:
                    end += 1
                input_string_strip = input_sentence[start:end]

                # fix error 4: the word is different from the substring given by indices
                if (
                    distance(input_string_strip.lower(), word) > 1 and
                    all(distance(parse_result.normal_form, word_parse_result.normal_form) > 1
                        for parse_result in RUSSEConfig.morph.parse(input_string_strip)
                        for word_parse_result in RUSSEConfig.morph.parse(word))
                ):
                    print(f"Levenshtein distance exceeds 1 for {input_sentence} "
                          f"({word}!={input_string_strip}, {start}, {end}). "
                          f"Resorting to fuzzy search")

                    parse_result = RUSSEConfig.morph.parse(word)[0]

                    best_match = None
                    best_match_diff = 999

                    for lexeme in parse_result.lexeme:
                        matches = find_near_matches(lexeme.word, input_sentence, max_l_dist=0)

                        for match in matches:
                            if (
                                abs(match.start - start) + abs(match.start - start) < best_match_diff
                                or (
                                abs(match.start - start) + abs(match.start - start) == best_match_diff
                                and len(match.matched) > len(best_match.matched)
                            )):
                                best_match = match
                                best_match_diff = abs(match.start - start) + abs(match.start - start)
                            elif (abs(match.start - start) + abs(match.start - start) == best_match_diff
                                  and len(match.matched) > len(best_match.matched)):
                                best_match = match
                                best_match_diff = abs(match.start - start) + abs(match.start - start)

                    if best_match is not None:
                        print(f"Found {best_match.start}:{best_match.end} ({best_match.matched})")
                        start = best_match.start
                        end = best_match.end
                    else:
                        print(f"!!! Could not find a suitable match for {examples['idx'][example_index]}. Giving up")

                entity_start = result.char_to_token(example_index, start, sequence_index=seq_index)
                assert entity_start is not None, (start, end, seq_index, input_sentence)

                # end-1 because [start:end] denotes a slice and we need the last character
                entity_end = result.char_to_token(example_index, end - 1, sequence_index=seq_index)
                assert entity_end is not None, (
                    start, end, input_sentence, input_ids, example_index, examples["idx"][example_index], seq_index)

                entity_tokens = result["input_ids"][example_index][entity_start:entity_end + 1]
                # here we verify that the indexing is correct
                assert tokenizer.decode(entity_tokens) in input_sentence, tokenizer.decode(
                    entity_tokens)

                mask[entity_start:entity_end + 1] = 1

            e1_masks.append(e1_mask)
            e2_masks.append(e2_mask)

        result["e1_mask"] = e1_masks
        result["e2_mask"] = e2_masks

        result["labels"] = examples['label']
        return result

    def compute_metrics(self, p: EvalPrediction, split: str, **kwargs):
        preds = p.predictions
        preds = np.argmax(preds, axis=1)

        acc_result = ACCURACY.compute(predictions=preds, references=p.label_ids)

        result = {'accuracy': acc_result['accuracy']}
        return result

    def process_predictions(self, p: np.ndarray, split: str, **kwargs):
        preds_list = p.argmax(axis=1).tolist()

        return [{"idx": idx, "label": self._index_to_label[predicted_class]}
                for idx, predicted_class in enumerate(preds_list)]


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace.
    From official ReCoRD eval script """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """ Compute normalized token level F1
    From official ReCoRD eval script """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """ Compute normalized exact match
    From official ReCoRD eval script """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """ Compute max metric between prediction and each ground truth.
    From official ReCoRD eval script """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


class RuCoSConfig(DatasetConfig):
    best_metric = "f1"
    num_classes = 2

    PUNCTUATION_TO_REMOVE = ' ' + string.punctuation

    @staticmethod
    def _get_leading_trailing_spaces(x):
        start_len = len(x)
        len_lstrip = len(x.lstrip(RuCoSConfig.PUNCTUATION_TO_REMOVE))
        len_rstrip = len(x.rstrip(RuCoSConfig.PUNCTUATION_TO_REMOVE))

        return start_len - len_lstrip, start_len - len_rstrip

    def process_data(self, examples, tokenizer, max_length):
        query_answers = []
        query_texts = []
        query_ids = []
        passage_entities = []
        passage_texts = []

        entity_masks = []
        target_masks = []

        for passage, qas in zip(examples["passage"], examples["qas"]):
            assert len(qas) == 1, examples["qas"]
            passage_text = passage["text"].replace('\u200b', " ")
            passage_texts.append(passage_text)

            entities = []
            for entity in passage["entities"]:
                entity_text = passage_text[entity["start"]:entity["end"]]
                leading_spaces, trailing_spaces = self._get_leading_trailing_spaces(entity_text)
                entity["start"] += leading_spaces
                entity["end"] -= trailing_spaces
                if entity["start"] < entity["end"]:
                    entity["text"] = passage_text[entity["start"]:entity["end"]]
                    entities.append(entity)
                else:
                    print(f'Discarding entity "{entity_text}" ({entity}) due to it consisting entirely of punctuation')

            qa = qas[0]
            query_texts.append(qa["query"])
            query_ids.append(qa["idx"])

            if "answers" in qa:
                for answer in qa["answers"]:  # `default` handles the test set
                    answer_text = passage_text[answer["start"]:answer["end"]]
                    assert answer_text == answer["text"]

                    leading_spaces, trailing_spaces = self._get_leading_trailing_spaces(answer_text)
                    answer["start"] += leading_spaces
                    answer["end"] -= trailing_spaces

                    assert answer["start"] < answer["end"]

                    if answer not in entities:
                        entities.append(answer)

                query_answers.append(qa["answers"])

            passage_entities.append(entities)

        result = tokenizer(passage_texts, query_texts, truncation='only_first',
                           return_token_type_ids=True, max_length=_quantize_max_length(max_length), padding=False
                           )

        entities = []

        lengths = []

        for example_index, (example_entities, input_ids) in enumerate(
            zip(passage_entities, result["input_ids"])
        ):
            input_length = len(input_ids)
            lengths.append(input_length)

            example_entity_masks = []
            example_target_masks = []
            example_correct_entities = []

            for entity in example_entities:
                start = entity["start"]
                end = entity["end"]

                entity_start = result.char_to_token(example_index, start, sequence_index=0)

                # end-1 because [start:end] denotes a slice, we need the last character
                entity_end = result.char_to_token(example_index, end - 1, sequence_index=0)

                if entity_start is not None and entity_end is not None:
                    entity_mask = np.zeros((input_length,), dtype=int)
                    entity_mask[entity_start:entity_end + 1] = 1
                    example_entity_masks.append(entity_mask)

                    if query_answers:
                        if entity in query_answers[example_index]:
                            example_target_masks.append(1)
                        else:
                            example_target_masks.append(0)

                    example_correct_entities.append(entity)
                else:
                    print(examples["passage"][example_index]["text"], examples["passage"][example_index]["entities"])

            entity_masks.append(np.stack(example_entity_masks))
            target_masks.append(np.array(example_target_masks))
            entities.append(example_correct_entities)

        result["entity_mask"] = entity_masks
        result["entities"] = entities
        result["idx"] = examples["idx"]
        result["length"] = lengths

        if query_answers:
            result["answers"] = query_answers
            result["labels"] = target_masks

        return result

    def compute_metrics(self, p: EvalPrediction, processed_dataset: datasets.Dataset, **kwargs):
        preds = p.predictions

        predicted_entities = np.argmax(preds, axis=1)

        text_entities = processed_dataset["entities"]
        text_answers = processed_dataset["answers"]

        f1_values = []
        em_values = []

        for pred_idx, entities, targets in zip(predicted_entities, text_entities, text_answers):
            prediction = entities[pred_idx]["text"]
            target_texts = [answer["text"] for answer in targets]

            f1 = metric_max_over_ground_truths(f1_score, prediction, target_texts)
            f1_values.append(f1)

            em = metric_max_over_ground_truths(exact_match_score, prediction, target_texts)
            em_values.append(em)

        result = {'f1': np.mean(f1_values), "em": np.mean(em_values)}
        return result

    def process_predictions(self, p: np.ndarray, processed_dataset: datasets.Dataset, **kwargs) -> List[Dict]:
        preds_list = p.argmax(axis=1).tolist()

        text_entities = processed_dataset["entities"]

        return [{"idx": idx, "label": text_entities[idx][predicted_entity]["text"]}
                for idx, predicted_entity in enumerate(preds_list)]


MCC = load_metric("matthews_correlation")


class RuCoLAConfig(DatasetConfig):
    best_metric = "mcc"
    num_classes = 2

    @staticmethod
    def process_data(examples, tokenizer, max_length):
        result = tokenizer(examples['sentence'], return_token_type_ids=True,
                           max_length=_quantize_max_length(max_length), padding=False)

        if "acceptable" in examples:
            result["labels"] = examples['acceptable']
        return result

    def compute_metrics(self, p: EvalPrediction, split: str, **kwargs):
        preds = p.predictions
        preds = np.argmax(preds, axis=1)

        acc_result = ACCURACY.compute(predictions=preds, references=p.label_ids)
        mcc_result = MCC.compute(predictions=preds, references=p.label_ids)

        result = {"accuracy": acc_result["accuracy"], "mcc": mcc_result["matthews_correlation"]}

        return result

    def process_predictions(self, p: np.ndarray, split: str, **kwargs):
        preds_list = p.argmax(axis=1).tolist()

        return [{"id": idx, "acceptable": predicted_class}
                for idx, predicted_class in enumerate(preds_list)]


TASK_TO_CONFIG: Dict[str, Type[DatasetConfig]] = {
    "rcb": RCBConfig,
    "terra": TerraConfig,
    "danetqa": DaNetQAConfig,
    "lidirus": LiDiRusConfig,
    "parus": PARusConfig,
    "muserc": MuSeRCConfig,
    "russe": RUSSEConfig,
    "rucos": RuCoSConfig,
    "rucola": RuCoLAConfig,
}

TASK_TO_NAME = {
    "rcb": "RCB",
    "terra": "TERRa",
    "danetqa": "DaNetQA",
    "rwsd": "RWSD",
    "lidirus": "LiDiRus",
    "parus": "PARus",
    "muserc": "MuSeRC",
    "russe": "RUSSE",
    "rucos": "RuCoS",
    "rucola": "RuCoLA",
}
