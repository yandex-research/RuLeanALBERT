import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import DataCollatorForLanguageModeling

try:
    from transformers.data.data_collator import _torch_collate_batch as collate_batch
    from transformers.data.data_collator import tolist
except ImportError:
    from transformers.data.data_collator import _collate_batch as collate_batch, tolist

from transformers.tokenization_utils_base import BatchEncoding


def _is_start_piece_sp(piece):
    """Check if the current word piece is the starting piece (sentence piece)."""
    special_pieces = set(list('!"#$%&"()*+,-./:;?@[\\]^_`{|}~'))
    special_pieces.add(u"€".encode("utf-8"))
    special_pieces.add(u"£".encode("utf-8"))
    if piece.startswith("▁") or piece.startswith("<") or piece in special_pieces:
        return True
    else:
        return False


@dataclass
class DataCollatorForWholeWordMask(DataCollatorForLanguageModeling):
    """
    Data collator used for language modeling that masks entire words.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling

    .. note::

        This collator relies on details of the implementation of subword tokenization by
        :class:`~transformers.AlbertTokenizer`, specifically that start-of-word tokens are prefixed with `▁`.
        For tokenizers that do not adhere to this scheme, this collator will produce an output that is roughly
        equivalent to :class:`.DataCollatorForLanguageModeling`.
    """

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(
                examples,
                return_tensors="pt",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        else:
            batch = {"input_ids": collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)}

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)

        mask_labels = []
        for example in batch["input_ids"]:
            ref_tokens = self.tokenizer.convert_ids_to_tokens(tolist(example))
            mask_labels.append(self._whole_word_mask(ref_tokens))
        batch_mask = collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        batch["input_ids"], batch["labels"] = self.mask_tokens(
            batch["input_ids"], batch_mask, special_tokens_mask=special_tokens_mask
        )
        return batch

    def _whole_word_mask(self, input_tokens: List[str]):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """

        cand_indexes = []
        num_tokens_exc_pad = 0
        for i, token in enumerate(input_tokens):
            if token in (
                self.tokenizer.cls_token,
                self.tokenizer.sep_token,
                self.tokenizer.pad_token,
            ):
                continue
            num_tokens_exc_pad += 1
            if len(cand_indexes) >= 1 and not _is_start_piece_sp(token):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)

        mask_labels = torch.zeros((len(input_tokens),), dtype=torch.long)
        num_tokens_to_mask = min(num_tokens_exc_pad, max(1, int(round(num_tokens_exc_pad * self.mlm_probability))))
        covered_indexes = set()

        for index_set in cand_indexes:
            if len(covered_indexes) >= num_tokens_to_mask:
                break

            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(covered_indexes) + len(index_set) > num_tokens_to_mask:
                continue

            is_any_index_covered = any(index in covered_indexes for index in index_set)
            if is_any_index_covered:
                continue

            for index in index_set:
                covered_indexes.add(index)
                mask_labels[index] = 1

        return mask_labels

    def mask_tokens(
        self,
        inputs: torch.Tensor,
        mask_labels: torch.Tensor,
        special_tokens_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (WMM), we directly mask idxs according to it's ref.
        """
        assert self.mlm
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)

        probability_matrix = mask_labels

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
