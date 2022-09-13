from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, Union, List, Dict, Any

import torch
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy


@dataclass
class FieldDataCollatorWithPadding:
    """
    A general-purpose data collator that can handle batches with arbitrary fields.
    Supports only PyTorch tensors as outputs.
    """
    tokenizer: PreTrainedTokenizerBase
    # (field name, pad token index, index of the sequence axis or None)
    fields_to_pad: Iterable[Tuple[str, int, Optional[int]]] = ()
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        field_values = defaultdict(list)
        for field_name, field_pad_idx, field_seq_idx in self.fields_to_pad:
            for example in features:
                if field_name in example:
                    field_values[field_name].append(example.pop(field_name))

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if "label" in batch:
            batch["labels"] = batch.pop("label")
        if "label_ids" in batch:
            batch["labels"] = batch.pop("label_ids")

        sequence_length = batch["input_ids"].size(1)
        batch_size = batch["input_ids"].size(0)

        for field_name, field_pad_idx, field_seq_idx in self.fields_to_pad:

            if field_values[field_name]:
                field_value_arrays = [torch.as_tensor(values) for values in field_values[field_name]]
                assert len(set(arr.ndim for arr in field_value_arrays)) == 1

                max_dim_lengths = [
                    max(arr.size(dim) for arr in field_value_arrays)
                    for dim in range(field_value_arrays[0].ndim)
                ]

                if field_seq_idx is not None:
                    max_dim_lengths[field_seq_idx] = sequence_length

                padded_tensor = torch.full([batch_size] + max_dim_lengths, field_pad_idx, dtype=torch.long)

                for i, ar in enumerate(field_value_arrays):
                    paste_inds = [i]

                    for dim in range(field_value_arrays[0].ndim):
                        if dim == field_seq_idx and self.tokenizer.padding_side == "left":
                            inds = slice(-ar.size(dim), -1)
                        else:
                            inds = slice(ar.size(dim))
                        paste_inds.append(inds)

                    padded_tensor[paste_inds] = ar

                batch[field_name] = padded_tensor

        return batch
