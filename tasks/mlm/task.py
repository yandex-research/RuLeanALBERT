import ctypes
import multiprocessing as mp
import os
from pathlib import Path

import hivemind
import torch.distributed
import transformers
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer
import torch

from arguments import BasePeerArguments, CollaborativeArguments, HFTrainerArguments
from src.models.albert import LeanAlbertConfig, LeanAlbertForPreTraining
from src.training.lamb_8bit import CPULAMB8Bit
from tasks.base import LRSchedulerBase, ParamGroups, TrainingTaskBase, register_task

from .data import make_training_dataset
from .whole_word_mask import DataCollatorForWholeWordMask

hivemind.use_hivemind_log_handler("in_root_logger")
logger = hivemind.get_logger()


@register_task("mlm")
class MLMTrainingTask(TrainingTaskBase):
    """A container that defines the training config, model, tokenizer, optimizer and other local training utilities"""

    _dht = _optimizer = _training_dataset = _authorizer = None

    def __init__(
        self, peer_args: BasePeerArguments, trainer_args: HFTrainerArguments, collab_args: CollaborativeArguments
    ):
        transformers.set_seed(trainer_args.seed)  # seed used for initialization

        self.config = LeanAlbertConfig.from_pretrained(peer_args.model_config_path)
        self.tokenizer = AutoTokenizer.from_pretrained(peer_args.tokenizer_path, cache_dir=peer_args.cache_dir)

        output_dir = Path(trainer_args.output_dir)
        logger.info(f'Checkpoint dir {output_dir}, contents {list(output_dir.glob("checkpoint*"))}')
        latest_checkpoint_dir = max(output_dir.glob("checkpoint*"), default=None, key=os.path.getctime)

        if latest_checkpoint_dir is None:
            logger.info(f"Creating model")
            model = LeanAlbertForPreTraining(self.config)
            model.resize_token_embeddings(len(self.tokenizer))
        else:
            logger.info(f"Loading model from {latest_checkpoint_dir}")
            model = LeanAlbertForPreTraining.from_pretrained(latest_checkpoint_dir)
        if trainer_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        super().__init__(model, peer_args, trainer_args, collab_args)
        self.current_sequence_length = mp.Value(ctypes.c_int64, self.trainer_args.max_sequence_length)
        self.update_sequence_length()  # updated by callback

    def _make_param_groups(self) -> ParamGroups:
        no_decay = ["bias", "norm.weight"]
        return [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(n.endswith(nd) for nd in no_decay)],
                "weight_decay": self.trainer_args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(n.endswith(nd) for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

    def _make_base_optimizer(self, param_groups: ParamGroups) -> torch.optim.Optimizer:
        return CPULAMB8Bit(
            param_groups,
            lr=self.trainer_args.learning_rate,
            betas=(self.trainer_args.adam_beta1, self.trainer_args.adam_beta2),
            min_8bit_size=self.trainer_args.min_8bit_size,
            max_grad_norm=self.trainer_args.max_grad_norm,
            clamp_value=self.trainer_args.clamp_value,
            eps=self.trainer_args.adam_epsilon,
            weight_decay=self.trainer_args.weight_decay,
            reuse_grad_buffers=True,
            bias_correction=True,
        )

    def _make_scheduler(self, optimizer: torch.optim.Optimizer) -> LRSchedulerBase:
        num_warmup_steps = self.trainer_args.warmup_steps
        num_training_steps = self.trainer_args.total_steps
        min_learning_rate = self.trainer_args.min_learning_rate



        def lr_lambda(current_step: int):
            if current_step < 50:
                return 0
            if current_step < num_warmup_steps:
                return float(current_step-50) / float(max(1, num_warmup_steps-50))
            decaying = float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.02, decaying)


        return LambdaLR(optimizer, lr_lambda)

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            decaying = float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            return max(min_learning_rate, decaying)
    @property
    def training_dataset(self):
        if self._training_dataset is None:
            self._training_dataset = make_training_dataset(
                self.tokenizer,
                shuffle_seed=hash(self.local_public_key) % 2 ** 31,
                max_sequence_length=self.current_sequence_length,  # this is a mp.Value that will be changed later
            )
        return self._training_dataset

    def on_step_end(self):
        return self.update_sequence_length()

    def update_sequence_length(self):
        """
        If ramp-up is enabled, start with smaller sequences of initial_sequence_length tokens, then increase linearly
        to the max_sequence_length over the period of first
        """
        current_epoch = self.get_current_epoch()
        if (
            self.trainer_args.sequence_length_warmup_steps == 0
            or current_epoch > self.trainer_args.sequence_length_warmup_steps
        ):
            current_sequence_length = self.trainer_args.max_sequence_length
        else:
            increment_size = self.trainer_args.pad_to_multiple_of
            max_sequence_length = self.trainer_args.max_sequence_length
            initial_sequence_length = self.trainer_args.initial_sequence_length or increment_size
            sequence_length_warmup_steps = self.trainer_args.sequence_length_warmup_steps
            assert sequence_length_warmup_steps > 0 and max_sequence_length >= initial_sequence_length
            length_range = max_sequence_length - initial_sequence_length
            warmup_relative = min(1, current_epoch / sequence_length_warmup_steps)
            current_sequence_length = initial_sequence_length + warmup_relative * length_range
            current_sequence_length = (current_sequence_length // increment_size) * increment_size
            current_sequence_length = min(max(current_sequence_length, initial_sequence_length), max_sequence_length)

        current_sequence_length = int(current_sequence_length)
        if current_sequence_length != self.current_sequence_length.value:
            logger.info(f"Transitioning to sequence length {current_sequence_length}")
            self.current_sequence_length.value = current_sequence_length
            # note: it may take time for new sequence length to take effect due to buffering

    @property
    def data_collator(self):
        return DataCollatorForWholeWordMask(
            tokenizer=self.tokenizer, pad_to_multiple_of=self.trainer_args.pad_to_multiple_of
        )
