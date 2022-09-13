from dataclasses import asdict
from typing import Optional, Type

import hivemind
import torch
import torch.nn as nn
from hivemind import Float16Compression, SizeAdaptiveCompression, Uniform8BitQuantization
from hivemind.utils.logging import get_logger
from torch.distributed.distributed_c10d import _get_default_store
from transformers.data.data_collator import DataCollatorMixin

from arguments import BasePeerArguments, CollaborativeArguments, HFTrainerArguments
from src.huggingface_auth import authorize_with_huggingface
from src.training.sync import AUTHORITATIVE_RANK, is_main_process

try:
    from hivemind.optim.experimental.state_averager import LRSchedulerBase, ParamGroups
except ImportError:
    from hivemind.optim.state_averager import LRSchedulerBase, ParamGroups

from src import utils

logger = get_logger(__name__)
TASKS = {}


def register_task(name: str):
    def _register(cls: Type[TrainingTaskBase]):
        if name in TASKS:
            logger.warning(f"Registering task {name} a second time, previous entry will be overwritten")
        TASKS[name] = cls
        return cls

    return _register


class TrainingTaskBase:
    """A container that defines the training config, model, tokenizer, optimizer and other local training utilities"""

    _dht = _optimizer = _authorizer = None  # for caching

    def __init__(
        self,
        model: nn.Module,
        peer_args: BasePeerArguments,
        trainer_args: HFTrainerArguments,
        collab_args: CollaborativeArguments,
    ):
        self.model, self.peer_args, self.trainer_args, self.collab_args = model, peer_args, trainer_args, collab_args
        self.validators, self.local_public_key = utils.make_validators(self.peer_args.run_id)

        if self.authorizer:
            self.trainer_args.run_name = self.authorizer.username  # For wandb

    @property
    def authorizer(self):
        if self._authorizer is None and self.peer_args.authorize:
            self._authorizer = authorize_with_huggingface()
        return self._authorizer

    @property
    def dht(self):
        if self._dht is None:
            assert is_main_process()
            self._dht = hivemind.DHT(
                start=True,
                initial_peers=self.peer_args.initial_peers,
                client_mode=self.peer_args.client_mode,
                host_maddrs=self.peer_args.host_maddrs,
                announce_maddrs=self.peer_args.announce_maddrs,
                use_ipfs=self.peer_args.use_ipfs,
                record_validators=self.validators,
                identity_path=self.peer_args.identity_path,
                authorizer=self.authorizer,
            )
            if self.peer_args.client_mode:
                logger.info(f"Created client mode peer with peer_id={self._dht.peer_id}")
            else:
                utils.log_visible_maddrs(self._dht.get_visible_maddrs(), only_p2p=self.peer_args.use_ipfs)
        return self._dht

    @property
    def optimizer(self) -> hivemind.Optimizer:
        if self._optimizer is None:
            assert is_main_process()
            averaging_compression = SizeAdaptiveCompression(
                threshold=2 ** 16 + 1, less=Float16Compression(), greater_equal=Uniform8BitQuantization()
            )

            self._optimizer = hivemind.Optimizer(
                dht=self.dht,
                params=self._make_param_groups(),
                run_id=self.peer_args.run_id,
                optimizer=self._make_base_optimizer,
                scheduler=self._make_scheduler,
                grad_compression=averaging_compression,
                state_averaging_compression=averaging_compression,
                batch_size_per_step=self.trainer_args.batch_size_per_step,
                client_mode=self.peer_args.client_mode,
                verbose=True,
                averager_opts=dict(min_vector_size=self.peer_args.min_vector_size, bandwidth=self.peer_args.bandwidth),
                **asdict(self.collab_args),
            )
        return self._optimizer

    def _make_param_groups(self) -> ParamGroups:
        """Return optimizer param groups: either list of parameters or a list of dictionaries in torch.optim format"""
        raise NotImplementedError()

    def _make_base_optimizer(self, param_groups: ParamGroups) -> torch.optim.Optimizer:
        """Return PyTorch optimizer to be wrapped with hivemind.Optimizer. Use only the specified param groups."""
        raise NotImplementedError()

    def _make_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[LRSchedulerBase]:
        """Return PyTorch scheduler that will be ran on each synchronization epoch (each target_batch_size samples)"""
        return None  # default = no scheduler

    @property
    def training_dataset(self):
        raise NotImplementedError()

    @property
    def data_collator(self) -> DataCollatorMixin:
        raise NotImplementedError()

    def on_step_end(self):
        """This will be called after each local step (in callback.py)"""
        pass

    def get_current_epoch(self):
        """
        Return current epoch in a ddp-friendly way. When implementing your own task, please use this instead of
        directly accessing self.optimizer.global_epoch because non-main DDP workers will not run hivemind.Optimizer
        """
        if not torch.distributed.is_initialized():
            return self.optimizer.tracker.global_epoch
        else:
            store = _get_default_store()
            if torch.distributed.get_rank() == AUTHORITATIVE_RANK:
                current_epoch = self.optimizer.tracker.global_epoch
                store.set("_hivemind_current_epoch", str(current_epoch))
                torch.distributed.barrier()
                return current_epoch
            else:
                torch.distributed.barrier()
                return int(store.get("_hivemind_current_epoch"))
