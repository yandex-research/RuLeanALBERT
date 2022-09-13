import time
from typing import Sequence

import torch
import transformers
from hivemind.utils import get_logger
from torch.distributed.distributed_c10d import _get_default_group, _get_default_store
from transformers import TrainerControl, TrainerState, TrainingArguments

import arguments
import tasks

AUTHORITATIVE_RANK = 0
BROADCAST_BUFFER_SIZE: int = 250 * 1024 * 1024
logger = get_logger(__name__)


def is_main_process() -> bool:
    """Whether this is the main process on **this peer's** distributed run. Non-distributed process is always main."""
    return (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == AUTHORITATIVE_RANK


class SynchronizationCallback(transformers.TrainerCallback):
    """Minimalistic callback for non-master DDP workers"""

    def __init__(self, task: "tasks.TrainingTaskBase", args: "arguments.TrainingPeerArguments"):
        self.task = task
        self.is_master = is_main_process()
        self._checksum_counter = 0
        self._state_tensors = None
        self._prev_version = self._prev_epoch = -1

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if torch.distributed.is_initialized():
            self._maybe_sync_model_state()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs,
    ):
        control.should_log = True
        model = self.task.model
        if torch.distributed.is_initialized():
            self._maybe_sync_model_state()

            self._checksum_counter += 1
            if self._checksum_counter % 100 == 0:
                rank = torch.distributed.get_rank()
                print(end=f"CHECKSUM({rank})={float(sum(p.sum().item() for p in model.parameters()))}\n")
        self.task.on_step_end()

    @property
    def state_tensors(self) -> Sequence[torch.Tensor]:
        if self._state_tensors is None:
            self._state_tensors = list(self.task.model.state_dict().values())
        return self._state_tensors

    def _compute_state_version(self) -> int:
        """return a non-decreasing integer that goes up whenever model params and/or buffers were updated"""
        assert self.is_master
        return sum(state["step"] for state in self.task.optimizer.opt.state.values())

    def _should_broadcast_state(self):
        store = _get_default_store()
        if self.is_master:
            current_version = self._compute_state_version()
            if current_version == self._prev_version and self.task.optimizer.local_epoch > self._prev_epoch + 1:
                logger.warning("Model state version has not changed during a full epoch; "
                               "broadcasting parameters between torch.distributed synchronization may be broken")

            if current_version != self._prev_version or self.task.optimizer.local_epoch > self._prev_epoch + 1:
                should_broadcast = True
            else:
                should_broadcast = False

            store.set(f"_hivemind_should_broadcast_state", str(int(should_broadcast)))
            torch.distributed.barrier()
            return should_broadcast
        else:
            torch.distributed.barrier()
            raw_should_broadcast = store.get(f"_hivemind_should_broadcast_state")
            return bool(int(raw_should_broadcast))

    def _maybe_sync_model_state(self):
        """Synchronize model params and buffers from master"""
        if self.state_tensors and self._should_broadcast_state():
            t_start = time.perf_counter()
            with torch.no_grad():
                torch.distributed._broadcast_coalesced(
                    _get_default_group(), self.state_tensors, BROADCAST_BUFFER_SIZE, AUTHORITATIVE_RANK
                )
            if self.is_master:
                self._prev_version = self._compute_state_version()
                self._prev_epoch = self.task.optimizer.local_epoch
                logger.info(f"Broadcasting master params took {time.perf_counter() - t_start} seconds")
        else:
            logger.debug("Not broadcasting")
