#!/usr/bin/env python

import os
from pathlib import Path

import scipy.stats  # compatibility for internal testing environment
import torch.distributed
import transformers
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from transformers import HfArgumentParser

from arguments import CollaborativeArguments, HFTrainerArguments, TrainingPeerArguments
from src import utils
from src.training.callback import CollaborativeCallback
from src.training.hf_trainer import CollaborativeHFTrainer, NOPtimizer
from src.training.sync import SynchronizationCallback, is_main_process
from tasks.mlm.task import MLMTrainingTask

use_hivemind_log_handler("in_root_logger")
logger = get_logger()
torch.set_num_threads(1)  # avoid quadratic number of threads


def main():
    parser = HfArgumentParser((TrainingPeerArguments, HFTrainerArguments, CollaborativeArguments))
    training_peer_args, trainer_args, collab_args = parser.parse_args_into_dataclasses()
    if torch.distributed.is_initialized():
        assert not collab_args.reuse_grad_buffers, "Reuse_grad_buffers are not supported in distributed mode"

    logger.info(f"Trying {len(training_peer_args.initial_peers)} initial peers: {training_peer_args.initial_peers}")
    if len(training_peer_args.initial_peers) == 0:
        logger.warning("Please specify initial peers or let others join your peer")

    utils.setup_logging(trainer_args)
    task = MLMTrainingTask(training_peer_args, trainer_args, collab_args)
    model = task.model.to(trainer_args.device)
    for param in model.parameters():
        if param.grad is None:
            param.grad = torch.zeros_like(param)

    callbacks = [(CollaborativeCallback if is_main_process() else SynchronizationCallback)(task, training_peer_args)]
    assert trainer_args.do_train and not trainer_args.do_eval

    # Note: the code below creates the trainer with dummy scheduler and removes some callbacks.
    # This is done because collaborative training has its own callbacks that take other peers into account.
    trainer = CollaborativeHFTrainer(
        model=model,
        args=trainer_args,
        tokenizer=task.tokenizer,
        data_collator=task.data_collator,
        data_seed=hash(task.local_public_key),
        train_dataset=task.training_dataset,
        reuse_grad_buffers=collab_args.reuse_grad_buffers,
        eval_dataset=None,
        optimizer=task.optimizer if is_main_process() else NOPtimizer(task._make_param_groups()),
        callbacks=callbacks,
    )
    trainer.remove_callback(transformers.trainer_callback.PrinterCallback)
    trainer.remove_callback(transformers.trainer_callback.ProgressCallback)

    latest_checkpoint_dir = max(Path(trainer_args.output_dir).glob("checkpoint*"), key=os.path.getctime, default=None)
    trainer.train(model_path=latest_checkpoint_dir)


if __name__ == "__main__":
    main()
