#!/usr/bin/env python
import threading
import time

import scipy.stats  # compatibility for internal testing environment
import torch
import transformers
import wandb
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from huggingface_hub import Repository
from transformers import HfArgumentParser

from arguments import AuxiliaryPeerArguments, CollaborativeArguments, HFTrainerArguments
from src import utils
from tasks.mlm.task import MLMTrainingTask

transformers.utils.logging.set_verbosity_warning()
use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)
torch.set_num_threads(1)  # avoid quadratic number of threads


class CheckpointHandler:
    def __init__(self, task: MLMTrainingTask, peer_args: AuxiliaryPeerArguments):
        self.task, self.peer_args = task, peer_args
        self.save_checkpoint_epoch_interval = peer_args.save_checkpoint_epoch_interval
        self.prefix = peer_args.run_id
        self.local_path = peer_args.local_path
        self.upload_interval = peer_args.upload_interval
        if self.upload_interval is not None:
            assert task.authorizer is not None, "Model uploading needs Hugging Face auth to be enabled"
            self.repo = Repository(
                local_dir=self.local_path,
                clone_from=peer_args.repo_url,
                use_auth_token=task.authorizer.hf_user_access_token,
            )
            self.last_upload_time = None
        self.previous_epoch = -1

    def should_save_state(self, current_epoch: int):
        if self.save_checkpoint_epoch_interval is None:
            return False
        elif current_epoch - self.previous_epoch >= self.save_checkpoint_epoch_interval:
            return True
        else:
            return False

    def save_state(self, current_epoch: int):
        logger.info("Saving state from peers")
        self.task.optimizer.load_state_from_peers()
        self.previous_epoch = current_epoch

    def is_time_to_upload(self):
        if self.upload_interval is None:
            return False
        elif self.last_upload_time is None or time.time() - self.last_upload_time >= self.upload_interval:
            return True
        else:
            return False

    def upload_checkpoint(self, current_loss: float):
        self.last_upload_time = time.time()

        logger.info("Saving model")
        torch.save(self.task.model.state_dict(), f"{self.local_path}/model_state.pt")
        logger.info("Saving optimizer")
        torch.save(self.task.optimizer.state_dict(), f"{self.local_path}/optimizer_state.pt")
        self.previous_timestamp = time.time()
        logger.info("Started uploading to Model Hub")
        try:
            # We start by pulling the remote changes (for example a change in the readme file)
            self.repo.git_pull()

            # Then we add / commmit and push the changes
            self.repo.push_to_hub(commit_message=f"Epoch {self.task.optimizer.local_epoch}, loss {current_loss:.3f}")
            logger.info("Finished uploading to Model Hub")
        except Exception:
            logger.exception("Uploading the checkpoint to HF Model Hub failed:")
            logger.warning("Ensure that your access token is valid and has WRITE permissions")


def assist_averaging_in_background(
    lock: threading.Lock, task: MLMTrainingTask, peer_args: AuxiliaryPeerArguments, finished: threading.Event
):
    while not finished.is_set():
        try:
            time.sleep(peer_args.assist_refresh)
            with lock:
                task.optimizer.step()
        except Exception as e:
            logger.exception(e, exc_info=True)


if __name__ == "__main__":
    parser = HfArgumentParser((AuxiliaryPeerArguments, HFTrainerArguments, CollaborativeArguments))
    peer_args, trainer_args, collab_args = parser.parse_args_into_dataclasses()
    finished, lock = threading.Event(), threading.Lock()

    task = MLMTrainingTask(peer_args, trainer_args, collab_args)
    dht, optimizer = task.dht, task.optimizer

    if peer_args.wandb_project is not None:
        wandb.init(project=peer_args.wandb_project)

    current_epoch = 0
    if peer_args.store_checkpoints:
        checkpoint_handler = CheckpointHandler(task, peer_args)

    if peer_args.assist_in_averaging:
        assert not peer_args.client_mode, "client-mode peers cannot assist in averaging"
        averaging_thread = threading.Thread(
            name="AveragingAuxThread",
            target=assist_averaging_in_background,
            args=[lock, task, peer_args, finished],
            daemon=True,
        )
        averaging_thread.start()

    try:
        while True:
            metrics_entry = dht.get(peer_args.run_id + "_metrics", latest=True)
            if metrics_entry is not None and len(metrics_entry.value) > 0:
                metrics_dict = metrics_entry.value
                metrics = [utils.LocalMetrics.parse_obj(metrics_dict[peer].value) for peer in metrics_dict]
                latest_epoch = max(item.epoch for item in metrics)

                if latest_epoch != current_epoch:
                    logger.debug(f"Got metrics from {len(metrics)} peers")

                    for i, metrics_for_peer in enumerate(metrics):
                        logger.debug(f"{i} peer {metrics_for_peer}")

                    current_epoch = latest_epoch
                    alive_peers = 0
                    sum_loss = 0
                    num_samples = 0
                    sum_perf = 0
                    sum_mini_steps = 0

                    for item in metrics:
                        sum_loss += item.loss
                        alive_peers += 1
                        sum_perf += item.samples_per_second
                        num_samples += item.samples_accumulated
                        sum_mini_steps += item.mini_steps
                    current_loss = sum_loss / sum_mini_steps
                    logger.info(f"Epoch #{current_epoch}\tloss = {current_loss:.5f}")

                    if peer_args.wandb_project is not None:
                        wandb.log(
                            {
                                "loss": current_loss,
                                "alive peers": alive_peers,
                                "samples": num_samples,
                                "performance": sum_perf,
                                "optimizer_step": latest_epoch,
                            },
                            step=latest_epoch,
                        )

                    if peer_args.store_checkpoints:
                        if checkpoint_handler.should_save_state(current_epoch):
                            with lock:
                                checkpoint_handler.save_state(current_epoch)
                                if checkpoint_handler.is_time_to_upload():
                                    checkpoint_handler.upload_checkpoint(current_loss)
            logger.debug("Peer is still alive...")
            time.sleep(peer_args.refresh_period)
    finally:
        finished.set()
