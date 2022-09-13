from dataclasses import dataclass, field
from typing import List, Optional

import torch
from transformers import TrainingArguments


@dataclass
class CollaborativeArguments:
    """Configuration for CollaborativeOptimizer and its internals"""

    target_batch_size: int = field(
        default=16384,
        metadata={"help": "Perform optimizer step after all peers collectively accumulate this many samples"},
    )
    matchmaking_time: float = field(
        default=60.0,
        metadata={"help": "Averaging group will wait for stragglers for at most this many seconds"},
    )
    next_chunk_timeout: float = field(
        default=60.0,
        metadata={"help": "Consider allreduce peer failed if it does not respond in this many seconds"},
    )
    averaging_timeout: float = field(
        default=600.0,
        metadata={"help": "Give up on averaging step after this many seconds"},
    )
    offload_optimizer: bool = field(default=True, metadata={"help": "Whether or not to offload optimizer into RAM"})
    delay_optimizer_step: bool = field(
        default=True,
        metadata={"help": "Whether or not to run optimizer step in background"},
    )
    delay_grad_averaging: bool = field(
        default=True,
        metadata={"help": "Whether or not to run gradient averaging in background"},
    )
    average_state_every: int = field(default=5, metadata={"help": "Average parameters every this many epochs"})
    reuse_grad_buffers: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to use model's .grad buffers for accumulating gradients across local steps. This "
                    "optimization reduces GPU memory consumption but may result in incorrect gradients when using some "
                    "advanced techniques (e.g. changing loss scaler to a custom one)."
        },
    )


@dataclass
class HFTrainerArguments(TrainingArguments):
    """Arguments for huggingface/transformers.Trainer"""

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1

    learning_rate: float = 2.5e-3  # based on https://arxiv.org/abs/1904.00962
    total_steps: int = 15625  # total number of collaborative optimizer updates, used for learning rate schedule
    warmup_steps: int = 3125  # based on https://arxiv.org/abs/1904.00962
    min_learning_rate: float = 1e-5  # learning rate after total_steps have passed
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # clipping performed by the optimizer; trainer is modified to disable builtin clipping
    clamp_value: float = 1e9  # no clipping by value
    min_8bit_size: int = 2 ** 20

    gradient_checkpointing: bool = False  # can be enabled to save memory at the cost of ~30% slower training
    fp16: bool = False  # can be enabled depending on the device

    max_sequence_length: int = 2048
    initial_sequence_length: Optional[int] = 256  # used only if warmup > 0, default = pad_to_multiple_of
    sequence_length_warmup_steps: int = 7_000
    pad_to_multiple_of: int = 128  # sequence length will be divisible by this value

    output_dir: str = "outputs"
    logging_steps: int = 100

    # params that should *not* be changed*
    do_train: bool = True
    do_eval: bool = False
    logging_first_step = True
    dataloader_num_workers: int = 0  # temporary fix for https://github.com/huggingface/datasets/issues/3148
    max_steps: int = 10 ** 30
    save_steps: int = 10 ** 30
    save_total_limit: int = 2
    ddp_find_unused_parameters: bool = False

    @property
    def batch_size_per_step(self):
        """Compute the number of training sequences contributed by each .step() from this peer"""
        total_batch_size_per_step = self.per_device_train_batch_size * self.gradient_accumulation_steps
        if torch.cuda.device_count() > 0:
            total_batch_size_per_step *= torch.cuda.device_count()
        return total_batch_size_per_step


@dataclass
class BasePeerArguments:
    """Base arguments that are used for both trainers and for auxiliary peers such as training monitor"""

    run_id: str = field(metadata={"help": "A unique experiment name, used as prefix for all DHT keys"})
    model_config_path: Optional[str] = field(default="./config.json", metadata={"help": "Path to the model config"})
    tokenizer_path: Optional[str] = field(default="./tokenizer", metadata={"help": "Path to the tokenizer"})
    cache_dir: Optional[str] = field(default="./cache", metadata={"help": "Path to the cache"})
    authorize: bool = field(default=False, metadata={"help": "Whether or not to use HF authorizer"})
    client_mode: bool = field(
        default=False,
        metadata={"help": "If True, runs training without incoming connections, in a firewall-compatible mode"},
    )
    bandwidth: Optional[float] = field(
        default=None,
        metadata={"help": "Min(upload & download speed) in megabits/s, used to assign averaging tasks between peers"},
    )
    min_vector_size: int = 4_000_000  # minimum slice of gradients assigned to one reducer, should be same across peers
    initial_peers: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Multiaddrs of the peers that will welcome you into the existing collaboration. "
                    "Example: /ip4/203.0.113.1/tcp/31337/p2p/XXXX /ip4/203.0.113.2/udp/7777/quic/p2p/YYYY"
        },
    )
    use_ipfs: bool = field(
        default=False,
        metadata={
            "help": "Use IPFS to find initial_peers. If enabled, you only need to provide /p2p/XXXX part of multiaddrs "
                    "for the initial_peers (no need to specify a particular IPv4/IPv6 address and port)"
        },
    )
    host_maddrs: List[str] = field(
        default_factory=lambda: ["/ip4/0.0.0.0/tcp/0"],
        metadata={
            "help": "Multiaddrs to listen for external connections from other p2p instances. "
                    "Defaults to all IPv4 interfaces with TCP protocol: /ip4/0.0.0.0/tcp/0"
        },
    )
    announce_maddrs: List[str] = field(
        default_factory=list,
        metadata={"help": "Visible multiaddrs the host announces for external connections from other p2p instances"},
    )
    identity_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a pre-generated private key file. If defined, makes the peer ID deterministic. "
                    "May be generated using ``./p2p-keygen`` from ``go-libp2p-daemon``."
        },
    )


@dataclass
class TrainingPeerArguments(BasePeerArguments):
    statistics_expiration: float = field(
        default=600,
        metadata={"help": "Statistics will be removed if not updated in this many seconds"},
    )
    backup_every_epochs: Optional[int] = field(
        default=None,
        metadata={
            "help": "Update training state backup on disk once in this many global steps "
                    "(default = do not update local state)"
        },
    )
    state_path: str = field(
        default="state.zip",
        metadata={"help": "Load this state upon init and when recovering from NaN parameters"},
    )


@dataclass
class AuxiliaryPeerArguments(BasePeerArguments):
    """
    Arguments for run_aux_peer.py that is responsible for connecting peers to one another, tracking
    learning curves, assisting in all-reduce and uploading checkpoints to the hub
    """

    refresh_period: float = field(
        default=10,
        metadata={"help": "Period (in seconds) for fetching the keys from DHT"},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Name of Weights & Biases project to report the training progress to"},
    )
    save_checkpoint_epoch_interval: int = field(
        default=5,
        metadata={"help": "Frequency (in steps) of fetching and saving state from peers"},
    )
    repo_url: Optional[str] = field(
        default=None,
        metadata={"help": "URL of Hugging Face Hub repository to upload the model and optimizer states"},
    )
    local_path: Optional[str] = field(
        default="Repo",
        metadata={"help": "Path to local repository to store the model and optimizer states"},
    )
    upload_interval: Optional[float] = field(
        default=None,
        metadata={"help": "Frequency (in seconds) of uploading the model to Hub"},
    )
    store_checkpoints: bool = field(default=True, metadata={"help": "If True, enables CheckpointHandler"})
    assist_in_averaging: bool = field(
        default=False,
        metadata={"help": "If True, this peer will facilitate averaging for other (training) peers"},
    )
    assist_refresh: float = field(
        default=1.0,
        metadata={"help": "Period (in seconds) for tryin to assist averaging"},
    )
