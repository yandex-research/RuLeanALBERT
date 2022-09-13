from typing import Dict, List, Tuple

import transformers.utils.logging
from hivemind import choose_ip_address
from hivemind.dht.crypto import RSASignatureValidator
from hivemind.dht.schema import BytesWithPublicKey, SchemaValidator
from hivemind.dht.validation import RecordValidatorBase
from hivemind.utils.logging import get_logger
from multiaddr import Multiaddr
from pydantic import BaseModel, StrictFloat, confloat, conint
from transformers.trainer_utils import is_main_process

logger = get_logger(__name__)


class LocalMetrics(BaseModel):
    epoch: conint(ge=0, strict=True)
    samples_per_second: confloat(ge=0.0, strict=True)
    samples_accumulated: conint(ge=0, strict=True)
    loss: StrictFloat
    mini_steps: conint(ge=0, strict=True)  # queries


class MetricSchema(BaseModel):
    metrics: Dict[BytesWithPublicKey, LocalMetrics]


def make_validators(run_id: str) -> Tuple[List[RecordValidatorBase], bytes]:
    signature_validator = RSASignatureValidator()
    validators = [SchemaValidator(MetricSchema, prefix=run_id), signature_validator]
    return validators, signature_validator.local_public_key


class TextStyle:
    BOLD = "\033[1m"
    BLUE = "\033[34m"
    RESET = "\033[0m"


def log_visible_maddrs(visible_maddrs: List[Multiaddr], only_p2p: bool) -> None:
    if only_p2p:
        unique_addrs = {addr["p2p"] for addr in visible_maddrs}
        initial_peers_str = " ".join(f"/p2p/{addr}" for addr in unique_addrs)
    else:
        available_ips = [Multiaddr(addr) for addr in visible_maddrs if "ip4" in addr or "ip6" in addr]
        if available_ips:
            preferred_ip = choose_ip_address(available_ips)
            selected_maddrs = [addr for addr in visible_maddrs if preferred_ip in str(addr)]
        else:
            selected_maddrs = visible_maddrs
        initial_peers_str = " ".join(str(addr) for addr in selected_maddrs)

    logger.info(
        f"Running a DHT peer. To connect other peers to this one over the Internet, use "
        f"{TextStyle.BOLD}{TextStyle.BLUE}--initial_peers {initial_peers_str}{TextStyle.RESET}"
    )
    logger.info(f"Full list of visible multiaddresses: {' '.join(str(addr) for addr in visible_maddrs)}")


def setup_logging(training_args):
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)
