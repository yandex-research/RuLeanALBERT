from transformers import AlbertConfig


class LeanAlbertConfig(AlbertConfig):
    rotary_embedding_base: int = 10_000
    hidden_act_gated: bool = False

    def __hash__(self):
        return hash("\t".join(f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")))
