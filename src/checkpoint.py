import hashlib

import torch

import config
from model import build_model
from my_tokenizer import JiebaTokenizer


def vocab_fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(model_path=None, device=None):
    """Load both new metadata checkpoints and the original plain state_dict."""
    model_path = model_path or config.MODELS_DIR / "best_model.pt"
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    legacy = "model_state" not in checkpoint
    state = checkpoint if legacy else checkpoint["model_state"]
    vocab_path = config.MODELS_DIR / "vocab.txt"
    if not legacy and checkpoint["vocab_sha256"] != vocab_fingerprint(vocab_path):
        raise ValueError("词表与模型权重不匹配；请使用训练该模型时的 vocab.txt")
    mode = "jieba" if legacy else checkpoint["tokenizer_mode"]
    tokenizer = JiebaTokenizer.from_vocab(vocab_path, mode=mode)
    if state["embedding.weight"].shape[0] != tokenizer.vocab_size:
        raise ValueError("词表大小与模型权重不匹配")
    if legacy:
        layers = len([key for key in state if key.startswith("lstm.weight_ih_l")
                      and not key.endswith("_reverse")])
        model_args = dict(embedding_dim=state["embedding.weight"].shape[1],
                          hidden_size=state["lstm.weight_hh_l0"].shape[1],
                          num_layers=layers, dropout=0.5, use_packing=False)
    else:
        model_args = checkpoint["model_config"]
    model = build_model(vocab_size=tokenizer.vocab_size, **model_args).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer, {
        "device": device,
        "seq_len": 128 if legacy else checkpoint["seq_len"],
        "threshold": 0.5 if legacy else checkpoint["threshold"],
        "legacy": legacy,
    }
