import json

import pandas as pd
from sklearn.model_selection import train_test_split

import config
from my_tokenizer import JiebaTokenizer


def process():
    source = config.RAW_DATA_DIR / "online_shopping_10_cats.csv"
    if not source.exists():
        raise FileNotFoundError(f"缺少原始数据：{source}")
    frame = pd.read_csv(source, usecols=["label", "review"], encoding="utf-8")
    frame = frame.dropna(subset=["label", "review"])
    frame = frame[frame["label"].isin([0, 1])].copy()
    frame["review"] = frame["review"].astype(str).str.strip()
    frame = frame[frame["review"].ne("")].drop_duplicates(subset=["review"])
    if config.SAMPLE_FRACTION < 1:
        frame, _ = train_test_split(
            frame, train_size=config.SAMPLE_FRACTION, stratify=frame["label"],
            random_state=config.SEED,
        )
    train, test = train_test_split(
        frame, test_size=config.TEST_FRACTION, stratify=frame["label"],
        random_state=config.SEED,
    )
    tokenizer = JiebaTokenizer.build_vocab(
        train["review"], config.MODELS_DIR / "vocab.txt",
        mode=config.TOKENIZER_MODE, min_freq=config.MIN_TOKEN_FREQ,
        max_size=config.MAX_VOCAB_SIZE,
    )
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("test", test)):
        path = config.PROCESSED_DATA_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as target:
            for row in split.itertuples(index=False):
                target.write(json.dumps({"review": tokenizer.encode(row.review, config.SEQ_LEN),
                                         "label": int(row.label)}, ensure_ascii=False) + "\n")
        print(f"{name}: {len(split)} 条 -> {path}")
    print(f"词表大小: {tokenizer.vocab_size}; 切词形式: {tokenizer.mode}")
    print("数据已更新，请重新训练；旧模型权重与新词表不兼容。")


if __name__ == "__main__":
    process()
