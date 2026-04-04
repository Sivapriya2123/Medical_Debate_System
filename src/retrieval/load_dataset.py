from datasets import load_dataset


def load_pubmedqa(split: str = "train", limit: int = 500):
    """
    Load PubMedQA labeled subset.
    """
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split=split, cache_dir="./data")

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    return dataset
