from datasets import load_dataset


def load_pubmedqa(split: str = "train", limit: int = 500):
    """
    Load PubMedQA labeled subset.
    """
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split=split)

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    return dataset
