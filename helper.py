import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast


def load_data(filepath):
    data = []
    with open(filepath, "rt", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin):
            if line_idx == 0:
                continue
            line = line.strip().split('\t')
            data.append({
                "id": line[0],
                "tweet": line[1],
                "label": eval(f'[{",".join(line[2:])}]')
            })
    return data


class SemEvalDataset(Dataset):
    def __init__(self, filepath) -> None:
        super().__init__()
        self.data = load_data(filepath)
    
    def __getitem__(self, index: int):
        return self.data[index]
    
    def __len__(self) -> int:
        return len(self.data)


class SemEvalDataLoader(object):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.train_dataset = SemEvalDataset(self.config.train_path)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.dev_dataset = SemEvalDataset(self.config.dev_path)
        self.dev_loader = DataLoader(self.dev_dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.test_dataset = SemEvalDataset(self.config.test_path)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=self.collate_fn)

        self.tokenizer = BertTokenizerFast.from_pretrained(self.config.bert_path)

    def collate_fn(self, batch):
        tweets = [x["tweet"] for x in batch]
        labels = [x["label"] for x in batch]

        tweets = self.tokenizer.batch_encode_plus(
            tweets,
            add_special_tokens=True,
            padding=True,
            max_length=self.config.max_seq_len,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True)
        labels = torch.tensor(labels, dtype=torch.float)

        return {
            "input_ids": tweets["input_ids"],
            "attention_mask": tweets["attention_mask"],
            "token_type_ids": tweets["token_type_ids"],
            "labels": labels
        }


if __name__ == "__main__":
    data = load_data("data/dev.txt")
    print(data[:5])
