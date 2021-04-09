import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm, trange
from sklearn import metrics

from utils import Config, set_seed
from helper import SemEvalDataLoader
from model import BertCls


class Task(object):
    def __init__(self, config_filepath) -> None:
        super().__init__()
        self.config = Config(config_filepath)
        set_seed(self.config.seed)

        self.loader = SemEvalDataLoader(self.config)
        self.model = BertCls(self.config)
        self.model = self.model.to(self.config.device)
        self.optimiser = optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.criterion = nn.BCELoss()

    def train(self):
        f1s = []
        for epoch_idx in trange(1, self.config.num_epoch + 1, desc="Epoch", ncols=80, ascii=True):
            self.model.train()
            loader = tqdm(self.loader.train_loader, desc="Train", ncols=80, ascii=True)
            for batch in loader:
                batch = {name: tensor.to(self.config.device) for name, tensor in batch.items()}
                out = self.model(**batch)
                loss = self.criterion(out.reshape(-1), batch["labels"].reshape(-1))
                loader.set_postfix({"loss": loss.item()})
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

            train_f1 = self.evaluate("train")
            dev_f1 = self.evaluate("dev")
            test_f1 = self.evaluate("test")
            f1s.append((epoch_idx, train_f1, dev_f1, test_f1))
            print(f"Epoch: {epoch_idx}, Train F1: {train_f1}, Dev F1: {dev_f1}, Test F1: {test_f1}")
        f1s.sort(key=lambda x: x[2], reverse=True)
        print(f"Best epoch: {f1s[0][0]}, Train F1: {f1s[0][1]}, Dev F1: {f1s[0][2]}, Test F1: {f1s[0][3]}")

    @torch.no_grad()
    def evaluate(self, dataset):
        self.model.eval()
        dataset2loader = {
            "train": self.loader.train_loader,
            "dev": self.loader.dev_loader,
            "test": self.loader.test_loader,
        }
        loader = tqdm(dataset2loader[dataset], desc=f"{dataset.title()} Eval", ncols=80, ascii=True)
        all_outs = []
        all_labels = []
        for batch in loader:
            batch = {name: tensor.to(self.config.device) for name, tensor in batch.items()}
            out = self.model(**batch)
            all_outs.extend(out.detach().cpu().ge(self.config.threshold).long().tolist())
            all_labels.extend(batch["labels"].detach().cpu().long().tolist())
        f1 = metrics.f1_score(np.array(all_labels), np.array(all_outs), average="macro")
        return f1


def main():
    task = Task("config.ini")
    task.train()


if __name__ == "__main__":
    main()
