import torch.nn as nn
import torch.optim as optim
import torch
import time
import yaml
import os

from tqdm import tqdm
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import List

from .base_runs import AbstractRunStep
from ..utils.util import prepare_device, unique_folder_path
from ..data.data_loader import get_dataloader
from ..metrics.classification import metrics


class Trainer(ABC):
    @abstractmethod
    def train_fn(self, **kwargs):
        pass

    @abstractmethod
    def save_model(self, **kwargs):
        pass


class FakeEvidenceTrainer(Trainer):
    # TODO Meta-data file

    def __init__(self, model:nn.Module, dataset:List[Dataset], config):
        """
        Trainer for FakeEvidence model.
        :param model: Instance of FakeEvidence model.
        :param dataset: A list contains train and val dataset.
        :param config: A dict contains configuration parameters.
        The accepted keys in "config" dict include:
            - use_cuda (int): how many GPUs to use. If "use_cuda" is 0, use CPU. If "use_cuda" is 1,
            use 1 GPU. If "use_cuda" >= 2, use multiple GPUs.
        """
        self._model = model

        cuda_list = prepare_device(config.get("use_cuda"))
        if cuda_list:
            if len(cuda_list) > 1:
                raise ValueError("Multiple GPUs are not supported yet.")
            else:
                self._model.to(f"cuda:{cuda_list[0]}")

        self._config = config
        self._dataset = dataset
        self._save_model_path = unique_folder_path(self._config.get("save_model_path"), "FakeEvidenceModel")
        if not os.path.exists(self._save_model_path):
            os.makedirs(self._save_model_path)


    def train_fn(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        st_tm = time.time()

        pred, label = [], []

        criterion = nn.BCELoss()
        optimizer = optim.AdamW(
            params=self._model.parameters(),
            lr=5e-5, weight_decay=1e-5
        )

        train_dataset = self._dataset[0]
        train_loader = get_dataloader(train_dataset, 20, shuffle=True)

        val_dataset = self._dataset[1]
        val_loader = get_dataloader(val_dataset, 20, shuffle=False)

        ed_tm = time.time()

        print("Time cost in model and data loading: {}s".format(ed_tm - st_tm))

        for epoch in range(75):
            print("Epoch {}/{}".format(epoch + 1, 75))
            self._model.train()

            for step_n, batch in enumerate(tqdm(train_loader)):
                device = torch.device("cuda:0")
                batch = {key: value.to(device) for key, value in batch.items()}
                batch['is_training'] = True
                optimizer.zero_grad()
                outputs = self._model(**batch)
                targets = batch["label"].float()
                loss = criterion(outputs, targets)

                label.extend(targets.detach().cpu().numpy().tolist())
                pred.extend(outputs.detach().cpu().numpy().tolist())

                loss.backward()
                optimizer.step()

            train_results = metrics(label, pred)

            for key, value in train_results.items():
                train_results[key] = float(value)

            print('     Valid process:')
            valid_results = self.evaluate(val_loader, criterion)

            for key, value in valid_results.items():
                valid_results[key] = float(value)

            print("results: ", valid_results['acc'], valid_results["f1"])

            self.save_model(train_metric=train_results, valid_metric=valid_results, trained_epoch=epoch + 1)


    def evaluate(self, val_loader, criterion):
        pred, label = [], []
        self._model.eval()
        for step_n, batch in enumerate(tqdm(val_loader)):
            with torch.no_grad():
                device = torch.device("cuda:0")

                batch = {key: value.to(device) for key, value in batch.items()}
                batch_label = batch["label"].float()
                batch['is_training'] = False
                results = self._model(**batch)

                loss = criterion(results, batch_label.float())
                label.extend(batch_label.cpu().numpy().tolist())
                pred.extend(results.cpu().numpy().tolist())

        return metrics(label, pred)

    def save_model(self, **kwargs):
        path = unique_folder_path(self._save_model_path, "train")

        os.makedirs(path)

        train_metric = kwargs.get("train_metric", -1)
        val_metric = kwargs.get("valid_metric", -1)

        trained_epoch = kwargs.get("trained_epoch", -1)

        meta_data = {
            "train": train_metric,
            "valid": val_metric,
            "trained_epoch": trained_epoch,
            "time": time.time()
        }

        with open(path / "meta_data.yaml", "w") as f:
            yaml.dump(meta_data, f, default_flow_style=False)

        torch.save(self._model.state_dict(), path / "model.pt")
        print("Model saved!")

class ModelTrainStep(AbstractRunStep):
    pass