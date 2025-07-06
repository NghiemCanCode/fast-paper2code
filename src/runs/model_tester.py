import torch

from tqdm import tqdm
from abc import abstractmethod, ABC

from ..utils.util import prepare_device
from ..data.data_loader import get_dataloader
from ..metrics.classification import metrics

class ModelTester(ABC):
    @abstractmethod
    def test_fn(self, **kwargs):
        pass



class RolePlayTester(ModelTester):
    def __init__(self, model, config, model_params_path, dataset):
        self._model = model
        para = torch.load(model_params_path)
        self._model.load_state_dict(para)
        self._dataset = dataset

        cuda_list = prepare_device(config.get("use_cuda"))
        if cuda_list:
            if len(cuda_list) > 1:
                raise ValueError("Multiple GPUs are not supported yet.")
            else:
                self._model.to(f"cuda:{cuda_list[0]}")

    def test_fn(self, **kwargs):
        pred, label = [], []
        self._model.eval()

        data_loader = get_dataloader(self._dataset, 16, shuffle=False)
        for step_n, batch in enumerate(tqdm(data_loader)):
            with torch.no_grad():
                device = torch.device("cuda:0")
                batch = {key: value.to(device) for key, value in batch.items()}

                batch['is_training'] = False

                batch_label = batch["label"].float()
                results = self._model(**batch)

                label.extend(batch_label.cpu().numpy().tolist())
                pred.extend(results.cpu().numpy().tolist())

        return metrics(label, pred)