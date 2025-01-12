import os
import json
from collections import OrderedDict
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
import torchmetrics.functional as metrics_F

from model import PromoterNet
from dataset import (
    PromoterDataset,
    batch_collate_fn,
)

DATASETS = {
    "all_dataset": PromoterDataset,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseInferer:
    """Inference loop for a trained model. Run the testing scheme."""

    def __init__(
        self,
        dataset: str,
        model: Optional[Module] = None,
        model_path: Optional[str] = None,
        model_params: Optional[Dict[str, Any]] = None,
        out_path: Optional[str] = None,
        metric_name: str = "r2",
    ) -> None:
        """
        Initialize the inference (testing) setup.

        Parameters
        ----------
        dataset: string
            Which dataset should be used for inference.
        model: torch Module, optional
            The model needs to be tested or inferred. If None, model_path
            and model_params should be specified to load a model.
        model_path: string, optional
            Path to the expected model.
        model_params: dictionary, optional
            Parameters that was specified before the training of the model.
        out_path: string, optional
            If you want to save the predictions, specify a path.
        metric_name: string
            Metric to evaluate the test performance of the model.
        """
        self.dataset = dataset
        self.model_path = model_path
        self.out_path = out_path
        self.model_params = model_params
        self.model: Module
        if model is None:
            assert (
                model_params is not None
            ), "Specify the model or: specify model_path and model_params"
            assert (
                model_path is not None
            ), "Specify the model or: specify model_path and model_params"
            self.model = self.load_model_from_file(model_path, model_params)
        else:
            self.model = model

        self.dataset = dataset

        if metric_name == "r2":
            self.metric = metrics_F.r2_score
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def run(self, test_split_only: bool = True) -> List[float]:
        """
        Run inference loop whether for testing purposes or in-production.

        Parameters
        ----------
        test_split_only: bool
            Whether to use all dataset samples or just the testing split. This can be handy
            when testing a pretrained model on your private dataset. Set false if you want to
            use your model in production.

        Returns
        -------
        test_losses: list of floats
            Test loss for each sample. Or any metric you will define. Calculates only if test_split_only is True.
        """
        self.model.eval()
        test_losses = []

        if test_split_only:
            mode = "test"
        else:
            mode = "inference"

        test_dataset = DATASETS[self.dataset](
            path_to_data="./data/test_sequences.txt", mode=mode, n_folds=1
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            collate_fn=batch_collate_fn,
        )
        preds = []
        for idx, (input_data, target_label) in enumerate(test_dataloader):
            prediction = self.model(input_data)
            if self.out_path is not None:
                torch.save(prediction, os.path.join(self.out_path, f"sample_{idx}.pt"))
            if test_split_only:
                test_loss = self.metric(prediction, target_label)
                test_losses.append(test_loss.item())
            preds.append(prediction)
        preds = torch.cat(preds, dim=0).cpu().numpy()
        self.collect_predictions_to_json(preds)
        self.model.train()
        return test_losses

    def collect_predictions_to_json(self, predictions: List[np.ndarray]) -> None:
        with open("./data/sample_submission.json", "r") as f:
            ground = json.load(f)

        indices = np.array([int(indice) for indice in list(ground.keys())])
        PRED_DATA = OrderedDict()
        for i in indices:
            PRED_DATA[str(i)] = float(predictions[i])

        with open("./submissions/test_submission.json", "w") as f:
            json.dump(PRED_DATA, f)

    def load_model_from_file(
        self, model_path: str, model_params: Dict[str, Any]
    ) -> Module:
        """
        Load a pretrained model from file.

        Parameters
        ----------
        model_path: string
            Path to the file which is model saved.
        model_params: dictionary
            Parameters of the model is needed to initialize.

        Returns
        -------
        model: pytorch Module
            Pretrained model ready for inference, or continue training.
        """
        model = PromoterNet(**model_params).to(device)
        model.load_state_dict(torch.load(model_path + ".pth"))
        return model
