"""
Code based on sampler from @mileyan/simple_shot
Adapted from https://github.com/mbanani/lgssl/blob/df45bae647fc24dce8a6329eb697944053e9a8a0/lgssl/evaluation/fewshot.py.
"""

import logging
from typing import Any, List, Tuple
import os

import numpy as np
import pandas as pd
import sklearn.neighbors
import torch
from torch.nn.functional import normalize
from torch.utils.data import Sampler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.special import softmax


from typing import Optional, Dict, Any, Union, List
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
    cohen_kappa_score,
    classification_report,
)

def get_eval_metrics(
    targets_all: Union[List[int], np.ndarray],
    preds_all: Union[List[int], np.ndarray],
    probs_all: Optional[Union[List[float], np.ndarray]] = None,
    get_report: bool = True,
    prefix: str = "",
    roc_kwargs: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics and return the evaluation metrics.

    Args:
        targets_all (array-like): True target values.
        preds_all (array-like): Predicted target values.
        probs_all (array-like, optional): Predicted probabilities for each class. Defaults to None.
        get_report (bool, optional): Whether to include the classification report in the results. Defaults to True.
        prefix (str, optional): Prefix to add to the result keys. Defaults to "".
        roc_kwargs (dict, optional): Additional keyword arguments for calculating ROC AUC. Defaults to {}.

    Returns:
        dict: Dictionary containing the evaluation metrics.

    """
    if len(torch.unique(targets_all)) >= 3:
        acc = accuracy_score(targets_all, preds_all)
        f1 = f1_score(targets_all, preds_all, average='macro')
        roc_auc = roc_auc_score(targets_all, probs_all, multi_class='ovr', average='micro')

    else:
        acc = accuracy_score(targets_all, preds_all)
        f1 = f1_score(targets_all, preds_all)
        roc_auc = roc_auc_score(targets_all, probs_all)

    eval_metrics = {
        'Accuracy': acc,
        'F1-score': f1,
        'AUC': roc_auc
    }
    if get_report:
        eval_metrics[f"{prefix}report"] = cls_rep

    return eval_metrics

def print_metrics(eval_metrics):
    for k, v in eval_metrics.items():
        if "report" in k:
            continue
        
        print(f"Test {k}: {v:.3f}")

def eval_fewshot(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    external_feats: torch.Tensor,
    external_labels: torch.Tensor,
    n_iter: int = 1000,
    n_way: int = -1,
    n_shot: int = 256,
    n_query: int = -1,
    center_feats: bool = True,
    normalize_feats: bool = True,
    average_feats: bool = True,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, dict]:
    """
    Evaluate few-shot learning performance.

    Args:
        train_feats (torch.Tensor): Training features.
        train_labels (torch.Tensor): Training labels.
        test_feats (torch.Tensor): Test features.
        test_labels (torch.Tensor): Test labels.
        external_feats (torch.Tensor): External features.
        external_labels (torch.Tensor): External labels.
        n_iter (int, optional): Num iterations. Defaults to 1000.
        n_way (int, optional): Num classes per few-shot task. Defaults to -1 (use all classes in test set).
        n_shot (int, optional): Num support examples per class. Defaults to 256 examples per class in train set.
        n_query (int, optional): Num query examples per class. Defaults to -1 (use all examples in test set).
        center_feats (bool, optional): Whether to center the features. Defaults to True.
        normalize_feats (bool, optional): Whether to normalize the features. Defaults to True.
        average_feats (bool, optional): Whether to average the features. Defaults to True.
        random_seed (int, optional): Random seed for reproducible sampling. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, dict]: A tuple containing the results from every few-shot episode and its mean/std.
    """
    # Set random seed for reproducible sampling
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # logging.info(
    #     f"FS Evaluation: n_iter: {n_iter}, n_way: {n_way}, n_shot: {n_shot}, n_query: {n_query}, center_feats: {center_feats}, normalize_feats: {normalize_feats}, average_feats: {average_feats}"
    # )
    # logging.info(f"FS Evaluation: Train Shape {train_feats.shape}")
    # logging.info(f"FS Evaluation: Test Shape {test_feats.shape}")
    if n_way == -1:
        n_way = len(np.unique(train_labels))
        assert n_way == len(np.unique(test_labels))

    if n_query == -1:
        # logging.info("Using all test samples for query")
        print("Using all test samples for query")

    # Set up sampler
    fewshot_sampler = FewShotEpisodeSampler(
        train_labels,
        test_labels,
        external_labels,
        n_iter,
        n_way,
        n_shot,
        n_query,
        random_seed,
    )
    # test model on dataset -- really more tasks than batches
    results_all = []
    results_all_external = []
    n_way = n_way
    n_shot = n_shot
    for task in tqdm(fewshot_sampler):
        source, query, external = task
        # get train and test
        feats_source = train_feats[source]
        labels_source = train_labels[source]
        if n_query == -1:
            feats_query = test_feats.detach().clone()
            labels_query = test_labels.detach().clone()
        else:
            feats_query = test_feats[query]
            labels_query = test_labels[query]
            
            feats_external = external_feats[external]
            labels_external = external_labels[external]
        # center
        if center_feats:
            feats_mean = feats_source.mean(dim=0, keepdims=True)
            feats_query = feats_query - feats_mean
            feats_source = feats_source - feats_mean
            feats_external = feats_external - feats_mean

        # normalize
        if normalize_feats:
            feats_source = normalize(feats_source, dim=-1, p=2)
            feats_query = normalize(feats_query, dim=-1, p=2)
            feats_external = normalize(feats_external, dim=-1, p=2)

        # compute prototypes & assert labels are correct
        if average_feats:
            feats_proto = feats_source.view(n_way, n_shot, -1).mean(dim=1)
            labels_proto = labels_source.view(n_way, n_shot)
            try:
                assert (labels_proto.min(dim=1).values == labels_proto.max(dim=1).values).all()
            except:
                breakpoint()
            labels_proto = labels_proto[:, 0]
        else:
            feats_proto = feats_source
            labels_proto = labels_source

        # classify to prototypes
        pw_dist = (feats_query[:, None] - feats_proto[None, :]).norm(dim=-1, p=3)
        pw_dist_external = (feats_external[:, None] - feats_proto[None, :]).norm(dim=-1, p=3)
        # Sum of each row (keeping dimensions for broadcasting)
        row_sums = pw_dist.sum(dim=1, keepdim=True)
        row_sums_external = pw_dist_external.sum(dim=1, keepdim=True)
        # Normalize each row
        labels_prob = pw_dist / row_sums
        labels_prob_external = pw_dist_external / row_sums_external
        labels_pred = labels_proto[pw_dist.min(dim=1).indices]
        labels_pred_external = labels_proto[pw_dist_external.min(dim=1).indices]
        inv_index = np.argsort(labels_proto.tolist())

        neg_scores = -labels_prob[:,inv_index]
        probabilities = softmax(neg_scores, axis=1)
        neg_scores_external = -labels_prob_external[:,inv_index]
        probabilities_external = softmax(neg_scores_external, axis=1)
        
        if len(torch.unique(labels_query)) >= 3:
            results = get_eval_metrics(labels_query, labels_pred, probabilities, get_report=False, prefix="internal_")
            results_external = get_eval_metrics(labels_external, labels_pred_external, probabilities_external, get_report=False, prefix="external_")
        else:
            results = get_eval_metrics(labels_query, labels_pred, labels_prob[:,labels_proto[0]], get_report=False, prefix="")
            results_external = get_eval_metrics(labels_external, labels_pred_external, labels_prob_external[:,labels_proto[0]], get_report=False, prefix="external_")
        
        results_all.append(results)
        results_all_external.append(results_external)
    # compute metrics for model
    results_df = pd.DataFrame(results_all)
    results_agg = dict(
        zip(
            list(results_df.columns + "_avg") + list(results_df.columns + "_std"),
            results_df.agg(["mean", "std"], axis=0).values.flatten(),
        )
    )
    
    results_df_external = pd.DataFrame(results_all_external)
    results_agg_external = dict(
        zip(
            list(results_df_external.columns + "_avg") + list(results_df_external.columns + "_std"),
            results_df_external.agg(["mean", "std"], axis=0).values.flatten(),
        )
    )
    return results_df, results_agg, results_df_external, results_agg_external


class FewShotEpisodeSampler(Sampler):
    """
    Sampler for generating few-shot episodes for training or evaluation.

    Adapted from https://github.com/mbanani/lgssl/blob/df45bae647fc24dce8a6329eb697944053e9a8a0/lgssl/evaluation/fewshot.py.
    """

    def __init__(
        self,
        train_labels: List[int],
        test_labels: List[int],
        external_labels: List[int],
        n_iter: int,
        n_way: int,
        n_shot: int,
        n_query: int,
        random_seed: int = 42,
    ) -> None:
        """
        Args:
            train_labels (list): List of training labels.
            test_labels (list): List of test labels.
            external_labels (list): List of external labels.
            n_iter (int): Number of iterations (episodes) to generate.
            n_way (int): Number of classes per episode.
            n_shot (int): Number of samples per class in the support set.
            n_query (int): Number of samples per class in the query set.
            random_seed (int): Random seed for reproducible sampling.
        """
        self.n_iter = n_iter
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.random_seed = random_seed
        
        train_labels = np.array(train_labels)
        self.train_ind = []
        self.test_ind = []
        self.external_ind = []
        unique = np.unique(train_labels)
        unique = np.sort(unique)
        for i in unique:
            train_ind = np.argwhere(train_labels == i).reshape(-1)
            self.train_ind.append(train_ind)

            test_ind = np.argwhere(test_labels == i).reshape(-1)
            self.test_ind.append(test_ind)

            external_ind = np.argwhere(external_labels == i).reshape(-1)
            self.external_ind.append(external_ind)

    def __len__(self) -> int:
        return self.n_iter

    def __iter__(self) -> Tuple[Any, Any]:
        # Set seed at the beginning of each iteration cycle
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        for iter_idx in range(self.n_iter):
            # Use a different seed for each iteration but consistent across models
            iter_seed = self.random_seed + iter_idx
            torch.manual_seed(iter_seed)
            
            batch_gallery = []
            batch_query = []
            batch_external = []
            classes = torch.randperm(len(self.train_ind))[: self.n_way]
            for c in classes:
                train_c = self.train_ind[c.item()]
                # assert len(train_c) >= (self.n_shot), f"{len(train_c)} < {self.n_shot}"
                train_pos = torch.multinomial(torch.ones(len(train_c)), self.n_shot, replacement=True)
                batch_gallery.append(train_c[train_pos])

                test_c = self.test_ind[c.item()]
                if len(test_c) < (self.n_query):
                    # logging.info(f"test class has {len(test_c)} ins. (< {self.n_query})")
                    batch_query.append(test_c)
                else:
                    test_pos = torch.multinomial(torch.ones(len(test_c)), self.n_query)
                    batch_query.append(test_c[test_pos])

                external_c = self.external_ind[c.item()]
                if len(external_c) < (self.n_query):
                    # logging.info(f"test class has {len(test_c)} ins. (< {self.n_query})")
                    batch_external.append(external_c)
                else:
                    external_pos = torch.multinomial(torch.ones(len(external_c)), self.n_query)
                    batch_external.append(external_c[external_pos])

            if self.n_shot == 1:
                batch_gallery = np.array(batch_gallery)
                batch_query = np.concatenate(batch_query)
                batch_external = np.concatenate(batch_external)
            else:
                batch_gallery = np.concatenate(batch_gallery)
                batch_query = np.concatenate(batch_query)
                batch_external = np.concatenate(batch_external)

            yield (batch_gallery, batch_query, batch_external)
            

