import os
import sys
import re
import glob
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import transformers
import numpy as np
import pandas as pd

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importing custom modules
from model.cmrclip import CMRCLIP, sim_matrix
from utils.util import state_dict_data_parallel_fix, Find_Optimal_Cutoff
from proto import eval_fewshot
from prompt import *

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=10, num_layers=3):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.network(x)
        return nn.functional.softmax(x, dim=-1)  # Apply softmax along the last dimension

@dataclass
class Config:
    """Configuration class for classification experiments."""
   
    # Paths
    FEATURE_PATH: str = 'comparison/downstream'
    MODEL_PATH: str = 'exps/models/cine_lge_64/0511_084929/model_best.pth'
    RESULTS_PATH: str = 'results'
    
    # Model parameters
    MODEL_NAME: str = 'CMR_CLIP'
    DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Training parameters
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.3
    
    # MLP parameters
    HIDDEN_DIM: int = 128
    NUM_LAYERS: int = 3
    NUM_EPOCHS: int = 30
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 0.0001
    
    # Few-shot parameters
    FEW_SHOT_SIZES: List[int] = None
    FEW_SHOT_ITERATIONS: int = 100
    
    def __post_init__(self):
        if self.FEW_SHOT_SIZES is None:
            self.FEW_SHOT_SIZES = [1, 2, 4, 8, 16, 32]
        
        # Create results directory if it doesn't exist
        Path(self.RESULTS_PATH).mkdir(exist_ok=True)


class DataManager:
    """Handles data loading and preprocessing."""
    
    def __init__(self, config: Config):
        self.config = config
        
    def get_data_paths(self, classification: str) -> Dict[str, str]:
        """Get data paths for different classification tasks."""
        data_paths = {
            'finding':f'{self.config.FEATURE_PATH}/{self.config.MODEL_NAME}/finding/labels.csv',
            'diagnosis': f'{self.config.FEATURE_PATH}/{self.config.MODEL_NAME}/disease/labels.csv',
            'acdc': f'{self.config.FEATURE_PATH}/{self.config.MODEL_NAME}/acdc/labels.csv'
        }
        emb_paths = {
            'finding': f'{self.config.FEATURE_PATH}/{self.config.MODEL_NAME}/finding/emb.npy',
            'diagnosis': f'{self.config.FEATURE_PATH}/{self.config.MODEL_NAME}/disease/emb.npy',
            'acdc': f'{self.config.FEATURE_PATH}/{self.config.MODEL_NAME}/acdc/emb.npy'
        }
        
        return {
            'data_path': data_paths.get(classification),
            'emb_path': emb_paths.get(classification)
        }
    
    def load_embeddings_and_labels(self, classification: str, patterns: Dict) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load embeddings and labels for a given classification task."""
        paths = self.get_data_paths(classification)
        if not paths['data_path'] or not paths['emb_path']:
            raise ValueError(f"Unknown classification type: {classification}")
        # Load classification data
        df_ids = pd.read_csv(paths['data_path'])
        
        # Load video embeddings
        vid_embs = np.load(f"{paths['emb_path']}")
        
        # Create train/test split
        train_id, test_id = train_test_split(
            df_ids.index, 
            test_size=self.config.TEST_SIZE, 
            random_state=self.config.RANDOM_STATE
        )
        
        df_ids.loc[df_ids.index.isin(test_id), 'train_test'] = 'test'
        df_ids.loc[~df_ids.index.isin(test_id), 'train_test'] = 'train'
        return df_ids, vid_embs


class ModelManager:
    """Handles model loading and inference."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained model and tokenizer."""
        video_params = {
            "model": "SpaceTimeTransformer",
            "arch_config": "base_patch16_224",
            "num_frames": 64,
            "pretrained": True,
            "time_init": "zeros"
        }
        
        text_params = {
            "model": "emilyalsentzer/Bio_ClinicalBERT",
            "pretrained": True,
            "input": "text"
        }
        self.model = CMRCLIP(video_params=video_params, text_params=text_params)
        
        try:
            checkpoint = torch.load(self.config.MODEL_PATH)
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(state_dict, self.model.state_dict())
            self.model.load_state_dict(new_state_dict, strict=True)
        except Exception as e:
            print(f"Warning: Could not load model checkpoint: {e}")
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            text_params['model'], 
            TOKENIZERS_PARALLELISM=False
        )
        
        self.model.eval()
        self.model.to(self.config.DEVICE)
    
    def tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize input text."""
        prompt_emb = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        return prompt_emb.to(self.config.DEVICE)
    
    def compute_text_embedding(self, text: str) -> torch.Tensor:
        """Compute text embedding using the model."""
        prompt_emb = self.tokenize_text(text)
        return self.model.compute_text(prompt_emb)


class Evaluator:
    """Handles different evaluation methods."""
    
    def __init__(self, config: Config, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
    
    def evaluate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute evaluation metrics."""
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'F1-score': f1_score(y_true, y_pred, average='binary'),
        }
        
        if y_prob is not None:
            try:
                metrics['AUC'] = roc_auc_score(y_true, y_prob)
            except Exception as e:
                print(f"Warning: Could not compute AUC: {e}")
                metrics['AUC'] = 0.0
        
        return metrics
    
    def run_zero_shot(self, df_ids: pd.DataFrame, vid_embs: np.ndarray, patterns: Dict) -> pd.DataFrame:
        """Run zero-shot evaluation."""
        results = []
        for split_name in ['train', 'test', 'external']:
            df_split = df_ids[df_ids['train_test'] == split_name]
            split_results = []
            
            for idx in df_split.index:
                vid_emb = torch.tensor(vid_embs[idx]).unsqueeze(0).to(self.config.DEVICE)
                result = {}
                
                for task in patterns:
                    similarities = []
                    for text_prompt in patterns[task]['prompt']:
                        txt_emb = self.model_manager.compute_text_embedding(text_prompt)
                        sim = sim_matrix(txt_emb, vid_emb)
                        similarities.append(sim.detach().cpu().numpy()[0][0])
                    result[task] =  df_split[task][idx]
                    result[f'{task}_pred'] = np.mean(similarities)
                
                split_results.append(result)
            
            df_prob = pd.DataFrame(split_results)
            if split_name == 'train':
                df_prob_train = df_prob
            elif split_name == 'test':
                df_prob_test = df_prob
            elif split_name == 'external':
                df_prob_external = df_prob
                
        
        # Evaluate performance
        evaluation_results = []
        eachcase_results = []
        for idx, df_prob in enumerate([df_prob_test, df_prob_external]):
        # for idx, df_prob in enumerate([df_prob_test]):
            if idx==0:
                test_case = 'internal'
            else:
                test_case = 'external'
            
            result_df = pd.DataFrame({'model': self.config.MODEL_NAME, 
                                    'AccessionNumber':df_prob['AccessionNumber'], 
                                    'test_case': test_case})
            
            for task in patterns.keys():
                target = df_prob[task]
                prob = df_prob[f'{task}_pred']
                try:
                    threshold = Find_Optimal_Cutoff(df_prob_train[task], df_prob_train[f'{task}_pred'])
                    pred = (prob > threshold[0]).astype(int)
                except Exception as e:
                    print(f"Warning: Could not find optimal cutoff for {task}: {e}")
                    pred = (prob > 0.5).astype(int)
                metrics = self.evaluate_metrics(target, pred, prob)
                
                result_df[f'{task}_true'] = target
                result_df[f'{task}_pred'] = prob
                result = {
                    'model': self.config.MODEL_NAME,
                    'Label': patterns[task]['label'],
                    'Thershold': threshold[0],
                    'test_case': test_case,
                    **metrics
                }
                evaluation_results.append(result)
            eachcase_results.append(result_df)
        return pd.DataFrame(evaluation_results) , pd.concat(eachcase_results)
    
    def run_few_shot(self, df_ids: pd.DataFrame, vid_embs: np.ndarray, patterns: Dict) -> pd.DataFrame:
        """Run few-shot evaluation."""
        results = []
        
        for task in patterns.keys():
            task_df = df_ids.dropna(subset=[task])
            train_feats = torch.tensor(vid_embs[task_df[task_df['train_test'] == 'train'].index])
            train_labels = torch.tensor(task_df[task_df['train_test'] == 'train'][task].astype(int).values)
            test_feats = torch.tensor(vid_embs[task_df[task_df['train_test'] == 'test'].index])
            test_labels = torch.tensor(task_df[task_df['train_test'] == 'test'][task].astype(int).values)
            external_feats = torch.tensor(vid_embs[task_df[task_df['train_test'] == 'external'].index])
            external_labels = torch.tensor(task_df[task_df['train_test'] == 'external'][task].astype(int).values)
            
            for shot in self.config.FEW_SHOT_SIZES:
                try:
                    results_df, results_agg, results_df_external, results_agg_external = eval_fewshot(
                        train_feats=train_feats,
                        train_labels=train_labels,
                        test_feats=test_feats,
                        test_labels=test_labels,
                        external_feats= external_feats,
                        external_labels= external_labels,
                        n_iter=self.config.FEW_SHOT_ITERATIONS,
                        n_way=-1,
                        n_shot=shot,
                        n_query=len(test_feats),
                        center_feats=True,
                        normalize_feats=True,
                        random_seed=self.config.RANDOM_STATE,  # Add this parameter
                    )
                    
                    result.update({
                        'model': self.config.MODEL_NAME,
                        'shot': shot,
                        'Label': patterns[task]['label'],
                        'test_case': 'internal'
                    })
                    result_external.update({
                        'model': self.config.MODEL_NAME,
                        'shot': shot,
                        'Label': patterns[task]['label'],
                        'test_case': 'external'
                    })
                    results.append(result)
                    results.append(result_external)
                    
                except Exception as e:
                    print(f"Warning: Few-shot evaluation failed for {task} with {shot} shots: {e}")
        
        return pd.DataFrame(results)
    
    def run_supervised(self, df_ids: pd.DataFrame, vid_embs: np.ndarray, patterns: List) -> pd.DataFrame:
        """Run supervised evaluation."""
        results = []
        for task in patterns:
            task_df = df_ids.dropna(subset=[task])
            train_feats = torch.tensor(vid_embs[task_df[task_df['train_test'] == 'train'].index])
            train_labels = torch.tensor(task_df[task_df['train_test'] == 'train'][task].astype(int).values)
            test_feats = torch.tensor(vid_embs[task_df[task_df['train_test'] == 'test'].index])
            test_labels = torch.tensor(task_df[task_df['train_test'] == 'test'][task].astype(int).values)
            
            # Normalize features
            scaler = StandardScaler()
            train_feats = torch.tensor(scaler.fit_transform(train_feats)).float()
            test_feats = torch.tensor(scaler.transform(test_feats)).float()
            
            try:
                # Use MLP classifier
                input_dim = train_feats.shape[1]
                output_dim = len(torch.unique(train_labels))
                
                mlp_model = MLP(
                    input_dim=input_dim,
                    hidden_dim=self.config.HIDDEN_DIM,
                    output_dim=output_dim,
                    num_layers=self.config.NUM_LAYERS
                )
                
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(mlp_model.parameters(), lr=self.config.LEARNING_RATE)
                
                # Training
                train_dataset = TensorDataset(train_feats, train_labels)
                train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
                
                for epoch in range(self.config.NUM_EPOCHS):
                    mlp_model.train()
                    for features, labels in train_loader:
                        optimizer.zero_grad()
                        outputs = mlp_model(features)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                
                # Evaluation
                mlp_model.eval()
                with torch.no_grad():
                    test_outputs = mlp_model(test_feats)
                    _, y_pred = torch.max(test_outputs, 1)
                    
                    # Get probabilities for AUC calculation
                    y_prob = F.softmax(test_outputs, dim=1)[:, 1] if output_dim == 2 else None
                
                metrics = self.evaluate_metrics(test_labels.numpy(), y_pred.numpy(), 
                                               y_prob.numpy() if y_prob is not None else None)
                
                result = {
                    'model': self.config.MODEL_NAME,
                    'Label': task,
                    **metrics
                }
                results.append(result)
                
            except Exception as e:
                print(f"Warning: Supervised evaluation failed for {task}: {e}")
        
        return pd.DataFrame(results)


class Classifier:
    """Main classifier class that orchestrates the evaluation process."""
    
    def __init__(self, classification: str, learning_type: str, config: Optional[Config] = None):
        self.classification = classification
        self.learning_type = learning_type
        self.config = config or Config()
        self.patterns = self._get_patterns()
        self.data_manager = DataManager(self.config)
        self.model_manager = ModelManager(self.config)
        self.evaluator = Evaluator(self.config, self.model_manager)
    
    def _get_patterns(self) ->  Union[Dict, List]:
        """Get patterns for the classification task."""
        pattern_map = {
            'finding': fining_prompt,
            'finding_ext': fining_prompt,
            'diagnosis': diagnosis_lst,
            'acdc': acdc_lst
        }
        
        if self.classification not in pattern_map:
            raise ValueError(f"Unknown classification type: {self.classification}")
        
        return pattern_map[self.classification]
    
    def run(self) -> pd.DataFrame:
        """Run the classification evaluation."""
        print(f"Running {self.learning_type} evaluation for {self.classification}")
        
        # Load data
        df_ids, vid_embs = self.data_manager.load_embeddings_and_labels(
            self.classification, self.patterns
        )
        
        # Run evaluation
        if self.learning_type == 'zeroshot':
            results, results2 = self.evaluator.run_zero_shot(df_ids, vid_embs, self.patterns)
            output_path = f"{self.config.RESULTS_PATH}/{self.classification}_{self.config.MODEL_NAME}_{self.learning_type}_pred.csv"
            results2.to_csv(output_path, index=False)
        elif self.learning_type == 'fewshot':
            results = self.evaluator.run_few_shot(df_ids, vid_embs, self.patterns)
        elif self.learning_type == 'supervised':
            results = self.evaluator.run_supervised(df_ids, vid_embs, self.patterns)
        else:
            raise ValueError(f"Unknown learning type: {self.learning_type}")
        
        # Save results
        output_path = f"{self.config.RESULTS_PATH}/{self.classification}_{self.config.MODEL_NAME}_{self.learning_type}.csv"
        results.to_csv(output_path, index=False)
        
        print(f"Results saved to {output_path}")
        print(results)
        
        return results


def main():
    """Main function to run all experiments."""
    config = Config()
    
    # Define experimental conditions
    experiments = {
        'finding': ['zeroshot', 'fewshot'],
        'finding_ext': ['zeroshot', 'fewshot'], 
        'diagnosis': ['supervised'],
        'acdc': ['supervised']
    }
    
    for classification, learning_types in experiments.items():
        for learning_type in learning_types:
            try:
                classifier = Classifier(classification, learning_type, config)
                classifier.run()
            except Exception as e:
                print(f"Error running {classification} {learning_type}: {e}")
                continue


if __name__ == "__main__":
    main()