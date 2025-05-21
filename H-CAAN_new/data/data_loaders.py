# data/data_loaders.py

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from torch.utils.data import DataLoader, Dataset, Sampler, SubsetRandomSampler, WeightedRandomSampler
from torch_geometric.data import Dataset as PyGDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

from data.dataset_processors import MolecularDataset


class CollateFunction:
    """Collate function for batching molecular data with multiple modalities."""
    
    def __init__(self, pad_value: int = 0, max_length: int = 512):
        self.pad_value = pad_value
        self.max_length = max_length
    
    def __call__(self, batch):
        """Collate a batch of data samples."""
        # Extract batch elements
        graphs = [data for data in batch]
        
        # Batch the graph data (this is handled by PyG's default collation)
        batch_data = {}
        
        # Handle SMILES encodings (padded sequences)
        if hasattr(batch[0], 'smiles_encoding'):
            smiles_encodings = [data.smiles_encoding for data in batch]
            max_len = min(max(len(encoding) for encoding in smiles_encodings), self.max_length)
            padded_encodings = []
            attention_masks = []
            
            for encoding in smiles_encodings:
                if len(encoding) > max_len:
                    padded = encoding[:max_len]
                    mask = torch.ones(max_len, dtype=torch.long)
                else:
                    padded = torch.cat([
                        encoding, 
                        torch.full((max_len - len(encoding),), self.pad_value, dtype=encoding.dtype)
                    ])
                    mask = torch.cat([
                        torch.ones(len(encoding), dtype=torch.long),
                        torch.zeros(max_len - len(encoding), dtype=torch.long)
                    ])
                padded_encodings.append(padded)
                attention_masks.append(mask)
            
            batch_data['smiles_encoding'] = torch.stack(padded_encodings)
            batch_data['smiles_attention_mask'] = torch.stack(attention_masks)
        
        # Handle ECFP fingerprints
        if hasattr(batch[0], 'ecfp'):
            batch_data['ecfp'] = torch.stack([data.ecfp for data in batch])
        
        # Handle complexity features
        if hasattr(batch[0], 'complexity'):
            batch_data['complexity'] = torch.stack([data.complexity for data in batch])
        
        # Handle MFBERT inputs
        if hasattr(batch[0], 'mfbert_input_ids'):
            mfbert_input_ids = [data.mfbert_input_ids for data in batch]
            mfbert_attention_mask = [data.mfbert_attention_mask for data in batch]
            
            # Ensure all tensors have the same length
            max_len = min(max(len(ids) for ids in mfbert_input_ids), self.max_length)
            padded_input_ids = []
            padded_attention_masks = []
            
            for ids, mask in zip(mfbert_input_ids, mfbert_attention_mask):
                if len(ids) > max_len:
                    padded_ids = ids[:max_len]
                    padded_mask = mask[:max_len]
                else:
                    padded_ids = torch.cat([
                        ids, 
                        torch.full((max_len - len(ids),), self.pad_value, dtype=ids.dtype)
                    ])
                    padded_mask = torch.cat([
                        mask,
                        torch.zeros(max_len - len(mask), dtype=mask.dtype)
                    ])
                padded_input_ids.append(padded_ids)
                padded_attention_masks.append(padded_mask)
            
            batch_data['mfbert_input_ids'] = torch.stack(padded_input_ids)
            batch_data['mfbert_attention_mask'] = torch.stack(padded_attention_masks)
        
        # Handle targets
        if hasattr(batch[0], 'y'):
            batch_data['y'] = torch.cat([data.y for data in batch])
        
        # Add raw SMILES for reference
        if hasattr(batch[0], 'smiles'):
            batch_data['smiles'] = [data.smiles for data in batch]
        
        # Use PyG's built-in batching for graph data
        batch_data.update(PyGDataLoader.collate([data for data in batch]))
        
        return batch_data


def create_data_loaders(train_dataset: MolecularDataset, 
                        val_dataset: MolecularDataset, 
                        test_dataset: MolecularDataset, 
                        batch_size: int = 32, 
                        num_workers: int = 4,
                        use_weighted_sampling: bool = False) -> Tuple[PyGDataLoader, PyGDataLoader, PyGDataLoader]:
    """Create DataLoaders for training, validation, and test sets."""
    collate_fn = CollateFunction()
    
    # Create weighted sampling if requested
    if use_weighted_sampling and hasattr(train_dataset, 'y'):
        # For regression tasks, create bins and sample to balance
        y_train = torch.cat([data.y for data in train_dataset])
        y_values = y_train.numpy().flatten()
        
        # Create bins
        n_bins = 10
        bins = np.linspace(y_values.min(), y_values.max(), n_bins + 1)
        bin_indices = np.digitize(y_values, bins[1:-1])
        
        # Count samples in each bin
        bin_counts = np.bincount(bin_indices, minlength=n_bins)
        
        # Create sample weights (inverse frequency)
        weights = np.zeros_like(y_values)
        for i in range(n_bins):
            if bin_counts[i] > 0:
                weights[bin_indices == i] = 1.0 / bin_counts[i]
            else:
                weights[bin_indices == i] = 0.0
        
        # Normalize weights
        weights = weights / weights.sum() * len(weights)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # Create data loaders
    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = PyGDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader


def create_fold_loaders(dataset: MolecularDataset, 
                        num_folds: int = 5, 
                        batch_size: int = 32, 
                        num_workers: int = 4,
                        stratified: bool = False,
                        random_state: int = 42) -> List[Tuple[PyGDataLoader, PyGDataLoader]]:
    """Create DataLoaders for k-fold cross-validation."""
    collate_fn = CollateFunction()
    
    if stratified and hasattr(dataset, 'y'):
        # For regression, we need to bin the targets for stratification
        y_values = torch.cat([data.y for data in dataset]).numpy().flatten()
        # Create 10 bins
        y_bins = np.digitize(y_values, np.linspace(y_values.min(), y_values.max(), 10))
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
        splits = list(kf.split(np.zeros(len(dataset)), y_bins))
    else:
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
        splits = list(kf.split(np.zeros(len(dataset))))
    
    fold_loaders = []
    for train_indices, val_indices in splits:
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        train_loader = PyGDataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        val_loader = PyGDataLoader(
            dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        fold_loaders.append((train_loader, val_loader))
    
    return fold_loaders


class TaskBatchSampler(Sampler):
    """Batch sampler for multi-task learning that ensures each batch contains samples from the same task."""
    
    def __init__(self, task_indices: Dict[str, List[int]], batch_size: int, shuffle: bool = True):
        """
        Args:
            task_indices: Dictionary mapping task names to lists of dataset indices
            batch_size: Batch size
            shuffle: Whether to shuffle the data
        """
        self.task_indices = task_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Calculate total number of batches
        self.task_batches = {}
        for task, indices in task_indices.items():
            n_batches = len(indices) // batch_size
            if n_batches == 0:
                n_batches = 1  # Ensure at least one batch per task
            self.task_batches[task] = n_batches
        
        self.total_batches = sum(self.task_batches.values())
    
    def __iter__(self):
        """Iterate through batches, ensuring each batch contains samples from the same task."""
        # Shuffle indices within each task if requested
        task_indices = {}
        for task, indices in self.task_indices.items():
            indices_copy = indices.copy()
            if self.shuffle:
                np.random.shuffle(indices_copy)
            task_indices[task] = indices_copy
        
        # Create batches for each task
        all_batches = []
        for task, indices in task_indices.items():
            n_samples = len(indices)
            n_batches = self.task_batches[task]
            
            # Handle case where batch_size > n_samples
            if n_samples < self.batch_size:
                # Repeat indices to fill a batch
                repeats = int(np.ceil(self.batch_size / n_samples))
                extended_indices = indices * repeats
                batch = extended_indices[:self.batch_size]
                all_batches.append(batch)
            else:
                # Create batches
                for i in range(n_batches):
                    start_idx = i * self.batch_size
                    end_idx = min((i + 1) * self.batch_size, n_samples)
                    
                    # Handle last batch if it's smaller than batch_size
                    if end_idx - start_idx < self.batch_size and i == n_batches - 1:
                        # Add indices from the beginning to make a full batch
                        batch = indices[start_idx:end_idx]
                        extra_needed = self.batch_size - (end_idx - start_idx)
                        batch.extend(indices[:extra_needed])
                    else:
                        batch = indices[start_idx:end_idx]
                    
                    all_batches.append(batch)
        
        # Shuffle batches if requested
        if self.shuffle:
            np.random.shuffle(all_batches)
        
        # Return batches
        for batch in all_batches:
            yield batch
    
    def __len__(self):
        """Return the number of batches."""
        return self.total_batches


def create_multitask_loaders(datasets: Dict[str, MolecularDataset], 
                            batch_size: int = 32, 
                            num_workers: int = 4) -> Dict[str, Tuple[PyGDataLoader, PyGDataLoader, PyGDataLoader]]:
    """Create DataLoaders for multi-task learning."""
    collate_fn = CollateFunction()
    loaders = {}
    
    for dataset_name, (train_dataset, val_dataset, test_dataset) in datasets.items():
        # Create task batch samplers for training
        train_indices = list(range(len(train_dataset)))
        val_indices = list(range(len(val_dataset)))
        test_indices = list(range(len(test_dataset)))
        
        train_sampler = TaskBatchSampler({dataset_name: train_indices}, batch_size, shuffle=True)
        
        train_loader = PyGDataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        val_loader = PyGDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        test_loader = PyGDataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        loaders[dataset_name] = (train_loader, val_loader, test_loader)
    
    return loaders


class MultitaskBatchGenerator:
    """Generator for multi-task batches that ensures balanced sampling across tasks."""
    
    def __init__(self, task_loaders: Dict[str, PyGDataLoader], batch_size: int = 32):
        """
        Args:
            task_loaders: Dictionary mapping task names to their DataLoaders
            batch_size: Batch size
        """
        self.task_loaders = task_loaders
        self.batch_size = batch_size
        self.task_iterators = {}
        self.reset_iterators()
        
        # Calculate total number of batches based on the smallest dataset
        min_batches = min(len(loader) for loader in task_loaders.values())
        # Use at least 100 batches per epoch or one pass through the smallest dataset
        self.batches_per_epoch = max(min_batches, 100)
    
    def reset_iterators(self):
        """Reset all task iterators."""
        self.task_iterators = {task: iter(loader) for task, loader in self.task_loaders.items()}
    
    def get_next_batch(self, task: str):
        """Get the next batch for a specific task, resetting the iterator if needed."""
        try:
            batch = next(self.task_iterators[task])
        except StopIteration:
            self.task_iterators[task] = iter(self.task_loaders[task])
            batch = next(self.task_iterators[task])
        return batch
    
    def __iter__(self):
        """Iterate through mixed batches from all tasks."""
        tasks = list(self.task_loaders.keys())
        
        for _ in range(self.batches_per_epoch):
            # Randomly select a task
            task = np.random.choice(tasks)
            batch = self.get_next_batch(task)
            
            # Add task information to the batch
            batch['task'] = task
            
            yield batch
    
    def __len__(self):
        """Return the number of batches per epoch."""
        return self.batches_per_epoch