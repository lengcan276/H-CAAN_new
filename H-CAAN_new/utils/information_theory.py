import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Union, Optional, Any
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
import logging
import math

logger = logging.getLogger(__name__)

class InformationTheory:
    """
    Tools for information theory analysis of molecular representations.
    """
    
    @staticmethod
    def calculate_entropy(data: np.ndarray) -> float:
        """
        Calculate Shannon entropy of data.
        
        Args:
            data (np.ndarray): Input data
            
        Returns:
            float: Entropy value in bits
        """
        if data is None or len(data) == 0:
            return 0.0
        
        try:
            # Discretize continuous data if needed
            if data.dtype in [np.float32, np.float64]:
                # Simple discretization using histogram
                hist, _ = np.histogram(data, bins=50, density=True)
                hist = hist[hist > 0]  # Remove zeros
                return entropy(hist, base=2)
            else:
                # For discrete data
                _, counts = np.unique(data, return_counts=True)
                probs = counts / len(data)
                return entropy(probs, base=2)
        except Exception as e:
            logger.error(f"Error calculating entropy: {str(e)}")
            return 0.0
    
    @staticmethod
    def estimate_mutual_information(x: np.ndarray, y: np.ndarray, discrete: bool = False) -> float:
        """
        Estimate mutual information between two variables.
        
        Args:
            x (np.ndarray): First variable
            y (np.ndarray): Second variable
            discrete (bool): Whether the data is already discrete
            
        Returns:
            float: Mutual information value in bits
        """
        if x is None or y is None or len(x) != len(y) or len(x) == 0:
            return 0.0
        
        try:
            if not discrete:
                # Discretize continuous data
                x_binned = np.digitize(x, bins=np.linspace(min(x), max(x), 50))
                y_binned = np.digitize(y, bins=np.linspace(min(y), max(y), 50))
                return mutual_info_score(x_binned, y_binned)
            else:
                return mutual_info_score(x, y)
        except Exception as e:
            logger.error(f"Error calculating mutual information: {str(e)}")
            return 0.0
    
    @staticmethod
    def calculate_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate Kullback-Leibler divergence between two distributions.
        
        Args:
            p (np.ndarray): First distribution
            q (np.ndarray): Second distribution
            
        Returns:
            float: KL divergence value
        """
        if p is None or q is None or len(p) != len(q) or len(p) == 0:
            return 0.0
        
        try:
            # Ensure distributions sum to 1
            p = p / np.sum(p)
            q = q / np.sum(q)
            
            # Replace zeros to avoid division by zero
            q = np.maximum(q, 1e-10)
            
            return np.sum(p * np.log2(p / q))
        except Exception as e:
            logger.error(f"Error calculating KL divergence: {str(e)}")
            return 0.0
    
    @staticmethod
    def calculate_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate Jensen-Shannon divergence between two distributions.
        
        Args:
            p (np.ndarray): First distribution
            q (np.ndarray): Second distribution
            
        Returns:
            float: JS divergence value
        """
        if p is None or q is None or len(p) != len(q) or len(p) == 0:
            return 0.0
        
        try:
            # Ensure distributions sum to 1
            p = p / np.sum(p)
            q = q / np.sum(q)
            
            # Compute midpoint distribution
            m = 0.5 * (p + q)
            
            # Calculate JS divergence
            return 0.5 * (InformationTheory.calculate_kl_divergence(p, m) + 
                          InformationTheory.calculate_kl_divergence(q, m))
        except Exception as e:
            logger.error(f"Error calculating JS divergence: {str(e)}")
            return 0.0
    
    @staticmethod
    def information_gain_ratio(data: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate information gain ratio for feature selection.
        
        Args:
            data (np.ndarray): Feature data
            labels (np.ndarray): Class labels
            
        Returns:
            float: Information gain ratio
        """
        if data is None or labels is None or len(data) != len(labels) or len(data) == 0:
            return 0.0
        
        try:
            # Discretize data if continuous
            if data.dtype in [np.float32, np.float64]:
                data_discrete = np.digitize(data, bins=np.linspace(min(data), max(data), 10))
            else:
                data_discrete = data
            
            # Calculate entropy of labels
            label_entropy = InformationTheory.calculate_entropy(labels)
            
            # Calculate conditional entropy
            conditional_entropy = 0.0
            unique_values, counts = np.unique(data_discrete, return_counts=True)
            
            for value, count in zip(unique_values, counts):
                prob = count / len(data_discrete)
                subset_labels = labels[data_discrete == value]
                subset_entropy = InformationTheory.calculate_entropy(subset_labels)
                conditional_entropy += prob * subset_entropy
            
            # Calculate information gain
            information_gain = label_entropy - conditional_entropy
            
            # Calculate intrinsic information
            intrinsic_info = InformationTheory.calculate_entropy(data_discrete)
            
            # Avoid division by zero
            if intrinsic_info == 0:
                return 0.0
            
            # Calculate information gain ratio
            return information_gain / intrinsic_info
        except Exception as e:
            logger.error(f"Error calculating information gain ratio: {str(e)}")
            return 0.0
    
    @staticmethod
    def calculate_mutual_information_matrix(features: List[np.ndarray]) -> np.ndarray:
        """
        Calculate mutual information matrix between multiple features.
        
        Args:
            features (List[np.ndarray]): List of feature arrays
            
        Returns:
            np.ndarray: Mutual information matrix
        """
        if not features or len(features) == 0:
            return np.array([])
        
        n_features = len(features)
        mi_matrix = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    # Self-information is entropy
                    mi_matrix[i, j] = 1.0
                else:
                    # Mutual information between different features
                    mi_matrix[i, j] = InformationTheory.estimate_mutual_information(
                        features[i], features[j]
                    )
        
        return mi_matrix
    
    @staticmethod
    def evaluate_representation_quality(
        representations: Dict[str, np.ndarray], 
        labels: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the quality of different representations using information theory.
        
        Args:
            representations (Dict[str, np.ndarray]): Dictionary of representations
            labels (np.ndarray): Target labels
            
        Returns:
            Dict[str, Dict[str, float]]: Evaluation results
        """
        if not representations or labels is None or len(labels) == 0:
            return {}
        
        results = {}
        
        for name, representation in representations.items():
            try:
                if representation is None or len(representation) == 0:
                    continue
                
                # Flattening representations if needed
                if len(representation.shape) > 1:
                    representation = representation.reshape(representation.shape[0], -1)
                
                if representation.shape[0] != len(labels):
                    logger.warning(f"Representation {name} shape mismatch with labels")
                    continue
                
                # For high-dimensional representations, we'll calculate mutual information 
                # between principal components and labels
                if representation.shape[1] > 10:
                    # Use PCA to reduce dimensionality
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=10)
                    representation_reduced = pca.fit_transform(representation)
                else:
                    representation_reduced = representation
                
                # Calculate entropy of the representation
                representation_entropy = 0
                for i in range(representation_reduced.shape[1]):
                    representation_entropy += InformationTheory.calculate_entropy(representation_reduced[:, i])
                
                # Calculate mutual information with labels
                mutual_info = 0
                for i in range(representation_reduced.shape[1]):
                    mutual_info += InformationTheory.estimate_mutual_information(
                        representation_reduced[:, i], labels
                    )
                
                # Calculate information efficiency
                if representation_entropy > 0:
                    information_efficiency = mutual_info / representation_entropy
                else:
                    information_efficiency = 0
                
                results[name] = {
                    'entropy': float(representation_entropy),
                    'mutual_information': float(mutual_info),
                    'information_efficiency': float(information_efficiency)
                }
            except Exception as e:
                logger.error(f"Error evaluating representation {name}: {str(e)}")
                results[name] = {
                    'entropy': 0.0,
                    'mutual_information': 0.0,
                    'information_efficiency': 0.0
                }
        
        return results
    
    @staticmethod
    def calculate_redundancy(mi_matrix: np.ndarray) -> float:
        """
        Calculate redundancy in the mutual information matrix.
        
        Args:
            mi_matrix (np.ndarray): Mutual information matrix
            
        Returns:
            float: Redundancy score (0-1)
        """
        if mi_matrix is None or mi_matrix.size == 0:
            return 0.0
        
        try:
            # Normalize matrix for redundancy calculation
            np.fill_diagonal(mi_matrix, 0)  # Ignore self-information
            if np.sum(mi_matrix) == 0:
                return 0.0
            
            n = mi_matrix.shape[0]
            max_mi = np.max(mi_matrix)
            
            if max_mi == 0:
                return 0.0
            
            # Calculate normalized redundancy
            redundancy = np.sum(mi_matrix) / (n * (n - 1) * max_mi)
            
            return redundancy
        except Exception as e:
            logger.error(f"Error calculating redundancy: {str(e)}")
            return 0.0
    
    @staticmethod
    def calculate_complementarity(mi_matrix: np.ndarray, target_mi: np.ndarray) -> float:
        """
        Calculate complementarity between features with respect to target.
        
        Args:
            mi_matrix (np.ndarray): Mutual information matrix between features
            target_mi (np.ndarray): Mutual information between features and target
            
        Returns:
            float: Complementarity score (0-1)
        """
        if mi_matrix is None or target_mi is None or mi_matrix.size == 0 or target_mi.size == 0:
            return 0.0
        
        try:
            n = mi_matrix.shape[0]
            if n != len(target_mi):
                logger.error("Mismatch between MI matrix and target MI vector dimensions")
                return 0.0
            
            # Calculate total mutual information with target
            total_target_mi = np.sum(target_mi)
            
            if total_target_mi == 0:
                return 0.0
            
            # Calculate average redundancy
            np.fill_diagonal(mi_matrix, 0)  # Ignore self-information
            avg_redundancy = np.sum(mi_matrix) / (n * (n - 1)) if n > 1 else 0
            
            if avg_redundancy == 0:
                return 1.0  # No redundancy means perfect complementarity
            
            # Normalize target MI and redundancy
            normalized_target_mi = total_target_mi / n
            
            # Calculate complementarity score
            complementarity = normalized_target_mi / (normalized_target_mi + avg_redundancy)
            
            return complementarity
        except Exception as e:
            logger.error(f"Error calculating complementarity: {str(e)}")
            return 0.0

    @staticmethod
    def estimate_multimodal_synergy(
        representations: Dict[str, np.ndarray], 
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Estimate synergy between different modalities using information theory.
        
        Args:
            representations (Dict[str, np.ndarray]): Dictionary of representations
            labels (np.ndarray): Target labels
            
        Returns:
            Dict[str, float]: Synergy scores for modality combinations
        """
        if not representations or labels is None or len(labels) == 0:
            return {}
        
        results = {}
        modalities = list(representations.keys())
        
        # Calculate information for individual modalities
        individual_mi = {}
        for name, representation in representations.items():
            try:
                if representation is None or len(representation) == 0:
                    individual_mi[name] = 0.0
                    continue
                
                # Flatten if needed
                if len(representation.shape) > 1:
                    representation = representation.reshape(representation.shape[0], -1)
                
                # Apply dimensionality reduction if needed
                if representation.shape[1] > 10:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=10)
                    representation = pca.fit_transform(representation)
                
                # Calculate mutual information with labels
                mi = 0.0
                for i in range(representation.shape[1]):
                    mi += InformationTheory.estimate_mutual_information(
                        representation[:, i], labels
                    )
                
                individual_mi[name] = mi
            except Exception as e:
                logger.error(f"Error calculating MI for {name}: {str(e)}")
                individual_mi[name] = 0.0
        
        # Calculate synergy for pairs of modalities
        for i in range(len(modalities)):
            for j in range(i+1, len(modalities)):
                mod1 = modalities[i]
                mod2 = modalities[j]
                
                try:
                    rep1 = representations[mod1]
                    rep2 = representations[mod2]
                    
                    if rep1 is None or rep2 is None or len(rep1) == 0 or len(rep2) == 0:
                        results[f"{mod1}+{mod2}"] = 0.0
                        continue
                    
                    # Flatten if needed
                    if len(rep1.shape) > 1:
                        rep1 = rep1.reshape(rep1.shape[0], -1)
                    if len(rep2.shape) > 1:
                        rep2 = rep2.reshape(rep2.shape[0], -1)
                    
                    # Apply dimensionality reduction if needed
                    if rep1.shape[1] > 10:
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=10)
                        rep1 = pca.fit_transform(rep1)
                    
                    if rep2.shape[1] > 10:
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=10)
                        rep2 = pca.fit_transform(rep2)
                    
                    # Combine representations
                    combined = np.hstack([rep1, rep2])
                    
                    # Calculate mutual information of combined representation
                    combined_mi = 0.0
                    for i in range(combined.shape[1]):
                        combined_mi += InformationTheory.estimate_mutual_information(
                            combined[:, i], labels
                        )
                    
                    # Calculate synergy
                    synergy = combined_mi - (individual_mi[mod1] + individual_mi[mod2])
                    
                    # Normalize by total information
                    if combined_mi > 0:
                        normalized_synergy = synergy / combined_mi
                    else:
                        normalized_synergy = 0.0
                    
                    results[f"{mod1}+{mod2}"] = float(normalized_synergy)
                except Exception as e:
                    logger.error(f"Error calculating synergy for {mod1}+{mod2}: {str(e)}")
                    results[f"{mod1}+{mod2}"] = 0.0
        
        return results