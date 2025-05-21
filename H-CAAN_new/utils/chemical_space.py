import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, PandasTools
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, UMAP
from typing import List, Dict, Tuple, Union, Optional, Any
import logging
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)

class ChemicalSpace:
    """
    Tools for analyzing and visualizing chemical space.
    """
    
    @staticmethod
    def calculate_pairwise_similarities(mols: List[Chem.Mol], fingerprint_radius: int = 2) -> np.ndarray:
        """
        Calculate pairwise similarity matrix between molecules.
        
        Args:
            mols (List[Chem.Mol]): List of RDKit molecules
            fingerprint_radius (int): Radius for Morgan fingerprint
            
        Returns:
            np.ndarray: Similarity matrix
        """
        if not mols or len(mols) == 0:
            return np.array([])
        
        n_mols = len(mols)
        similarity_matrix = np.zeros((n_mols, n_mols))
        
        # Generate fingerprints
        fingerprints = []
        for mol in mols:
            if mol is None:
                fingerprints.append(None)
            else:
                fingerprints.append(AllChem.GetMorganFingerprintAsBitVect(mol, fingerprint_radius, nBits=2048))
        
        # Calculate similarities
        for i in range(n_mols):
            for j in range(i, n_mols):
                if fingerprints[i] is None or fingerprints[j] is None:
                    similarity = 0.0
                else:
                    try:
                        from rdkit import DataStructs
                        similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                    except Exception as e:
                        logger.error(f"Error calculating similarity: {str(e)}")
                        similarity = 0.0
                
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        return similarity_matrix
    
    @staticmethod
    def dimensionality_reduction(
        features: np.ndarray, 
        method: str = 'pca', 
        n_components: int = 2, 
        random_state: int = 42
    ) -> np.ndarray:
        """
        Perform dimensionality reduction on molecular features.
        
        Args:
            features (np.ndarray): Feature matrix
            method (str): Reduction method ('pca', 'tsne', 'umap', 'svd')
            n_components (int): Number of components
            random_state (int): Random state for reproducibility
            
        Returns:
            np.ndarray: Reduced features
        """
        if features is None or features.size == 0:
            return np.array([])
        
        try:
            # Apply dimensionality reduction
            if method.lower() == 'pca':
                model = PCA(n_components=n_components, random_state=random_state)
            elif method.lower() == 'tsne':
                model = TSNE(n_components=n_components, random_state=random_state)
            elif method.lower() == 'umap':
                model = UMAP(n_components=n_components, random_state=random_state)
            elif method.lower() == 'svd':
                model = TruncatedSVD(n_components=n_components, random_state=random_state)
            else:
                logger.warning(f"Unknown method: {method}. Using PCA.")
                model = PCA(n_components=n_components, random_state=random_state)
            
            # Fit and transform
            reduced_features = model.fit_transform(features)
            
            return reduced_features
        except Exception as e:
            logger.error(f"Error in dimensionality reduction: {str(e)}")
            return np.zeros((features.shape[0], n_components))
    
    @staticmethod
    def plot_chemical_space(
        reduced_features: np.ndarray,
        labels: Optional[np.ndarray] = None,
        smiles: Optional[List[str]] = None,
        title: str = 'Chemical Space Visualization',
        colormap: str = 'viridis',
        alpha: float = 0.7,
        figsize: Tuple[int, int] = (12, 10),
        annotations: List[str] = None
    ) -> plt.Figure:
        """
        Plot chemical space visualization.
        
        Args:
            reduced_features (np.ndarray): Reduced feature matrix (2D)
            labels (np.ndarray, optional): Labels for coloring points
            smiles (List[str], optional): SMILES strings for annotations
            title (str): Plot title
            colormap (str): Matplotlib colormap
            alpha (float): Point transparency
            figsize (Tuple[int, int]): Figure size
            annotations (List[str], optional): Custom annotations for points
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        if reduced_features is None or reduced_features.shape[1] < 2:
            logger.error("Invalid reduced features for plotting")
            return plt.figure()
        
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot points
            if labels is not None:
                scatter = ax.scatter(
                    reduced_features[:, 0],
                    reduced_features[:, 1],
                    c=labels,
                    cmap=colormap,
                    alpha=alpha,
                    s=50,
                    edgecolors='w',
                    linewidth=0.5
                )
                
                # Add colorbar if labels are continuous
                if np.issubdtype(labels.dtype, np.number):
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('Property Value')
                else:
                    # For categorical labels
                    unique_labels = np.unique(labels)
                    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=plt.cm.get_cmap(colormap)(i/len(unique_labels)), 
                                         markersize=10) for i in range(len(unique_labels))]
                    ax.legend(handles, unique_labels, title='Categories')
            else:
                ax.scatter(
                    reduced_features[:, 0],
                    reduced_features[:, 1],
                    alpha=alpha,
                    s=50,
                    edgecolors='w',
                    linewidth=0.5
                )
            
            # Add annotations
            if annotations is not None and len(annotations) > 0:
                for i, ann in enumerate(annotations):
                    if i < reduced_features.shape[0]:
                        ax.annotate(
                            ann,
                            xy=(reduced_features[i, 0], reduced_features[i, 1]),
                            xytext=(5, 5),
                            textcoords='offset points',
                            fontsize=9
                        )
            elif smiles is not None and len(smiles) > 0:
                # Select a subset of points to annotate to avoid overcrowding
                n_smiles = len(smiles)
                n_annotations = min(20, n_smiles)
                step = max(1, n_smiles // n_annotations)
                
                for i in range(0, n_smiles, step):
                    if i < reduced_features.shape[0]:
                        # Shorten SMILES for annotation
                        short_smiles = smiles[i][:20] + '...' if len(smiles[i]) > 20 else smiles[i]
                        ax.annotate(
                            short_smiles,
                            xy=(reduced_features[i, 0], reduced_features[i, 1]),
                            xytext=(5, 5),
                            textcoords='offset points',
                            fontsize=8
                        )
            
            # Set labels and title
            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Dimension 1', fontsize=14)
            ax.set_ylabel('Dimension 2', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            return fig
        except Exception as e:
            logger.error(f"Error plotting chemical space: {str(e)}")
            return plt.figure()
    
    @staticmethod
    def identify_clusters(
        reduced_features: np.ndarray,
        n_clusters: int = 5,
        method: str = 'kmeans',
        random_state: int = 42
    ) -> np.ndarray:
        """
        Identify clusters in chemical space.
        
        Args:
            reduced_features (np.ndarray): Reduced feature matrix
            n_clusters (int): Number of clusters
            method (str): Clustering method ('kmeans', 'dbscan', 'hierarchical')
            random_state (int): Random state for reproducibility
            
        Returns:
            np.ndarray: Cluster labels
        """
        if reduced_features is None or reduced_features.size == 0:
            return np.array([])
        
        try:
            # Apply clustering
            if method.lower() == 'kmeans':
                from sklearn.cluster import KMeans
                model = KMeans(n_clusters=n_clusters, random_state=random_state)
                labels = model.fit_predict(reduced_features)
            elif method.lower() == 'dbscan':
                from sklearn.cluster import DBSCAN
                model = DBSCAN(eps=0.5, min_samples=5)
                labels = model.fit_predict(reduced_features)
            elif method.lower() == 'hierarchical':
                from sklearn.cluster import AgglomerativeClustering
                model = AgglomerativeClustering(n_clusters=n_clusters)
                labels = model.fit_predict(reduced_features)
            else:
                logger.warning(f"Unknown method: {method}. Using KMeans.")
                from sklearn.cluster import KMeans
                model = KMeans(n_clusters=n_clusters, random_state=random_state)
                labels = model.fit_predict(reduced_features)
            
            return labels
        except Exception as e:
            logger.error(f"Error in clustering: {str(e)}")
            return np.zeros(reduced_features.shape[0])
    
    @staticmethod
    def analyze_cluster_properties(
        df: pd.DataFrame, 
        cluster_labels: np.ndarray, 
        property_cols: List[str]
    ) -> pd.DataFrame:
        """
        Analyze properties of molecules in each cluster.
        
        Args:
            df (pd.DataFrame): Dataframe with molecular properties
            cluster_labels (np.ndarray): Cluster labels
            property_cols (List[str]): Property column names
            
        Returns:
            pd.DataFrame: Cluster property analysis
        """
        if df is None or cluster_labels is None or not property_cols:
            return pd.DataFrame()
        
        try:
            # Add cluster labels to dataframe
            df_copy = df.copy()
            df_copy['Cluster'] = cluster_labels
            
            # Initialize results dataframe
            results = []
            
            # Calculate statistics for each cluster
            for cluster_id in sorted(df_copy['Cluster'].unique()):
                cluster_df = df_copy[df_copy['Cluster'] == cluster_id]
                cluster_size = len(cluster_df)
                
                cluster_stats = {
                    'Cluster_ID': cluster_id,
                    'Size': cluster_size,
                    'Percentage': round(100 * cluster_size / len(df_copy), 2)
                }
                
                # Calculate property statistics
                for col in property_cols:
                    if col in df_copy.columns:
                        if pd.api.types.is_numeric_dtype(df_copy[col]):
                            cluster_stats[f'{col}_Mean'] = cluster_df[col].mean()
                            cluster_stats[f'{col}_Std'] = cluster_df[col].std()
                            cluster_stats[f'{col}_Min'] = cluster_df[col].min()
                            cluster_stats[f'{col}_Max'] = cluster_df[col].max()
                        else:
                            # For categorical properties, calculate most common value
                            value_counts = cluster_df[col].value_counts()
                            if not value_counts.empty:
                                cluster_stats[f'{col}_MostCommon'] = value_counts.index[0]
                                cluster_stats[f'{col}_MostCommonPercentage'] = round(100 * value_counts.iloc[0] / cluster_size, 2)
                
                results.append(cluster_stats)
            
            return pd.DataFrame(results)
        except Exception as e:
            logger.error(f"Error analyzing cluster properties: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def visualize_cluster_representatives(
        df: pd.DataFrame, 
        cluster_labels: np.ndarray, 
        smiles_col: str = 'smiles',
        n_per_cluster: int = 3
    ) -> plt.Figure:
        """
        Visualize representative molecules from each cluster.
        
        Args:
            df (pd.DataFrame): Dataframe with molecules
            cluster_labels (np.ndarray): Cluster labels
            smiles_col (str): Column containing SMILES strings
            n_per_cluster (int): Number of representatives per cluster
            
        Returns:
            plt.Figure: Figure with molecular structures
        """
        if df is None or cluster_labels is None or smiles_col not in df.columns:
            return plt.figure()
        
        try:
            # Add cluster labels to dataframe
            df_copy = df.copy()
            df_copy['Cluster'] = cluster_labels
            
            # Get unique clusters
            unique_clusters = sorted(df_copy['Cluster'].unique())
            
            # Initialize lists for molecules and legends
            mols = []
            legends = []
            
            # Get representatives from each cluster
            for cluster_id in unique_clusters:
                cluster_df = df_copy[df_copy['Cluster'] == cluster_id]
                
                # Take a sample of molecules from the cluster
                if len(cluster_df) > n_per_cluster:
                    sample_df = cluster_df.sample(n_per_cluster)
                else:
                    sample_df = cluster_df
                
                # Convert SMILES to molecules
                for _, row in sample_df.iterrows():
                    smiles = row[smiles_col]
                    mol = Chem.MolFromSmiles(smiles)
                    
                    if mol is not None:
                        mols.append(mol)
                        legends.append(f"Cluster {cluster_id}")
            
            # Create molecule grid
            if mols:
                img = Draw.MolsToGridImage(
                    mols,
                    molsPerRow=n_per_cluster,
                    subImgSize=(200, 200),
                    legends=legends
                )
                
                # Convert to matplotlib figure
                fig, ax = plt.subplots(figsize=(15, 10))
                ax.imshow(img)
                ax.axis('off')
                plt.tight_layout()
                
                return fig
            else:
                logger.warning("No valid molecules for visualization")
                return plt.figure()
        except Exception as e:
            logger.error(f"Error visualizing cluster representatives: {str(e)}")
            return plt.figure()
    
    @staticmethod
    def calculate_diversity_metrics(similarity_matrix: np.ndarray) -> Dict[str, float]:
        """
        Calculate diversity metrics from similarity matrix.
        
        Args:
            similarity_matrix (np.ndarray): Similarity matrix
            
        Returns:
            Dict[str, float]: Diversity metrics
        """
        if similarity_matrix is None or similarity_matrix.size == 0:
            return {}
        
        try:
            # Extract upper triangle (excluding diagonal)
            triu_indices = np.triu_indices_from(similarity_matrix, k=1)
            similarities = similarity_matrix[triu_indices]
            
            # Calculate metrics
            avg_similarity = np.mean(similarities)
            median_similarity = np.median(similarities)
            min_similarity = np.min(similarities)
            max_similarity = np.max(similarities)
            
            # Calculate diversity
            diversity = 1.0 - avg_similarity
            
            return {
                'average_similarity': float(avg_similarity),
                'median_similarity': float(median_similarity),
                'min_similarity': float(min_similarity),
                'max_similarity': float(max_similarity),
                'diversity_score': float(diversity)
            }
        except Exception as e:
            logger.error(f"Error calculating diversity metrics: {str(e)}")
            return {}
    
    @staticmethod
    def analyze_scaffold_diversity(
        df: pd.DataFrame, 
        smiles_col: str = 'smiles'
    ) -> Dict[str, Any]:
        """
        Analyze scaffold diversity of a molecule set.
        
        Args:
            df (pd.DataFrame): Dataframe with molecules
            smiles_col (str): Column containing SMILES strings
            
        Returns:
            Dict[str, Any]: Scaffold analysis results
        """
        if df is None or smiles_col not in df.columns:
            return {}
        
        try:
            from rdkit.Chem.Scaffolds import MurckoScaffold
            
            # Calculate scaffolds
            scaffolds = []
            for smiles in df[smiles_col]:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=True)
                    scaffolds.append(scaffold_smiles)
                else:
                    scaffolds.append(None)
            
            # Filter valid scaffolds
            valid_scaffolds = [s for s in scaffolds if s is not None]
            
            if not valid_scaffolds:
                return {'scaffold_count': 0}
            
            # Count unique scaffolds
            unique_scaffolds = set(valid_scaffolds)
            scaffold_count = len(unique_scaffolds)
            
            # Calculate scaffold diversity
            scaffold_diversity = scaffold_count / len(valid_scaffolds)
            
            # Calculate scaffold frequencies
            scaffold_freq = {}
            for scaffold in valid_scaffolds:
                if scaffold in scaffold_freq:
                    scaffold_freq[scaffold] += 1
                else:
                    scaffold_freq[scaffold] = 1
            
            # Sort by frequency
            sorted_scaffolds = sorted(scaffold_freq.items(), key=lambda x: x[1], reverse=True)
            
            # Calculate top scaffolds
            top_scaffolds = []
            for scaffold, count in sorted_scaffolds[:10]:  # Top 10
                mol = Chem.MolFromSmiles(scaffold)
                top_scaffolds.append({
                    'scaffold': scaffold,
                    'count': count,
                    'percentage': round(100 * count / len(valid_scaffolds), 2)
                })
            
            return {
                'total_molecules': len(df),
                'valid_molecules': len(valid_scaffolds),
                'unique_scaffold_count': scaffold_count,
                'scaffold_diversity': scaffold_diversity,
                'top_scaffolds': top_scaffolds
            }
        except Exception as e:
            logger.error(f"Error analyzing scaffold diversity: {str(e)}")
            return {}
    
    @staticmethod
    def evaluate_coverage(
        train_features: np.ndarray, 
        test_features: np.ndarray, 
        n_neighbors: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate chemical space coverage of test set compared to training set.
        
        Args:
            train_features (np.ndarray): Training set features
            test_features (np.ndarray): Test set features
            n_neighbors (int): Number of nearest neighbors
            
        Returns:
            Dict[str, float]: Coverage metrics
        """
        if train_features is None or test_features is None:
            return {}
        
        try:
            from sklearn.neighbors import NearestNeighbors
            
            # Initialize nearest neighbors model
            nn_model = NearestNeighbors(n_neighbors=n_neighbors)
            nn_model.fit(train_features)
            
            # Find distances to nearest neighbors
            distances, _ = nn_model.kneighbors(test_features)
            
            # Calculate average distance
            avg_distance = np.mean(distances)
            
            # Calculate distance statistics
            min_distance = np.min(distances)
            max_distance = np.max(distances)
            median_distance = np.median(distances)
            
            # Calculate coverage score (inverse of average distance)
            if avg_distance > 0:
                coverage_score = 1.0 / (1.0 + avg_distance)
            else:
                coverage_score = 1.0
            
            return {
                'average_distance': float(avg_distance),
                'min_distance': float(min_distance),
                'max_distance': float(max_distance),
                'median_distance': float(median_distance),
                'coverage_score': float(coverage_score)
            }
        except Exception as e:
            logger.error(f"Error evaluating coverage: {str(e)}")
            return {}
    
    @staticmethod
    def identify_outliers(
        features: np.ndarray, 
        method: str = 'isolation_forest',
        contamination: float = 0.05,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Identify outlier molecules in chemical space.
        
        Args:
            features (np.ndarray): Feature matrix
            method (str): Outlier detection method ('isolation_forest', 'lof')
            contamination (float): Expected proportion of outliers
            random_state (int): Random state for reproducibility
            
        Returns:
            np.ndarray: Boolean array with True for outliers
        """
        if features is None or features.size == 0:
            return np.array([])
        
        try:
            # Apply outlier detection
            if method.lower() == 'isolation_forest':
                from sklearn.ensemble import IsolationForest
                model = IsolationForest(
                    contamination=contamination,
                    random_state=random_state
                )
                # IsolationForest returns -1 for outliers and 1 for inliers
                outliers = model.fit_predict(features) == -1
            elif method.lower() == 'lof':
                from sklearn.neighbors import LocalOutlierFactor
                model = LocalOutlierFactor(
                    n_neighbors=20,
                    contamination=contamination
                )
                # LocalOutlierFactor returns -1 for outliers and 1 for inliers
                outliers = model.fit_predict(features) == -1
            else:
                logger.warning(f"Unknown method: {method}. Using IsolationForest.")
                from sklearn.ensemble import IsolationForest
                model = IsolationForest(
                    contamination=contamination,
                    random_state=random_state
                )
                outliers = model.fit_predict(features) == -1
            
            return outliers
        except Exception as e:
            logger.error(f"Error identifying outliers: {str(e)}")
            return np.zeros(features.shape[0], dtype=bool)
    
    @staticmethod
    def plot_property_distribution(
        df: pd.DataFrame, 
        property_col: str, 
        figsize: Tuple[int, int] = (10, 6),
        bins: int = 30
    ) -> plt.Figure:
        """
        Plot distribution of a molecular property.
        
        Args:
            df (pd.DataFrame): Dataframe with molecules
            property_col (str): Property column name
            figsize (Tuple[int, int]): Figure size
            bins (int): Number of histogram bins
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        if df is None or property_col not in df.columns:
            return plt.figure()
        
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Check if property is numeric
            if pd.api.types.is_numeric_dtype(df[property_col]):
                # Create histogram with kernel density estimate
                sns.histplot(df[property_col].dropna(), bins=bins, kde=True, ax=ax)
                
                # Add vertical lines for mean and median
                mean_val = df[property_col].mean()
                median_val = df[property_col].median()
                
                ax.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='g', linestyle='--', label=f'Median: {median_val:.2f}')
                
                ax.legend()
            else:
                # For categorical properties, create bar plot
                value_counts = df[property_col].value_counts().sort_values(ascending=False)
                sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
                
                # Rotate x-labels for better readability
                plt.xticks(rotation=45, ha='right')
            
            # Set labels and title
            ax.set_title(f'Distribution of {property_col}', fontsize=16)
            ax.set_xlabel(property_col, fontsize=14)
            ax.set_ylabel('Frequency', fontsize=14)
            
            plt.tight_layout()
            
            return fig
        except Exception as e:
            logger.error(f"Error plotting property distribution: {str(e)}")
            return plt.figure()