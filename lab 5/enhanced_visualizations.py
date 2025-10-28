"""
Enhanced visualization methods for Earth surface image clustering
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

def create_2d_scatter_plots(self, cluster_labels, features_scaled):
    """Create 2D scatter plots with PCA"""
    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    features_2d_pca = pca.fit_transform(features_scaled)
    
    # Create a figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Earth Surface Image Clustering - 2D Projection (PCA)', fontsize=16)
    
    # Plot 1: Clustering results
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        axes[0].scatter(features_2d_pca[mask, 0], features_2d_pca[mask, 1], 
                      c=[colors[i]], label=f'Cluster {label}', alpha=0.7, s=60, edgecolors='w')
    
    axes[0].set_title('Clustering Results')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[0].legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: True classes
    class_names = list(self.label_map.keys())
    unique_classes = np.unique(self.labels)
    class_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))
    
    for i, label in enumerate(unique_classes):
        mask = self.labels == label
        axes[1].scatter(features_2d_pca[mask, 0], features_2d_pca[mask, 1], 
                      c=[class_colors[i]], label=class_names[label], alpha=0.7, s=60, edgecolors='w')
    
    axes[1].set_title('True Classes')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[1].legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('clustering_2d_pca.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Try to use t-SNE if available
    try:
        from sklearn.manifold import TSNE
        print("\nComputing t-SNE projection (this may take a while)...")
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        features_2d_tsne = tsne.fit_transform(features_scaled)
        
        # Create a figure with 2 subplots for t-SNE
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Earth Surface Image Clustering - 2D Projection (t-SNE)', fontsize=16)
        
        # Plot 1: Clustering results with t-SNE
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            axes[0].scatter(features_2d_tsne[mask, 0], features_2d_tsne[mask, 1], 
                          c=[colors[i]], label=f'Cluster {label}', alpha=0.7, s=60, edgecolors='w')
        
        axes[0].set_title('Clustering Results (t-SNE)')
        axes[0].set_xlabel('t-SNE Component 1')
        axes[0].set_ylabel('t-SNE Component 2')
        axes[0].legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        axes[0].grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: True classes with t-SNE
        for i, label in enumerate(unique_classes):
            mask = self.labels == label
            axes[1].scatter(features_2d_tsne[mask, 0], features_2d_tsne[mask, 1], 
                          c=[class_colors[i]], label=class_names[label], alpha=0.7, s=60, edgecolors='w')
        
        axes[1].set_title('True Classes (t-SNE)')
        axes[1].set_xlabel('t-SNE Component 1')
        axes[1].set_ylabel('t-SNE Component 2')
        axes[1].legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        axes[1].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('clustering_2d_tsne.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except (ImportError, Exception) as e:
        print(f"Could not create t-SNE visualization: {e}")

def create_3d_scatter_plot(self, cluster_labels, features_scaled):
    """Create 3D scatter plot for better cluster visualization"""
    try:
        # PCA for 3D visualization
        pca = PCA(n_components=3)
        features_3d = pca.fit_transform(features_scaled)
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot clusters
        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            ax.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                      c=[colors[i]], label=f'Cluster {label}', alpha=0.7, s=60, edgecolors='w')
        
        # Set labels and title
        ax.set_title('3D Visualization of Clusters', fontsize=14)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)')
        
        # Add legend and grid
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Improve 3D view
        ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        plt.savefig('clustering_3d.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Could not create 3D visualization: {e}")

def create_cluster_analysis_plots(self, cluster_labels):
    """Create plots for cluster analysis"""
    # Create a figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Cluster Analysis', fontsize=16)
    
    # Plot 1: Cluster distribution
    cluster_counts = Counter(cluster_labels)
    cluster_ids = sorted(cluster_counts.keys())
    cluster_sizes = [cluster_counts[id] for id in cluster_ids]
    
    bars = axes[0].bar(cluster_ids, cluster_sizes, 
                     color=plt.cm.tab20(np.linspace(0, 1, len(cluster_ids))))
    axes[0].set_title('Image Distribution Across Clusters', fontsize=14)
    axes[0].set_xlabel('Cluster ID', fontsize=12)
    axes[0].set_ylabel('Number of Images', fontsize=12)
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bar, size in zip(bars, cluster_sizes):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(size), ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Cluster purity
    cluster_purity = {}
    cluster_dominant_class = {}
    
    for cluster_id in sorted(set(cluster_labels)):
        cluster_mask = cluster_labels == cluster_id
        cluster_true_labels = self.labels[cluster_mask]
        
        if len(cluster_true_labels) > 0:
            most_common_class = Counter(cluster_true_labels).most_common(1)[0]
            purity = most_common_class[1] / len(cluster_true_labels)
            class_name = list(self.label_map.keys())[most_common_class[0]]
            
            cluster_purity[cluster_id] = purity
            cluster_dominant_class[cluster_id] = class_name
    
    purity_clusters = sorted(cluster_purity.keys())
    purity_values = [cluster_purity[id] for id in purity_clusters]
    
    # Create color gradient based on purity
    purity_colors = plt.cm.RdYlGn(np.array(purity_values))
    
    bars2 = axes[1].bar(purity_clusters, purity_values, color=purity_colors)
    axes[1].set_title('Cluster Purity', fontsize=14)
    axes[1].set_xlabel('Cluster ID', fontsize=12)
    axes[1].set_ylabel('Purity (%)', fontsize=12)
    axes[1].set_ylim(0, 1.1)  # Leave room for labels
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value and class labels on bars
    for bar, cluster_id, purity in zip(bars2, purity_clusters, purity_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, purity + 0.02, 
                    f'{purity:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Add dominant class name below the bar
        dominant_class = cluster_dominant_class[cluster_id]
        axes[1].text(bar.get_x() + bar.get_width()/2, -0.05, 
                    f'{dominant_class}', ha='center', va='top', 
                    rotation=45, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('cluster_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_confusion_matrix(self, cluster_labels):
    """Create a confusion matrix between clusters and true classes"""
    # Get unique clusters and classes
    unique_clusters = sorted(set(cluster_labels))
    unique_classes = sorted(set(self.labels))
    class_names = list(self.label_map.keys())
    
    # Create confusion matrix
    confusion_matrix = np.zeros((len(unique_clusters), len(unique_classes)))
    
    for i, cluster in enumerate(unique_clusters):
        cluster_mask = cluster_labels == cluster
        for j, class_id in enumerate(unique_classes):
            confusion_matrix[i, j] = np.sum((self.labels == class_id) & cluster_mask)
    
    # Plot confusion matrix
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    
    # Create heatmap
    im = ax.imshow(confusion_matrix, cmap='Blues')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Images', rotation=270, labelpad=15)
    
    # Add labels and title
    ax.set_xticks(np.arange(len(unique_classes)))
    ax.set_yticks(np.arange(len(unique_clusters)))
    ax.set_xticklabels([class_names[class_id] for class_id in unique_classes])
    ax.set_yticklabels([f'Cluster {cluster}' for cluster in unique_clusters])
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add title and labels
    ax.set_title('Confusion Matrix: Clusters vs. True Classes', fontsize=14)
    ax.set_ylabel('Clusters', fontsize=12)
    ax.set_xlabel('True Classes', fontsize=12)
    
    # Add text annotations
    for i in range(len(unique_clusters)):
        for j in range(len(unique_classes)):
            count = confusion_matrix[i, j]
            if count > 0:
                text_color = 'white' if count > confusion_matrix.max() / 2 else 'black'
                ax.text(j, i, f'{int(count)}', ha="center", va="center", 
                       color=text_color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
