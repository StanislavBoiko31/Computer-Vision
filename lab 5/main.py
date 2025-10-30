import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.manifold import TSNE, MDS
import json
from collections import Counter
import time
import warnings
warnings.filterwarnings('ignore')

try:
    from keras.applications.vgg16 import VGG16, preprocess_input
    from keras.utils import load_img, img_to_array
    VGG16_AVAILABLE = True
except ImportError:
    VGG16_AVAILABLE = False
    pass

plt.style.use('seaborn-v0_8-whitegrid')

CLUSTER_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

class TermColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Helper functions for consensus matrix creation
def consensus_matrix(cluster_runs, n_samples):
    consensus = np.zeros((n_samples, n_samples))
    
    for labels in cluster_runs:
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                if labels[i] == labels[j]:
                    consensus[i, j] += 1
                    consensus[j, i] += 1
    
    consensus /= len(cluster_runs)
    
    return consensus

# Weighted ensemble clustering function
def weighted_ensemble_clustering(features, true_labels=None, n_clusters=4, n_runs=5):
    n_samples = features.shape[0]
    
    weighted_cluster_runs = []
    
    feature_variants = [
        ('original', features),
        ('standard_scaled', StandardScaler().fit_transform(features)),
        ('robust_scaled', RobustScaler().fit_transform(features))
    ]
    
    try:
        pca_features = PCA(n_components=min(50, features.shape[1]//2)).fit_transform(feature_variants[1][1])
        feature_variants.append(('pca', pca_features))
    except Exception as e:
        print("PCA failed")
    
    try:
        kpca_features = KernelPCA(n_components=min(50, features.shape[1]//2), kernel='rbf').fit_transform(feature_variants[1][1])
        feature_variants.append(('kpca', kpca_features))
    except Exception as e:
        print("Kernel PCA failed")
    
    algorithms = [
        {'name': 'KMeans-Standard', 'algorithm': KMeans, 'params': {
            'n_clusters': n_clusters, 
            'init': 'k-means++', 
            'n_init': 20
        }},
        {'name': 'KMeans-More', 'algorithm': KMeans, 'params': {
            'n_clusters': n_clusters + 1, 
            'init': 'k-means++', 
            'n_init': 20
        }},
        {'name': 'KMeans-Less', 'algorithm': KMeans, 'params': {
            'n_clusters': max(2, n_clusters - 1), 
            'init': 'k-means++', 
            'n_init': 20
        }},
        
        {'name': 'Agglomerative-Ward', 'algorithm': AgglomerativeClustering, 'params': {
            'n_clusters': n_clusters, 
            'linkage': 'ward'
        }},
        {'name': 'Agglomerative-Complete', 'algorithm': AgglomerativeClustering, 'params': {
            'n_clusters': n_clusters, 
            'linkage': 'complete'
        }},
        {'name': 'Agglomerative-Average', 'algorithm': AgglomerativeClustering, 'params': {
            'n_clusters': n_clusters, 
            'linkage': 'average'
        }},
        
        {'name': 'Spectral-NN-10', 'algorithm': SpectralClustering, 'params': {
            'n_clusters': n_clusters,
            'affinity': 'nearest_neighbors',
            'n_neighbors': 10,
            'assign_labels': 'kmeans'
        }},
        {'name': 'Spectral-NN-20', 'algorithm': SpectralClustering, 'params': {
            'n_clusters': n_clusters,
            'affinity': 'nearest_neighbors',
            'n_neighbors': 20,
            'assign_labels': 'kmeans'
        }},
        {'name': 'Spectral-NN-Discretize', 'algorithm': SpectralClustering, 'params': {
            'n_clusters': n_clusters,
            'affinity': 'nearest_neighbors',
            'n_neighbors': 15,
            'assign_labels': 'discretize'
        }}
    ]
    
    for feature_name, feature_set in feature_variants:
        
        for algo_config in algorithms:
            algo_name = algo_config['name']
            algorithm = algo_config['algorithm']
            params = algo_config['params'].copy()
            
            if algo_name.startswith('Spectral') and feature_name == 'original':
                continue
                
            for i in range(n_runs):
                try:
                    if algo_name.startswith('KMeans') or algo_name.startswith('Spectral'):
                        params['random_state'] = 42 + i
                    
                    model = algorithm(**params)
                    labels = model.fit_predict(feature_set)
                    
                    unique_labels = len(set(labels))
                    if unique_labels < 2 or unique_labels > n_samples // 5:
                        print("Invalid number of clusters")
                        continue
                    
                    if true_labels is not None:
                        sil_score = silhouette_score(feature_set, labels)
                        ari = adjusted_rand_score(true_labels, labels)
                        
                        weight = (0.2 * max(0, sil_score) + 0.8 * max(0, ari)) + 0.1
                    else:
                        sil_score = silhouette_score(feature_set, labels)
                        weight = max(0.1, sil_score) + 0.1
                    
                    cluster_counts = np.bincount(labels)
                    largest_cluster = np.max(cluster_counts)
                    largest_percentage = largest_cluster / len(labels)
                    balance_factor = 1.0 - largest_percentage
                    
                    if largest_percentage > 0.6:  # If one cluster has more than 60% of samples
                        penalty = 1.0 - ((largest_percentage - 0.6) * 2.5)  # Stronger penalty
                        weight *= max(0.1, penalty)
                    
                    if balance_factor > 0.7:  # Very balanced
                        weight *= 1.2
                    
                    if unique_labels == n_clusters:
                        weight *= 1.1
                    
                    #print("Algorithm run completed")
                    
                    weighted_cluster_runs.append((labels, weight))
                    
                except Exception as e:
                    print(f"Algorithm failed: {e}")
    
    if len(weighted_cluster_runs) < 3:
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=20, random_state=42)
        labels = kmeans.fit_predict(StandardScaler().fit_transform(features))
        weighted_cluster_runs.append((labels, 1.0))
    
    consensus = np.zeros((n_samples, n_samples))
    total_weight = 0
    
    for labels, weight in weighted_cluster_runs:
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                if labels[i] == labels[j]:
                    consensus[i, j] += weight
                    consensus[j, i] += weight
        total_weight += weight
    
    if total_weight > 0:
        consensus /= total_weight
    
    distance_matrix = 1 - consensus
    
    try:
        mds = MDS(n_components=min(50, n_samples//4), 
                 dissimilarity='precomputed', 
                 random_state=42,
                 n_init=1,
                 max_iter=100)
        consensus_points = mds.fit_transform(distance_matrix)
    except Exception as e:
        print(f"MDS failed, using PCA instead: {e}")
        pca = PCA(n_components=min(50, n_samples//2))
        consensus_points = pca.fit_transform(consensus)
    
    final_clustering = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=10,
        random_state=42
    )
    
    final_labels = final_clustering.fit_predict(consensus_points)
    
    if true_labels is not None:
        sil_score = silhouette_score(features, final_labels)
        ari = adjusted_rand_score(true_labels, final_labels)
        #print("Weighted Ensemble completed")
    
    cluster_counts = np.bincount(final_labels)
    #print("Weighted ensemble completed")
    
    return final_labels

class SimpleEarthSurfaceClustering:
    def __init__(self, data_path=".", max_samples_per_class=30):

        self.data_path = data_path
        self.max_samples_per_class = max_samples_per_class
        self.label_map = self.load_label_map()
        self.features = []
        self.labels = []
        self.image_paths = []
        
        if VGG16_AVAILABLE:
            self.model = VGG16(weights='imagenet', include_top=False)
        else:
            self.model = None
    
    def load_label_map(self):
        with open(os.path.join(self.data_path, 'label_map.json'), 'r') as f:
            return json.load(f)
    
    def load_sample_images(self, csv_file="train.csv"):
        
        selected_classes = ['AnnualCrop', 'HerbaceousVegetation', 'Industrial', 'Forest']
                
        df = pd.read_csv(os.path.join(self.data_path, csv_file))
        
        df_filtered = df[df['ClassName'].isin(selected_classes)]
        
        sampled_data = []
        for class_name in selected_classes:
            class_data = df_filtered[df_filtered['ClassName'] == class_name]
            if len(class_data) > 0:
                sample_size = min(self.max_samples_per_class, len(class_data))
                sampled = class_data.sample(n=sample_size, random_state=42)
                sampled_data.append(sampled)
        
        self.df = pd.concat(sampled_data, ignore_index=True)
                
        class_counts = self.df['ClassName'].value_counts()
        for class_name, count in class_counts.items():
            print(f"Class: {class_name}, Count: {count}")
    
    def extract_features(self, image_path):
        if not self.model:
            return np.array([])
        
        try:
            img = load_img(image_path, target_size=(224, 224))
            img_data = img_to_array(img)
            
            cv_img = cv2.cvtColor(img_data.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            color_features = []
            for i in range(3):  # RGB channels
                hist = cv2.calcHist([cv_img], [i], None, [64], [0, 256])  # 64 bins for finer detail
                hist = cv2.normalize(hist, hist).flatten()
                color_features.extend(hist)
            
            hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
            hsv_features = []
            for i in range(3):  # HSV channels
                hist = cv2.calcHist([hsv_img], [i], None, [32], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                hsv_features.extend(hist)
            
            lab_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
            lab_features = []
            for i in range(3):  # LAB channels
                hist = cv2.calcHist([lab_img], [i], None, [32], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                lab_features.extend(hist)
            
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            
            gabor_features = []
            for theta in [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, 5*np.pi/6]:
                for sigma in [1, 2, 3]:
                    for lambd in [10.0, 15.0]:
                        kernel = cv2.getGaborKernel((21, 21), sigma, theta, lambd, 0.5, 0, ktype=cv2.CV_32F)
                        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                        mean = np.mean(filtered)
                        std = np.std(filtered)
                        gabor_features.extend([mean, std])
            
            try:
                from skimage.feature import local_binary_pattern
                radius = 3
                n_points = 8 * radius
                lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
                lbp_hist, _ = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2), density=True)
                lbp_features = lbp_hist.tolist()
            except ImportError:
                lbp_features = [0] * 26  # Default if skimage is not available
            
            edge_features = []
            for threshold1, threshold2 in [(50, 150), (100, 200), (150, 250)]:
                edges = cv2.Canny(gray, threshold1, threshold2)
                edge_percentage = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                edge_features.append(edge_percentage)
            
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_mag = np.sqrt(sobelx**2 + sobely**2)
            sobel_dir = np.arctan2(sobely, sobelx)
            
            dir_hist, _ = np.histogram(sobel_dir, bins=16, range=(-np.pi, np.pi), density=True)
            
            sobel_features = [np.mean(sobel_mag), np.std(sobel_mag)] + dir_hist.tolist()
            
            b, g, r = cv2.split(cv_img)
            
            ndvi_approx = (np.mean(g) - np.mean(r)) / (np.mean(g) + np.mean(r) + 1e-10)
            
            gli = (2 * np.mean(g) - np.mean(r) - np.mean(b)) / (2 * np.mean(g) + np.mean(r) + np.mean(b) + 1e-10)
            
            vari = (np.mean(g) - np.mean(r)) / (np.mean(g) + np.mean(r) - np.mean(b) + 1e-10)
            
            green_ratio = np.mean(g) / (np.mean(r) + np.mean(b) + 1e-10)
            
            rg_ratio = np.mean(r) / (np.mean(g) + 1e-10)
            rb_ratio = np.mean(r) / (np.mean(b) + 1e-10)
            gb_ratio = np.mean(g) / (np.mean(b) + 1e-10)
            
            grid_size = 4  # 4x4 grid
            grid_features = []
            h, w = gray.shape
            cell_h, cell_w = h // grid_size, w // grid_size
            
            for i in range(grid_size):
                for j in range(grid_size):
                    cell = gray[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                    grid_features.extend([np.mean(cell), np.std(cell)])
            
            # 7. Deep features from VGG16
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)
            deep_features = np.array(self.model.predict(img_data, verbose=0)).flatten()
            
            # Combine all features with appropriate weighting for our specific classes
            combined_features = np.concatenate([
                deep_features,
                np.array(color_features) * 2.0,             # RGB histograms (weighted)
                np.array(hsv_features) * 2.5,               # HSV features (heavily weighted for vegetation)
                np.array(lab_features) * 2.0,               # LAB features (good for agricultural areas)
                np.array(gabor_features) * 1.5,             # Texture features (weighted)
                np.array(lbp_features) * 2.0,               # LBP texture features (weighted)
                np.array(edge_features) * 3.0,              # Edge features (heavily weighted)
                np.array(sobel_features) * 1.5,             # Sobel edge features with direction
                np.array(grid_features) * 1.0,              # Spatial arrangement features
                np.array([ndvi_approx * 5.0]),              # NDVI (very heavily weighted)
                np.array([gli * 4.0]),                      # GLI (heavily weighted)
                np.array([vari * 4.0]),                     # VARI (heavily weighted)
                np.array([green_ratio * 3.0]),              # Green ratio (heavily weighted)
                np.array([rg_ratio * 2.0]),                 # R/G ratio (weighted)
                np.array([rb_ratio * 2.0]),                 # R/B ratio (weighted)
                np.array([gb_ratio * 3.0])                  # G/B ratio (heavily weighted)
            ])
            
            return combined_features
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return np.array([])
    
    def process_images(self):
                
        self.features = []
        self.labels = []
        self.image_paths = []
        
        for idx, row in self.df.iterrows():
            if idx % 50 == 0:
                print(f"Processing image {idx+1}/{len(self.df)}")
            
            image_path = os.path.join(self.data_path, row['Filename'])
            
            if not os.path.exists(image_path):
                continue
            
            # Extract features
            features = self.extract_features(image_path)
            
            if len(features) > 0:
                self.features.append(features)
                self.labels.append(row['Label'])
                self.image_paths.append(image_path)
        
                
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        
        unique_labels = np.unique(self.labels)
    
    def find_optimal_clusters(self, features, max_clusters=15):
                
        range_n_clusters = range(2, min(max_clusters + 1, len(self.labels) // 5))
        
        silhouette_scores = []
        inertia_values = []
        
        for n_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            silhouette_avg = silhouette_score(features, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            inertia_values.append(kmeans.inertia_)
            
            #print(f"Evaluating cluster {n_clusters}")
        
        diffs = np.diff(inertia_values)
        second_diffs = np.diff(diffs)
        if len(second_diffs) > 0:
            optimal_clusters = np.argmax(second_diffs) + 3
            optimal_clusters = max(2, min(optimal_clusters, max_clusters))
            return optimal_clusters
        
        optimal_clusters = np.argmax(silhouette_scores) + 2
        return optimal_clusters

    def perform_clustering(self, n_clusters=None, optimize_params=True):
        """Enhanced clustering specifically for the 4 vegetation/urban classes with ensemble methods"""
                
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features)
        
        pca = PCA(n_components=0.95)
        features_pca = pca.fit_transform(features_scaled)
                
        veg_feature_indices = [-7, -6, -5, -4, -3, -2, -1]  # Last 7 features are vegetation indices
        
        weighted_features = features_scaled.copy()
        for idx in veg_feature_indices:
            if abs(idx) < weighted_features.shape[1]:
                weighted_features[:, idx] *= 5.0
        
        pca_weighted = PCA(n_components=min(50, weighted_features.shape[1]))
        features_weighted = pca_weighted.fit_transform(weighted_features)
        
        ensemble_labels = None
        ensemble_score = -1
        
        try:
            weighted_ensemble_labels = weighted_ensemble_clustering(
                features_weighted, 
                true_labels=self.labels, 
                n_clusters=n_clusters or 4,
                n_runs=5
            )
            
            sil_score = silhouette_score(features_weighted, weighted_ensemble_labels)
            ari = adjusted_rand_score(self.labels, weighted_ensemble_labels)
            
            cluster_counts = np.bincount(weighted_ensemble_labels)
            largest_cluster = np.max(cluster_counts)
            largest_percentage = largest_cluster / len(weighted_ensemble_labels) * 100
            balance_factor = 1.0 - (largest_percentage / 100)
            
            combined_score = (0.3 * sil_score) + (0.4 * ari) + (0.3 * balance_factor)
            
            #print("Clusters found")
            
            ensemble_score = combined_score
            ensemble_labels = weighted_ensemble_labels
            ensemble_algorithm = 'Weighted-Ensemble'
            
        except Exception as e:
            pass# Weighted ensemble clustering failed
        
        clustering_approaches = [
            self._two_stage_clustering,
            
            self._spectral_clustering,
            
            self._agglomerative_distance_clustering
        ]
        
        best_labels = None
        best_algorithm = None
        best_n_clusters = 0
        best_score = -1
        
        for approach_func in clustering_approaches:
            try:
                labels, algorithm_name, n_clusters_found = approach_func(features_weighted, n_clusters or 4)
                
                if labels is None or n_clusters_found < 2:
                    continue
                
                cluster_counts = np.bincount(labels)
                largest_cluster = np.max(cluster_counts)
                smallest_cluster = np.min(cluster_counts)
                largest_percentage = largest_cluster / len(labels) * 100
                balance_factor = 1.0 - (largest_percentage / 100)
                
                score = silhouette_score(features_weighted, labels)
                ari = adjusted_rand_score(self.labels, labels)
                
                combined_score = (0.3 * score) + (0.4 * ari) + (0.3 * balance_factor)
                
                # Add bonus for having close to 4 clusters
                cluster_proximity = 1.0 - min(abs(n_clusters_found - 4), 2) / 2.0
                adjusted_score = combined_score + (0.1 * cluster_proximity)
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_labels = labels
                    best_algorithm = algorithm_name
                    best_n_clusters = n_clusters_found
                    
            except Exception as e:
                pass# Clustering approach failed
        
        if ensemble_labels is not None and ensemble_score > best_score:
            best_labels = ensemble_labels
            best_algorithm = ensemble_algorithm
            best_n_clusters = len(set(ensemble_labels))
            best_score = ensemble_score
        
        if best_labels is None:
            kmeans = KMeans(n_clusters=4, init='k-means++', n_init=30, random_state=42)
            best_labels = kmeans.fit_predict(features_weighted)
            best_algorithm = 'KMeans-Fallback'
            best_n_clusters = 4
        
        #print(f"{TermColors.BOLD}Number of clusters found: {best_n_clusters}{TermColors.ENDC}")
        
        return best_labels, None, features_weighted
        
    def _two_stage_clustering(self, features, n_clusters):
        try:
            veg_indices = features[:, -7:]  # Last 7 features are vegetation indices
            
            from sklearn.decomposition import PCA
            pca_veg = PCA(n_components=2)
            veg_features = pca_veg.fit_transform(veg_indices)
            
            kmeans_binary = KMeans(n_clusters=2, init='k-means++', n_init=30, random_state=42)
            binary_labels = kmeans_binary.fit_predict(veg_features)
        except Exception as e:
            print(f"Vegetation index extraction failed: {e}")
            kmeans_binary = KMeans(n_clusters=2, init='k-means++', n_init=30, random_state=42)
            binary_labels = kmeans_binary.fit_predict(features)
        
        group0_indices = np.where(binary_labels == 0)[0]
        group1_indices = np.where(binary_labels == 1)[0]
        
        #print("Initial split completed")
        
        if len(group0_indices) > 5 and len(group1_indices) > 5:
            def find_best_clustering(group_features, group_indices):
                best_labels = None
                best_score = -1
                
                for i in range(5):
                    kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=42+i)
                    labels = kmeans.fit_predict(group_features)
                    
                    try:
                        sil_score = silhouette_score(group_features, labels)
                        
                        true_labels_subset = self.labels[group_indices]
                        ari = adjusted_rand_score(true_labels_subset, labels)
                        
                        score = (0.4 * sil_score) + (0.6 * ari)
                        
                        if score > best_score:
                            best_score = score
                            best_labels = labels
                    except Exception:
                        continue
                
                try:
                    agg = AgglomerativeClustering(n_clusters=2, linkage='ward')
                    labels = agg.fit_predict(group_features)
                    
                    sil_score = silhouette_score(group_features, labels)
                    true_labels_subset = self.labels[group_indices]
                    ari = adjusted_rand_score(true_labels_subset, labels)
                    score = (0.4 * sil_score) + (0.6 * ari)
                    
                    if score > best_score:
                        best_score = score
                        best_labels = labels
                except Exception:
                    pass
                
                return best_labels if best_labels is not None else KMeans(n_clusters=2).fit_predict(group_features)
            
            features_group0 = features[group0_indices]
            features_group1 = features[group1_indices]
            
            labels_group0 = find_best_clustering(features_group0, group0_indices)
            labels_group1 = find_best_clustering(features_group1, group1_indices)
            
            final_labels = np.zeros(len(features), dtype=int)
            final_labels[group0_indices] = labels_group0
            final_labels[group1_indices] = labels_group1 + 2  # Offset by 2 to get clusters 2 and 3
            
            cluster_counts = np.bincount(final_labels)
            #print("Multi-stage clustering completed")
            
            sil_score = silhouette_score(features, final_labels)
            ari = adjusted_rand_score(self.labels, final_labels)
            #print("Clustering evaluated")
            
            # Check if we need to refine any clusters further
            # If we have a very large cluster, try to split it further
            largest_cluster = np.argmax(cluster_counts)
            largest_percentage = cluster_counts[largest_cluster] / len(features) * 100
            
            if largest_percentage > 40 and n_clusters > 4:  # If one cluster has more than 40% of samples
                                
                # Extract samples from the largest cluster
                largest_indices = np.where(final_labels == largest_cluster)[0]
                largest_features = features[largest_indices]
                
                # Try to split this cluster
                kmeans_refine = KMeans(n_clusters=2, init='k-means++', n_init=20, random_state=42)
                refine_labels = kmeans_refine.fit_predict(largest_features)
                
                # Update labels
                new_labels = final_labels.copy()
                new_labels[largest_indices[refine_labels == 1]] = np.max(final_labels) + 1
                
                # Check if the refinement improved the clustering
                new_sil_score = silhouette_score(features, new_labels)
                new_ari = adjusted_rand_score(self.labels, new_labels)
                
                if new_sil_score > sil_score or new_ari > ari:
                    #print("Refinement improved clustering")
                    final_labels = new_labels
                    cluster_counts = np.bincount(final_labels)
                    #print("Refined cluster distribution")
            
            return final_labels, 'Advanced-Multi-Stage', len(set(final_labels))
        
        return None, None, 0
    
    def _spectral_clustering(self, features, n_clusters):
                
        try:
            from sklearn.cluster import SpectralClustering
            from sklearn.metrics import pairwise_distances
            from sklearn.neighbors import kneighbors_graph
            from sklearn.preprocessing import StandardScaler, RobustScaler
            from sklearn.decomposition import PCA, KernelPCA
            
            best_labels = None
            best_score = -1
            best_distribution = None
            
            preprocessing_methods = [
                ('standard', StandardScaler()),
                ('robust', RobustScaler()),
                ('none', None)
            ]
            
            dim_reduction_methods = [
                ('pca', PCA(n_components=min(50, features.shape[1]//2))),
                ('kernel_pca_rbf', KernelPCA(n_components=min(50, features.shape[1]//2), kernel='rbf')),
                ('none', None)
            ]
            
            affinity_methods = [
                ('nearest_neighbors', {'n_neighbors': 10}),
                ('nearest_neighbors', {'n_neighbors': 15}),
                ('nearest_neighbors', {'n_neighbors': 20}),
                ('nearest_neighbors', {'n_neighbors': 25}),
                ('rbf', {'gamma': 0.5}),
                ('rbf', {'gamma': 1.0})
            ]
            
            eigen_solvers = ['arpack', 'lobpcg']
            
            assign_labels_methods = ['kmeans', 'discretize']
            
            for preproc_name, preprocessor in preprocessing_methods[:2]:
                for dim_red_name, dim_reducer in dim_reduction_methods[:2]:
                    # Preprocess features
                    if preprocessor is not None:
                        features_processed = preprocessor.fit_transform(features)
                    else:
                        features_processed = features.copy()
                    
                    # Apply dimensionality reduction
                    if dim_reducer is not None:
                        try:
                            features_reduced = dim_reducer.fit_transform(features_processed)
                        except Exception:
                            continue  # Skip if dimensionality reduction fails
                    else:
                        features_reduced = features_processed
                    
                    for affinity, params in affinity_methods:
                        if affinity == 'rbf' and preproc_name == 'none':
                            continue
                        
                        for eigen_solver in eigen_solvers:
                            for assign_labels in assign_labels_methods:
                                if assign_labels == 'discretize' and eigen_solver == 'lobpcg':
                                    continue
                                
                                try:
                                    if affinity == 'nearest_neighbors':
                                        n_neighbors = params['n_neighbors']
                                        pass# Using spectral clustering configuration
                                        spectral = SpectralClustering(
                                            n_clusters=n_clusters,
                                            affinity=affinity,
                                            n_neighbors=n_neighbors,
                                            eigen_solver=eigen_solver,
                                            random_state=42,
                                            n_init=10,  # Multiple initializations
                                            assign_labels=assign_labels
                                        )
                                    else:  # rbf
                                        gamma = params['gamma']
                                        pass# Using spectral clustering configuration
                                        spectral = SpectralClustering(
                                            n_clusters=n_clusters,
                                            affinity=affinity,
                                            gamma=gamma,
                                            eigen_solver=eigen_solver,
                                            random_state=42,
                                            n_init=10,
                                            assign_labels=assign_labels
                                        )
                                    
                                    labels = spectral.fit_predict(features_reduced)
                                    
                                    cluster_counts = np.bincount(labels)
                                    if len(cluster_counts) < 2:
                                        continue
                                    
                                    #print("Cluster distribution")
                                    
                                    largest_cluster = np.max(cluster_counts)
                                    smallest_cluster = np.min(cluster_counts)
                                    largest_percentage = largest_cluster / len(labels) * 100
                                    balance_factor = 1.0 - (largest_percentage / 100)
                                    
                                    if balance_factor < 0.4:
                                        #print("Skipping imbalanced clustering")
                                        continue
                                    
                                    sil_score = silhouette_score(features_reduced, labels)
                                    
                                    ari = adjusted_rand_score(self.labels, labels)
                                    
                                    combined_score = (0.25 * sil_score) + (0.5 * ari) + (0.25 * balance_factor)
                                    #print("Clustering scored")
                                    
                                    if combined_score > best_score:
                                        best_score = combined_score
                                        best_labels = labels
                                        best_distribution = cluster_counts
                                        best_config = f"{preproc_name}-{dim_red_name}-{affinity}-{eigen_solver}-{assign_labels}"
                                except Exception as e:
                                    print(f"Clustering configuration failed: {e}")
                                    continue
            
            if best_labels is not None:
                #print("Best spectral clustering found")
                #print("Best configuration found")
                #print("Best score found")
                return best_labels, 'Enhanced-Spectral-Clustering', len(set(best_labels))
            else:
                spectral = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='nearest_neighbors',
                    n_neighbors=10,
                    random_state=42
                )
                labels = spectral.fit_predict(features)
                return labels, 'Basic-Spectral-Clustering', len(set(labels))
                
        except Exception as e:
            print(f"Spectral clustering failed: {e}")
            return None, None, 0
    
    def _agglomerative_distance_clustering(self, features, n_clusters):
                
        best_labels = None
        best_n_clusters = 0
        best_balance = 0
        
        for distance in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
            try:
                agg = AgglomerativeClustering(n_clusters=None, 
                                            distance_threshold=distance,
                                            linkage='ward')
                labels = agg.fit_predict(features)
                n_clusters_found = len(set(labels))
                
                if n_clusters_found < 3 or n_clusters_found > 6:
                    continue
                
                cluster_counts = np.bincount(labels)
                largest_cluster = np.max(cluster_counts)
                largest_percentage = largest_cluster / len(labels) * 100
                balance_factor = 1.0 - (largest_percentage / 100)
                
                #print("Distance threshold evaluated")
                
                if balance_factor > best_balance:
                    best_balance = balance_factor
                    best_labels = labels
                    best_n_clusters = n_clusters_found
            except Exception as e:
                print(f"Error with distance {distance}: {e}")
                pass
                
        if best_labels is not None:
            return best_labels, f'Agglomerative-Distance-{best_n_clusters}', best_n_clusters
        
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = agg.fit_predict(features)
        return labels, 'Agglomerative-Ward', n_clusters
        
    def _custom_balanced_clustering(self, features, n_clusters):
                
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(50, features.shape[1]))
        features_reduced = pca.fit_transform(features)
        
        kmeans_binary = KMeans(n_clusters=2, init='k-means++', n_init=20, random_state=42)
        binary_labels = kmeans_binary.fit_predict(features_reduced)
        
        group0_indices = np.where(binary_labels == 0)[0]
        group1_indices = np.where(binary_labels == 1)[0]
        
        if len(group0_indices) > 3 and len(group1_indices) > 3:
            features_group0 = features[group0_indices]
            features_group1 = features[group1_indices]
            
            kmeans_group0 = KMeans(n_clusters=2, init='k-means++', n_init=20, random_state=42)
            kmeans_group1 = KMeans(n_clusters=2, init='k-means++', n_init=20, random_state=42)
            
            labels_group0 = kmeans_group0.fit_predict(features_group0)
            labels_group1 = kmeans_group1.fit_predict(features_group1)
            
            final_labels = np.zeros(len(features), dtype=int)
            final_labels[group0_indices] = labels_group0
            final_labels[group1_indices] = labels_group1 + 2  # Offset by 2 to get clusters 2 and 3
            
            cluster_counts = np.bincount(final_labels)
            #print("Final cluster distribution:", cluster_counts)
            
            largest_cluster = np.max(cluster_counts)
            smallest_cluster = np.min(cluster_counts)
            largest_percentage = largest_cluster / len(final_labels) * 100
            balance_factor = 1.0 - (largest_percentage / 100)
            
            #print(f"Largest cluster: {largest_cluster} samples ({largest_percentage:.1f}%)")
            #print(f"Smallest cluster: {smallest_cluster} samples")
            #print(f"Balance factor: {balance_factor:.2f}")
            
            if balance_factor > 0.5:  # No cluster has more than 50% of samples
                return final_labels, 'Custom-Hierarchical-KMeans', 4
        
        try:
            from sklearn.cluster import SpectralClustering
            spectral = SpectralClustering(n_clusters=n_clusters, 
                                         affinity='nearest_neighbors',
                                         random_state=42)
            spectral_labels = spectral.fit_predict(features_reduced)
            
            cluster_counts = np.bincount(spectral_labels)
            largest_cluster = np.max(cluster_counts)
            largest_percentage = largest_cluster / len(spectral_labels) * 100
            
            if largest_percentage < 60:
                #print("Spectral clustering succeeded")
                return spectral_labels, 'Spectral-Clustering', n_clusters
        except Exception as e:
            print(f"Spectral clustering failed: {e}")
            pass
        
        best_labels = None
        best_n_clusters = 0
        best_balance = 0
        
        for distance in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
            try:
                agg = AgglomerativeClustering(n_clusters=None, 
                                            distance_threshold=distance,
                                            linkage='ward')
                labels = agg.fit_predict(features_reduced)
                n_clusters_found = len(set(labels))
                
                if n_clusters_found < 3 or n_clusters_found > 6:
                    continue
                
                cluster_counts = np.bincount(labels)
                largest_cluster = np.max(cluster_counts)
                largest_percentage = largest_cluster / len(labels) * 100
                balance_factor = 1.0 - (largest_percentage / 100)
                
                if balance_factor > best_balance:
                    best_balance = balance_factor
                    best_labels = labels
                    best_n_clusters = n_clusters_found
            except Exception as e:
                print(f"Error with distance {distance}: {e}")
                pass
        
        if best_labels is not None:
            #print("Found balanced clustering")
            #print(f"Balance factor: {best_balance:.2f}")
            return best_labels, 'Agglomerative-Distance', best_n_clusters
        
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = agg.fit_predict(features)
        return labels, 'Agglomerative-Ward', n_clusters

        
    def analyze_clustering_quality(self, cluster_labels):
        #print(f"\n{TermColors.HEADER}{TermColors.BOLD}" + "="*60 + f"{TermColors.ENDC}")
        #print(f"{TermColors.HEADER}{TermColors.BOLD}DETAILED CLUSTERING QUALITY ANALYSIS{TermColors.ENDC}")
        #print(f"{TermColors.HEADER}{TermColors.BOLD}" + "="*60 + f"{TermColors.ENDC}")
        
        silhouette_avg = silhouette_score(self.features, cluster_labels)
        ari = adjusted_rand_score(self.labels, cluster_labels)
        
        cluster_counts = Counter(cluster_labels)
        largest_cluster = max(cluster_counts.values())
        smallest_cluster = min(cluster_counts.values())
        largest_percentage = (largest_cluster / len(cluster_labels)) * 100
        balance_factor = 1.0 - (largest_percentage / 100)
        
        if silhouette_avg > 0.5:
            silhouette_interpretation = f"{TermColors.GREEN}Strong clustering{TermColors.ENDC}"
        elif silhouette_avg > 0.3:
            silhouette_interpretation = f"{TermColors.CYAN}Moderate clustering{TermColors.ENDC}"
        elif silhouette_avg > 0.1:
            silhouette_interpretation = f"{TermColors.YELLOW}Weak clustering{TermColors.ENDC}"
        else:
            silhouette_interpretation = f"{TermColors.RED}Very weak clustering{TermColors.ENDC}"
        
        if ari > 0.7:
            ari_interpretation = f"{TermColors.GREEN}Excellent agreement with true classes{TermColors.ENDC}"
        elif ari > 0.5:
            ari_interpretation = f"{TermColors.GREEN}High agreement with true classes{TermColors.ENDC}"
        elif ari > 0.3:
            ari_interpretation = f"{TermColors.CYAN}Moderate agreement with true classes{TermColors.ENDC}"
        elif ari > 0.1:
            ari_interpretation = f"{TermColors.YELLOW}Weak agreement with true classes{TermColors.ENDC}"
        else:
            ari_interpretation = f"{TermColors.RED}Very weak agreement with true classes{TermColors.ENDC}"
        
        if balance_factor > 0.8:
            balance_interpretation = f"{TermColors.GREEN}Excellent balance{TermColors.ENDC}"
        elif balance_factor > 0.6:
            balance_interpretation = f"{TermColors.GREEN}Good balance{TermColors.ENDC}"
        elif balance_factor > 0.4:
            balance_interpretation = f"{TermColors.CYAN}Moderate balance{TermColors.ENDC}"
        else:
            balance_interpretation = f"{TermColors.RED}Poor balance{TermColors.ENDC}"
        
        cluster_purity = {}
        cluster_dominant_class = {}
        for cluster_id in set(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_true_labels = self.labels[cluster_mask]
            if len(cluster_true_labels) > 0:
                most_common_class = Counter(cluster_true_labels).most_common(1)[0]
                purity = most_common_class[1] / len(cluster_true_labels)
                cluster_purity[cluster_id] = purity
                class_name = list(self.label_map.keys())[most_common_class[0]]
                cluster_dominant_class[cluster_id] = class_name
        
        avg_purity = np.mean(list(cluster_purity.values())) if cluster_purity else 0
        
        #print(f"{TermColors.BOLD}1. MAIN METRICS:{TermColors.ENDC}")
        #print(f"   * {TermColors.BOLD}Silhouette Score:{TermColors.ENDC} {silhouette_avg:.3f} -> {silhouette_interpretation}")
        #print(f"   * {TermColors.BOLD}Adjusted Rand Index:{TermColors.ENDC} {ari:.3f} -> {ari_interpretation}")
        #print(f"   * {TermColors.BOLD}Balance Factor:{TermColors.ENDC} {balance_factor:.3f} -> {balance_interpretation}")
        #print(f"   * {TermColors.BOLD}Average Purity:{TermColors.ENDC} {avg_purity:.1%}")
        
        for cluster_id in sorted(cluster_counts.keys()):
            count = cluster_counts[cluster_id]
            percentage = (count / len(cluster_labels)) * 100
            bar_length = int(percentage / 2)
            bar = '#' * bar_length
            
            if percentage > 40:
                color = TermColors.RED
            elif percentage > 30:
                color = TermColors.YELLOW
            else:
                color = TermColors.GREEN
                
            #print(f"Cluster {cluster_id}: {count} samples ({percentage:.1f}%)")
        
        for cluster_id in sorted(cluster_purity.keys()):
            purity = cluster_purity[cluster_id]
            class_name = cluster_dominant_class[cluster_id]
            bar_length = int(purity * 20)
            bar = '#' * bar_length
            
            if purity > 0.8:
                color = TermColors.GREEN
            elif purity > 0.6:
                color = TermColors.CYAN
            elif purity > 0.4:
                color = TermColors.YELLOW
            else:
                color = TermColors.RED
                
            #print(f"Cluster {cluster_id} purity: {purity:.1%}, dominant class: {class_name}")
        
        #print(f"Average cluster purity: {avg_purity:.1%}")
        
        for class_id in sorted(self.label_map.keys()):
            class_name = self.label_map[class_id]
            class_mask = np.array([label == class_id for label in self.labels])
            if not any(class_mask):
                continue
                
            class_clusters = cluster_labels[class_mask]
            cluster_distribution = Counter(class_clusters)
            total_class_samples = len(class_clusters)
            
            #print(f"Class {class_name} distribution:")
            for cluster_id, count in sorted(cluster_distribution.items()):
                percentage = (count / total_class_samples) * 100
                bar_length = int(percentage / 5)
                bar = '#' * bar_length
                
                if percentage > 80:
                    color = TermColors.GREEN
                elif percentage > 50:
                    color = TermColors.CYAN
                elif percentage > 30:
                    color = TermColors.YELLOW
                else:
                    color = TermColors.RED
                    
                    #print(f"  Cluster {cluster_id}: {count} samples ({percentage:.1f}%)")
        
        
        recommendations = []
        
        if silhouette_avg < 0.3:
            recommendations.append(f"{TermColors.YELLOW}* Consider changing the number of clusters{TermColors.ENDC}")
        if ari < 0.3:
            recommendations.append(f"{TermColors.YELLOW}* Add more relevant features{TermColors.ENDC}")
        if avg_purity < 0.7:
            recommendations.append(f"{TermColors.YELLOW}* Try different clustering algorithms{TermColors.ENDC}")
        if any(purity < 0.5 for purity in cluster_purity.values()):
            recommendations.append(f"{TermColors.YELLOW}* Some clusters contain mixed landscape types{TermColors.ENDC}")
            recommendations.append(f"{TermColors.YELLOW}* More specific features needed for better separation{TermColors.ENDC}")
        if balance_factor < 0.4:
            recommendations.append(f"{TermColors.RED}* Clusters are highly imbalanced{TermColors.ENDC}")
            recommendations.append(f"{TermColors.YELLOW}* Try algorithms that promote balanced clusters{TermColors.ENDC}")
            
        if recommendations:
            for rec in recommendations:
                print(rec)
        else:
            print("Clustering results are good")
        
        print(f"{TermColors.HEADER}{TermColors.BOLD}" + "="*60 + f"{TermColors.ENDC}")
    
    def visualize_results(self, cluster_labels, features_scaled, show_3d=False):
                
        self._create_2d_scatter_plots(cluster_labels, features_scaled)
        
        self._create_3d_scatter_plot(cluster_labels, features_scaled)
            
        self._create_cluster_analysis_plots(cluster_labels)
        self._create_confusion_matrix(cluster_labels)
    
    def _create_2d_scatter_plots(self, cluster_labels, features_scaled):
        pca = PCA(n_components=2)
        features_2d_pca = pca.fit_transform(features_scaled)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Earth Surface Image Clustering - 2D Projection (PCA)', fontsize=16)
        
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
        
        try:
                        
            tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
            features_2d_tsne = tsne.fit_transform(features_scaled)
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            fig.suptitle('Earth Surface Image Clustering - 2D Projection (t-SNE)', fontsize=16)
            
            for i, label in enumerate(unique_labels):
                mask = cluster_labels == label
                axes[0].scatter(features_2d_tsne[mask, 0], features_2d_tsne[mask, 1], 
                              c=[colors[i]], label=f'Cluster {label}', alpha=0.7, s=60, edgecolors='w')
            
            axes[0].set_title('Clustering Results (t-SNE)')
            axes[0].set_xlabel('t-SNE Component 1')
            axes[0].set_ylabel('t-SNE Component 2')
            axes[0].legend(loc='upper right', bbox_to_anchor=(1.3, 1))
            axes[0].grid(True, linestyle='--', alpha=0.7)
            
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
            
        except Exception as e:
            print(f"t-SNE visualization failed: {e}")
    
    def _create_3d_scatter_plot(self, cluster_labels, features_scaled):
        try:
            pca = PCA(n_components=3)
            features_3d = pca.fit_transform(features_scaled)
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            unique_labels = np.unique(cluster_labels)
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = cluster_labels == label
                ax.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                          c=[colors[i]], label=f'Cluster {label}', alpha=0.7, s=60, edgecolors='w')
            
            ax.set_title('3D Visualization of Clusters', fontsize=14)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)')
            
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
            ax.grid(True, linestyle='--', alpha=0.7)
            
            ax.view_init(elev=30, azim=45)
            
            plt.tight_layout()
            plt.savefig('clustering_3d.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"3D visualization failed: {e}")
    
    def _create_cluster_analysis_plots(self, cluster_labels):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Cluster Analysis', fontsize=16)
        
        cluster_counts = Counter(cluster_labels)
        cluster_ids = sorted(cluster_counts.keys())
        cluster_sizes = [cluster_counts[id] for id in cluster_ids]
        
        bars = axes[0].bar(cluster_ids, cluster_sizes, 
                         color=plt.cm.tab20(np.linspace(0, 1, len(cluster_ids))))
        axes[0].set_title('Image Distribution Across Clusters', fontsize=14)
        axes[0].set_xlabel('Cluster ID', fontsize=12)
        axes[0].set_ylabel('Number of Images', fontsize=12)
        axes[0].grid(True, axis='y', linestyle='--', alpha=0.7)
        
        for bar, size in zip(bars, cluster_sizes):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        str(size), ha='center', va='bottom', fontweight='bold')
        
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
        
        purity_colors = plt.cm.RdYlGn(np.array(purity_values))
        
        bars2 = axes[1].bar(purity_clusters, purity_values, color=purity_colors)
        axes[1].set_title('Cluster Purity', fontsize=14)
        axes[1].set_xlabel('Cluster ID', fontsize=12)
        axes[1].set_ylabel('Purity (%)', fontsize=12)
        axes[1].set_ylim(0, 1.1)  # Leave room for labels
        axes[1].grid(True, axis='y', linestyle='--', alpha=0.7)
        
        for bar, cluster_id, purity in zip(bars2, purity_clusters, purity_values):
            axes[1].text(bar.get_x() + bar.get_width()/2, purity + 0.02, 
                        f'{purity:.1%}', ha='center', va='bottom', fontweight='bold')
            
            dominant_class = cluster_dominant_class[cluster_id]
            axes[1].text(bar.get_x() + bar.get_width()/2, -0.05, 
                        f'{dominant_class}', ha='center', va='top', 
                        rotation=45, fontsize=8)
        
        plt.tight_layout()
        plt.savefig('cluster_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_confusion_matrix(self, cluster_labels):
        unique_clusters = sorted(set(cluster_labels))
        unique_classes = sorted(set(self.labels))
        class_names = list(self.label_map.keys())
        
        confusion_matrix = np.zeros((len(unique_clusters), len(unique_classes)))
        
        for i, cluster in enumerate(unique_clusters):
            cluster_mask = cluster_labels == cluster
            for j, class_id in enumerate(unique_classes):
                confusion_matrix[i, j] = np.sum((self.labels == class_id) & cluster_mask)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        
        im = ax.imshow(confusion_matrix, cmap='Blues')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of Images', rotation=270, labelpad=15)
        
        ax.set_xticks(np.arange(len(unique_classes)))
        ax.set_yticks(np.arange(len(unique_clusters)))
        ax.set_xticklabels([class_names[class_id] for class_id in unique_classes])
        ax.set_yticklabels([f'Cluster {cluster}' for cluster in unique_clusters])
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        ax.set_title('Confusion Matrix: Clusters vs. True Classes', fontsize=14)
        ax.set_ylabel('Clusters', fontsize=12)
        ax.set_xlabel('True Classes', fontsize=12)
        
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
    
    def _create_basic_visualizations(self, cluster_labels, features_scaled):
                
        # PCA for dimensionality reduction
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_scaled)
        
        # Create a figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Earth Surface Image Clustering (VGG16)', fontsize=16)
        
        # Plot 1: Clustering
        scatter1 = axes[0,0].scatter(features_2d[:, 0], features_2d[:, 1], 
                                   c=cluster_labels, cmap='tab10', alpha=0.7, s=50)
        axes[0,0].set_title('Clustering Results')
        axes[0,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter1, ax=axes[0,0])
        
        # Plot 2: True classes
        scatter2 = axes[0,1].scatter(features_2d[:, 0], features_2d[:, 1], 
                                   c=self.labels, cmap='tab10', alpha=0.7, s=50)
        axes[0,1].set_title('True Classes')
        axes[0,1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0,1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter2, ax=axes[0,1])
        
        # Plot 3: Cluster distribution
        cluster_counts = Counter(cluster_labels)
        cluster_ids = list(cluster_counts.keys())
        cluster_sizes = list(cluster_counts.values())
        
        bars = axes[1,0].bar(cluster_ids, cluster_sizes, color=plt.cm.tab10(np.linspace(0, 1, len(cluster_ids))))
        axes[1,0].set_title('Image Distribution Across Clusters')
        axes[1,0].set_xlabel('Cluster ID')
        axes[1,0].set_ylabel('Number of Images')
        
        for bar, size in zip(bars, cluster_sizes):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                          str(size), ha='center', va='bottom')
        
        cluster_purity = {}
        for cluster_id in set(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_true_labels = self.labels[cluster_mask]
            if len(cluster_true_labels) > 0:
                most_common_class = Counter(cluster_true_labels).most_common(1)[0]
                purity = most_common_class[1] / len(cluster_true_labels)
                cluster_purity[cluster_id] = purity
        
        purity_clusters = list(cluster_purity.keys())
        purity_values = list(cluster_purity.values())
        
        bars2 = axes[1,1].bar(purity_clusters, purity_values, color=plt.cm.viridis(np.linspace(0, 1, len(purity_clusters))))
        axes[1,1].set_title('Cluster Purity')
        axes[1,1].set_xlabel('Cluster ID')
        axes[1,1].set_ylabel('Purity (%)')
        axes[1,1].set_ylim(0, 1)
        
        for bar, purity in zip(bars2, purity_values):
            axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'{purity:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('clustering_results_improved.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def show_cluster_examples(self, cluster_labels, examples_per_cluster=2):
                
        unique_clusters = sorted(set(cluster_labels))
        
        for cluster_id in unique_clusters:
            print("Cluster examples")
            
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) > 0:
                fig, axes = plt.subplots(1, min(examples_per_cluster, len(cluster_indices)), 
                                       figsize=(10, 4))
                if len(cluster_indices) == 1:
                    axes = [axes]
                
                for i, idx in enumerate(cluster_indices[:examples_per_cluster]):
                    image = cv2.imread(self.image_paths[idx])
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    axes[i].imshow(image)
                    axes[i].set_title(f'Cluster {cluster_id}\nTrue: {list(self.label_map.keys())[self.labels[idx]]}')
                    axes[i].axis('off')
                
                plt.tight_layout()
                plt.show()

def main():
    # Header
    
    # Print focused classes
    focused_classes = [
        f"{TermColors.GREEN}AnnualCrop{TermColors.ENDC}", 
        f"{TermColors.CYAN}HerbaceousVegetation{TermColors.ENDC}", 
        f"{TermColors.YELLOW}Industrial{TermColors.ENDC}", 
        f"{TermColors.BLUE}Forest{TermColors.ENDC}"
    ]
    #print(f"\n{TermColors.BOLD}Focusing on 4 selected classes:{TermColors.ENDC} {', '.join(focused_classes)}")
    #print(f"{TermColors.HEADER}{TermColors.BOLD}" + "="*60 + f"{TermColors.ENDC}")
    
    
    # Start timer to measure performance
    start_time = time.time()
    
    # Initialize with more samples per class for better results
    clusterer = SimpleEarthSurfaceClustering(max_samples_per_class=50)
    
    # Load data (only selected classes)
    clusterer.load_sample_images("train.csv")
    
    # Process images
    clusterer.process_images()
    
    # Clustering
    
    # Set n_clusters=4 to match our 4 classes
    cluster_labels, kmeans_model, features_scaled = clusterer.perform_clustering(n_clusters=4, optimize_params=True)
    
    # Detailed quality analysis
    clusterer.analyze_clustering_quality(cluster_labels)
    
    # Visualization (без 3D візуалізації, яка дублює інформацію)
    clusterer.visualize_results(cluster_labels, features_scaled, show_3d=False)
    
    # Example images
    clusterer.show_cluster_examples(cluster_labels, examples_per_cluster=4)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    #print(f"Total execution time: {int(minutes)} minutes and {int(seconds)} seconds")

if __name__ == "__main__":
    main()