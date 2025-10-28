"""
Simplified version of Earth surface image clustering
Uses ONLY VGG16 (as in the original Lab_work_5 examples)
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import enhanced visualization methods
try:
    from enhanced_visualizations import create_2d_scatter_plots, create_3d_scatter_plot, create_cluster_analysis_plots, create_confusion_matrix
    ENHANCED_VIZ_AVAILABLE = True
except ImportError:
    ENHANCED_VIZ_AVAILABLE = False
    print("Enhanced visualizations not available. Using basic visualizations.")

# VGG16 (as in the original examples)
try:
    from keras.applications.vgg16 import VGG16, preprocess_input
    from keras.utils import load_img, img_to_array
    VGG16_AVAILABLE = True
except ImportError:
    VGG16_AVAILABLE = False
    print("VGG16 is not available. Please install TensorFlow and Keras.")

class SimpleEarthSurfaceClustering:
    def __init__(self, data_path=".", max_samples_per_class=30):
        """
        Simplified class for Earth surface image clustering
        Uses ONLY VGG16 (as in the original examples)
        """
        self.data_path = data_path
        self.max_samples_per_class = max_samples_per_class
        self.label_map = self.load_label_map()
        self.features = []
        self.labels = []
        self.image_paths = []
        
        # VGG16 model (as in the original)
        if VGG16_AVAILABLE:
            print("Initializing VGG16 model...")
            self.model = VGG16(weights='imagenet', include_top=False)
            print("VGG16 model loaded successfully!")
        else:
            self.model = None
            print("VGG16 is not available!")
    
    def load_label_map(self):
        """Load label map"""
        with open(os.path.join(self.data_path, 'label_map.json'), 'r') as f:
            return json.load(f)
    
    def load_sample_images(self, csv_file="train.csv"):
        """Load sample images for selected classes only"""
        print(f"Loading data from {csv_file}...")
        
        # Define the classes we want to keep
        selected_classes = ['AnnualCrop', 'HerbaceousVegetation', 'Industrial', 'Forest']
        print(f"Focusing on {len(selected_classes)} selected classes: {', '.join(selected_classes)}")
        
        df = pd.read_csv(os.path.join(self.data_path, csv_file))
        
        # Filter dataframe to only include selected classes
        df_filtered = df[df['ClassName'].isin(selected_classes)]
        
        # Group by class and sample
        sampled_data = []
        for class_name in selected_classes:
            class_data = df_filtered[df_filtered['ClassName'] == class_name]
            if len(class_data) > 0:
                sample_size = min(self.max_samples_per_class, len(class_data))
                sampled = class_data.sample(n=sample_size, random_state=42)
                sampled_data.append(sampled)
        
        self.df = pd.concat(sampled_data, ignore_index=True)
        print(f"Loaded {len(self.df)} images from {len(selected_classes)} classes")
        
        # Show class distribution
        class_counts = self.df['ClassName'].value_counts()
        print("\nClass distribution:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} images")
    
    def extract_features(self, image_path):
        """
        Enhanced feature extraction with vegetation-specific features
        """
        if not self.model:
            return np.array([])
        
        try:
            # Load and preprocess image
            img = load_img(image_path, target_size=(224, 224))
            img_data = img_to_array(img)
            
            # Convert to OpenCV format for additional processing
            cv_img = cv2.cvtColor(img_data.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # 1. Color features (enhanced)
            # Calculate color histograms (RGB) with more bins for better color discrimination
            color_features = []
            for i in range(3):  # RGB channels
                hist = cv2.calcHist([cv_img], [i], None, [64], [0, 256])  # More bins (64 instead of 32)
                hist = cv2.normalize(hist, hist).flatten()
                color_features.extend(hist)
            
            # 2. Color space transformations
            # HSV color space (better for vegetation)
            hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
            hsv_features = []
            for i in range(3):  # HSV channels
                hist = cv2.calcHist([hsv_img], [i], None, [32], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                hsv_features.extend(hist)
            
            # 3. Texture features
            # Convert to grayscale for texture analysis
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            
            # Gabor filter bank for texture analysis (especially good for vegetation vs urban)
            gabor_features = []
            for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:  # Different orientations
                for sigma in [1, 3]:  # Different scales
                    kernel = cv2.getGaborKernel((21, 21), sigma, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
                    filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                    # Extract mean and std as features
                    mean = np.mean(filtered)
                    std = np.std(filtered)
                    gabor_features.extend([mean, std])
            
            # 4. Edge features
            # Canny edge detection
            edges = cv2.Canny(gray, 100, 200)
            edge_percentage = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Sobel edge detection (captures different edge types)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_mag = np.sqrt(sobelx**2 + sobely**2)
            sobel_features = [np.mean(sobel_mag), np.std(sobel_mag)]
            
            # 5. Vegetation-specific indices
            # Simple green ratio (approximation of vegetation index)
            b, g, r = cv2.split(cv_img)
            green_ratio = np.mean(g) / (np.mean(r) + np.mean(b) + 1e-10)  # Avoid division by zero
            
            # Green minus red difference (another vegetation indicator)
            green_red_diff = np.mean(g) - np.mean(r)
            
            # 6. Deep features from VGG16
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)
            deep_features = np.array(self.model.predict(img_data, verbose=0)).flatten()
            
            # Combine all features with appropriate weighting
            # We'll give more weight to features that help distinguish vegetation types
            combined_features = np.concatenate([
                deep_features,                                # VGG16 features
                np.array(color_features) * 2.0,             # Color histograms (weighted)
                np.array(hsv_features) * 1.5,               # HSV features (weighted)
                np.array(gabor_features) * 1.2,             # Texture features (weighted)
                np.array([edge_percentage * 2.0]),          # Edge density (weighted)
                np.array(sobel_features),                   # Sobel edge features
                np.array([green_ratio * 3.0]),              # Vegetation index (heavily weighted)
                np.array([green_red_diff * 3.0])            # Green-red difference (heavily weighted)
            ])
            
            return combined_features
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return np.array([])
    
    def process_images(self):
        """Process all images (as in the original examples)"""
        print("Processing images and extracting features...")
        
        self.features = []
        self.labels = []
        self.image_paths = []
        
        for idx, row in self.df.iterrows():
            if idx % 50 == 0:
                print(f"    Status: {idx} / {len(self.df)}", end="\r")
            
            image_path = os.path.join(self.data_path, row['Filename'])
            
            if not os.path.exists(image_path):
                continue
            
            # Extract features
            features = self.extract_features(image_path)
            
            if len(features) > 0:
                self.features.append(features)
                self.labels.append(row['Label'])
                self.image_paths.append(image_path)
        
        print(f"\nSuccessfully processed {len(self.features)} images")
        
        # Convert to numpy arrays
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        
        print(f"Feature dimensions: {self.features.shape}")
    
    def find_optimal_clusters(self, features, max_clusters=15):
        """Find optimal number of clusters using the elbow method"""
        print("\nSearching for optimal number of clusters...")
        
        # Range of cluster numbers to check
        range_n_clusters = range(2, min(max_clusters + 1, len(self.labels) // 5))
        
        # Store metric values
        silhouette_scores = []
        inertia_values = []
        
        for n_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            # Calculate metrics
            silhouette_avg = silhouette_score(features, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            inertia_values.append(kmeans.inertia_)
            
            print(f"  • Clusters: {n_clusters}, Silhouette: {silhouette_avg:.3f}, Inertia: {kmeans.inertia_:.1f}")
        
        # Find optimal number of clusters using the elbow method
        # Look for the biggest drop in inertia
        diffs = np.diff(inertia_values)
        second_diffs = np.diff(diffs)
        if len(second_diffs) > 0:
            optimal_clusters = np.argmax(second_diffs) + 3  # +3 because we're looking at the second derivative
            optimal_clusters = max(2, min(optimal_clusters, max_clusters))
            print(f"\nOptimal number of clusters (elbow method): {optimal_clusters}")
            return optimal_clusters
        
        # If we can't determine using the second derivative, use Silhouette
        optimal_clusters = np.argmax(silhouette_scores) + 2  # +2 because we start from 2 clusters
        print(f"\nOptimal number of clusters (by Silhouette): {optimal_clusters}")
        return optimal_clusters

    def perform_clustering(self, n_clusters=None, optimize_params=True):
        """Enhanced clustering specifically for the 4 vegetation/urban classes"""
        print(f"\nRunning specialized clustering for vegetation/urban classes...")
        
        # Feature normalization
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features)
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=0.95)  # Keep 95% of variance
        features_pca = pca.fit_transform(features_scaled)
        print(f"Reduced dimensions from {features_scaled.shape[1]} to {features_pca.shape[1]} components")
        
        # For our 4-class problem, we'll directly use our specialized approach
        print("\nUsing specialized clustering approach for the 4 vegetation/urban classes...")
        
        # Step 1: Create a custom feature space that emphasizes vegetation differences
        print("Step 1: Creating specialized feature space...")
        
        # Extract vegetation-specific features from the original features
        # These are the indices of the vegetation-related features we added
        # (green ratio and green-red difference are at the end of our feature vector)
        veg_feature_indices = [-2, -1]  # Last two features (vegetation indices)
        
        # Weight these features more heavily
        weighted_features = features_scaled.copy()
        for idx in veg_feature_indices:
            if abs(idx) < weighted_features.shape[1]:
                weighted_features[:, idx] *= 5.0  # Increase weight of vegetation features
        
        # Apply PCA to the weighted features
        pca_weighted = PCA(n_components=min(50, weighted_features.shape[1]))
        features_weighted = pca_weighted.fit_transform(weighted_features)
        
        # Step 2: Try multiple clustering approaches and select the best one
        print("Step 2: Trying multiple specialized clustering approaches...")
        
        clustering_approaches = [
            # Approach 1: Two-stage hierarchical clustering
            self._two_stage_clustering,
            
            # Approach 2: Spectral clustering with nearest neighbors affinity
            self._spectral_clustering,
            
            # Approach 3: Agglomerative clustering with distance threshold
            self._agglomerative_distance_clustering
        ]
        
        best_labels = None
        best_algorithm = None
        best_n_clusters = 0
        best_score = -1
        
        for approach_func in clustering_approaches:
            try:
                # Apply the clustering approach
                labels, algorithm_name, n_clusters_found = approach_func(features_weighted, n_clusters or 4)
                
                if labels is None or n_clusters_found < 2:
                    continue
                
                # Evaluate clustering quality
                cluster_counts = np.bincount(labels)
                largest_cluster = np.max(cluster_counts)
                smallest_cluster = np.min(cluster_counts)
                largest_percentage = largest_cluster / len(labels) * 100
                balance_factor = 1.0 - (largest_percentage / 100)
                
                print(f"\nEvaluating {algorithm_name}:")
                print(f"  - Found {n_clusters_found} clusters")
                print(f"  - Cluster distribution: {cluster_counts}")
                print(f"  - Largest cluster: {largest_cluster} samples ({largest_percentage:.1f}%)")
                print(f"  - Smallest cluster: {smallest_cluster} samples")
                print(f"  - Balance factor: {balance_factor:.3f}")
                
                # Calculate quality metrics
                score = silhouette_score(features_weighted, labels)
                ari = adjusted_rand_score(self.labels, labels)
                
                print(f"  - Silhouette Score: {score:.3f}")
                print(f"  - Adjusted Rand Index: {ari:.3f}")
                
                # Combined score with emphasis on ARI and balance
                combined_score = (0.3 * score) + (0.4 * ari) + (0.3 * balance_factor)
                print(f"  - Combined Score: {combined_score:.3f}")
                
                # Add bonus for having close to 4 clusters
                cluster_proximity = 1.0 - min(abs(n_clusters_found - 4), 2) / 2.0
                adjusted_score = combined_score + (0.1 * cluster_proximity)
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_labels = labels
                    best_algorithm = algorithm_name
                    best_n_clusters = n_clusters_found
                    
            except Exception as e:
                print(f"Error with clustering approach: {e}")
        
        # If no approach worked well, fall back to K-means
        if best_labels is None:
            print("\nNo specialized approach worked well. Falling back to K-means...")
            kmeans = KMeans(n_clusters=4, init='k-means++', n_init=30, random_state=42)
            best_labels = kmeans.fit_predict(features_weighted)
            best_algorithm = 'KMeans-Fallback'
            best_n_clusters = 4
        
        # Final processing of results
        print(f"\nSelected algorithm: {best_algorithm}")
        print(f"Number of clusters found: {best_n_clusters}")
        
        # Analyze results
        self.analyze_clustering_results(best_labels, features_weighted)
        
        return best_labels, None, features_weighted
        
    def _two_stage_clustering(self, features, n_clusters):
        """Two-stage hierarchical clustering approach"""
        print("\nTrying two-stage hierarchical clustering...")
        
        # First try with 2 clusters (vegetation vs non-vegetation)
        kmeans_binary = KMeans(n_clusters=2, init='k-means++', n_init=20, random_state=42)
        binary_labels = kmeans_binary.fit_predict(features)
        
        # Get indices for each binary group
        group0_indices = np.where(binary_labels == 0)[0]
        group1_indices = np.where(binary_labels == 1)[0]
        
        # Check if we have enough samples in each group
        if len(group0_indices) > 3 and len(group1_indices) > 3:
            # Extract features for each group
            features_group0 = features[group0_indices]
            features_group1 = features[group1_indices]
            
            # Apply K-means to each group separately
            kmeans_group0 = KMeans(n_clusters=2, init='k-means++', n_init=20, random_state=42)
            kmeans_group1 = KMeans(n_clusters=2, init='k-means++', n_init=20, random_state=42)
            
            # Get subgroup labels
            labels_group0 = kmeans_group0.fit_predict(features_group0)
            labels_group1 = kmeans_group1.fit_predict(features_group1)
            
            # Create final labels
            final_labels = np.zeros(len(features), dtype=int)
            final_labels[group0_indices] = labels_group0
            final_labels[group1_indices] = labels_group1 + 2  # Offset by 2 to get clusters 2 and 3
            
            # Check cluster distribution
            cluster_counts = np.bincount(final_labels)
            print(f"Two-stage cluster distribution: {cluster_counts}")
            
            return final_labels, 'Two-Stage-Hierarchical', len(set(final_labels))
        
        return None, None, 0
    
    def _spectral_clustering(self, features, n_clusters):
        """Spectral clustering approach"""
        print("\nTrying spectral clustering...")
        
        try:
            from sklearn.cluster import SpectralClustering
            spectral = SpectralClustering(n_clusters=n_clusters, 
                                         affinity='nearest_neighbors',
                                         random_state=42)
            labels = spectral.fit_predict(features)
            
            # Check cluster distribution
            cluster_counts = np.bincount(labels)
            print(f"Spectral clustering distribution: {cluster_counts}")
            
            return labels, 'Spectral-Clustering', len(set(labels))
        except Exception as e:
            print(f"Spectral clustering failed: {e}")
            return None, None, 0
    
    def _agglomerative_distance_clustering(self, features, n_clusters):
        """Agglomerative clustering with distance threshold"""
        print("\nTrying agglomerative clustering with distance threshold...")
        
        best_labels = None
        best_n_clusters = 0
        best_balance = 0
        
        # Try different distance thresholds
        for distance in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
            try:
                # Use distance threshold instead of fixed number of clusters
                agg = AgglomerativeClustering(n_clusters=None, 
                                            distance_threshold=distance,
                                            linkage='ward')
                labels = agg.fit_predict(features)
                n_clusters_found = len(set(labels))
                
                # Skip if too few or too many clusters
                if n_clusters_found < 3 or n_clusters_found > 6:
                    continue
                
                # Check balance
                cluster_counts = np.bincount(labels)
                largest_cluster = np.max(cluster_counts)
                largest_percentage = largest_cluster / len(labels) * 100
                balance_factor = 1.0 - (largest_percentage / 100)
                
                print(f"Distance {distance}: {n_clusters_found} clusters, balance: {balance_factor:.3f}")
                
                if balance_factor > best_balance:
                    best_balance = balance_factor
                    best_labels = labels
                    best_n_clusters = n_clusters_found
            except Exception as e:
                print(f"Error with distance {distance}: {e}")
        
        if best_labels is not None:
            return best_labels, f'Agglomerative-Distance-{best_n_clusters}', best_n_clusters
        
        # If distance threshold didn't work, try standard approach
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = agg.fit_predict(features)
        return labels, 'Agglomerative-Ward', n_clusters
        
    def _custom_balanced_clustering(self, features, n_clusters):
        """Custom clustering approach optimized for 4 vegetation/urban classes"""
        print("Using custom clustering approach optimized for vegetation/urban classes...")
        
        # For our specific 4-class problem, we'll try a two-stage approach:
        # 1. First separate vegetation (Forest, HerbaceousVegetation) from non-vegetation (AnnualCrop, Industrial)
        # 2. Then further separate each group into its two classes
        
        # Step 1: Try to separate vegetation from non-vegetation using weighted features
        print("Step 1: Separating vegetation from non-vegetation...")
        
        # Apply PCA with fewer components to focus on the most important features
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(50, features.shape[1]))
        features_reduced = pca.fit_transform(features)
        
        # First try with 2 clusters (vegetation vs non-vegetation)
        kmeans_binary = KMeans(n_clusters=2, init='k-means++', n_init=20, random_state=42)
        binary_labels = kmeans_binary.fit_predict(features_reduced)
        
        # Step 2: For each of the 2 groups, further split into 2 subgroups
        print("Step 2: Further separating each group...")
        
        # Get indices for each binary group
        group0_indices = np.where(binary_labels == 0)[0]
        group1_indices = np.where(binary_labels == 1)[0]
        
        # Check if we have enough samples in each group
        if len(group0_indices) > 3 and len(group1_indices) > 3:
            # Extract features for each group
            features_group0 = features[group0_indices]
            features_group1 = features[group1_indices]
            
            # Apply K-means to each group separately
            kmeans_group0 = KMeans(n_clusters=2, init='k-means++', n_init=20, random_state=42)
            kmeans_group1 = KMeans(n_clusters=2, init='k-means++', n_init=20, random_state=42)
            
            # Get subgroup labels
            labels_group0 = kmeans_group0.fit_predict(features_group0)
            labels_group1 = kmeans_group1.fit_predict(features_group1)
            
            # Create final labels
            final_labels = np.zeros(len(features), dtype=int)
            final_labels[group0_indices] = labels_group0
            final_labels[group1_indices] = labels_group1 + 2  # Offset by 2 to get clusters 2 and 3
            
            # Check cluster distribution
            cluster_counts = np.bincount(final_labels)
            print(f"Final cluster distribution: {cluster_counts}")
            
            # Calculate balance metrics
            largest_cluster = np.max(cluster_counts)
            smallest_cluster = np.min(cluster_counts)
            largest_percentage = largest_cluster / len(final_labels) * 100
            balance_factor = 1.0 - (largest_percentage / 100)
            
            print(f"Largest cluster: {largest_cluster} samples ({largest_percentage:.1f}%)")
            print(f"Smallest cluster: {smallest_cluster} samples")
            print(f"Balance factor: {balance_factor:.3f}")
            
            # If reasonably balanced, return these labels
            if balance_factor > 0.5:  # No cluster has more than 50% of samples
                return final_labels, 'Custom-Hierarchical-KMeans', 4
        
        # If the two-stage approach didn't work well, try a different approach
        print("Two-stage approach didn't produce balanced clusters. Trying spectral clustering...")
        
        # Try Spectral Clustering which often works well for complex data
        try:
            from sklearn.cluster import SpectralClustering
            spectral = SpectralClustering(n_clusters=n_clusters, 
                                         affinity='nearest_neighbors',
                                         random_state=42)
            spectral_labels = spectral.fit_predict(features_reduced)
            
            # Check cluster distribution
            cluster_counts = np.bincount(spectral_labels)
            largest_cluster = np.max(cluster_counts)
            largest_percentage = largest_cluster / len(spectral_labels) * 100
            
            if largest_percentage < 60:  # If reasonably balanced
                print(f"Spectral clustering produced balanced results: {cluster_counts}")
                return spectral_labels, 'Spectral-Clustering', n_clusters
        except Exception as e:
            print(f"Spectral clustering failed: {e}")
        
        # Final fallback: Use Agglomerative clustering with ward linkage
        print("Trying Agglomerative clustering with custom distance threshold...")
        
        # Try different distance thresholds to get approximately 4 clusters
        best_labels = None
        best_n_clusters = 0
        best_balance = 0
        
        for distance in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
            try:
                # Use distance threshold instead of fixed number of clusters
                agg = AgglomerativeClustering(n_clusters=None, 
                                            distance_threshold=distance,
                                            linkage='ward')
                labels = agg.fit_predict(features_reduced)
                n_clusters_found = len(set(labels))
                
                # Skip if too few or too many clusters
                if n_clusters_found < 3 or n_clusters_found > 6:
                    continue
                
                # Check balance
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
        
        if best_labels is not None:
            print(f"Found balanced clustering with {best_n_clusters} clusters")
            print(f"Balance factor: {best_balance:.3f}")
            return best_labels, 'Agglomerative-Distance', best_n_clusters
        
        # If all else fails, use standard Agglomerative clustering
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = agg.fit_predict(features)
        return labels, 'Agglomerative-Ward', n_clusters

    def analyze_clustering_results(self, labels, features):
        """Analyze clustering results"""
        # Remove noise points for analysis
        core_samples_mask = labels != -1
        if -1 in labels:  # If there are noise points
            print(f"\nFound {np.sum(labels == -1)} noise points")
            labels_clean = labels[core_samples_mask]
            features_clean = features[core_samples_mask]
        else:
            labels_clean = labels
            features_clean = features
        
        # Quality assessment
        if len(set(labels_clean)) > 1:
            silhouette_avg = silhouette_score(features_clean, labels_clean)
            print(f"\nClustering quality assessment:")
            print(f"  • Silhouette Score: {silhouette_avg:.3f}")
        
        # Cluster distribution analysis
        cluster_counts = Counter(labels_clean)
        print("\nImage distribution across clusters:")
        for cluster_id, count in sorted(cluster_counts.items()):
            print(f"  • Cluster {cluster_id}: {count} images")
        
        # Cluster purity analysis
        print("\nCluster purity:")
        for cluster_id in set(labels_clean):
            cluster_mask = (labels == cluster_id)
            cluster_true_labels = self.labels[cluster_mask]
            if len(cluster_true_labels) > 0:
                most_common = Counter(cluster_true_labels).most_common(1)[0]
                purity = most_common[1] / len(cluster_true_labels)
                class_name = list(self.label_map.keys())[most_common[0]]
                print(f"  • Cluster {cluster_id}: {purity:.1%} purity (dominant: {class_name})")
    
    def analyze_clustering_quality(self, cluster_labels):
        """Detailed clustering quality analysis"""
        print("\n" + "="*60)
        print("DETAILED CLUSTERING QUALITY ANALYSIS")
        print("="*60)
        
        # Main metrics
        silhouette_avg = silhouette_score(self.features, cluster_labels)
        ari = adjusted_rand_score(self.labels, cluster_labels)
        
        print(f"1. MAIN METRICS:")
        print(f"   • Silhouette Score: {silhouette_avg:.3f}")
        print(f"   • Adjusted Rand Index: {ari:.3f}")
        
        # Interpretation
        if silhouette_avg > 0.5:
            silhouette_interpretation = "Strong clustering"
        elif silhouette_avg > 0.3:
            silhouette_interpretation = "Moderate clustering"
        elif silhouette_avg > 0.1:
            silhouette_interpretation = "Weak clustering"
        else:
            silhouette_interpretation = "Very weak clustering"
        
        if ari > 0.5:
            ari_interpretation = "High agreement with true classes"
        elif ari > 0.3:
            ari_interpretation = "Moderate agreement with true classes"
        elif ari > 0.1:
            ari_interpretation = "Weak agreement with true classes"
        else:
            ari_interpretation = "Very weak agreement with true classes"
        
        print(f"   • Silhouette interpretation: {silhouette_interpretation}")
        print(f"   • ARI interpretation: {ari_interpretation}")
        
        # Cluster distribution
        cluster_counts = Counter(cluster_labels)
        print(f"\n2. CLUSTER DISTRIBUTION:")
        for cluster_id in sorted(cluster_counts.keys()):
            count = cluster_counts[cluster_id]
            percentage = (count / len(cluster_labels)) * 100
            print(f"   • Cluster {cluster_id}: {count} images ({percentage:.1f}%)")
        
        # Cluster purity analysis
        print(f"\n3. CLUSTER PURITY ANALYSIS:")
        cluster_purity = {}
        for cluster_id in set(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_true_labels = self.labels[cluster_mask]
            if len(cluster_true_labels) > 0:
                most_common_class = Counter(cluster_true_labels).most_common(1)[0]
                purity = most_common_class[1] / len(cluster_true_labels)
                cluster_purity[cluster_id] = purity
                
                class_name = list(self.label_map.keys())[most_common_class[0]]
                print(f"   • Cluster {cluster_id}: {purity:.1%} purity (dominant: {class_name})")
        
        avg_purity = np.mean(list(cluster_purity.values())) if cluster_purity else 0
        print(f"   • Average purity: {avg_purity:.1%}")
        
        # True class distribution across clusters
        print(f"\n4. TRUE CLASS DISTRIBUTION ACROSS CLUSTERS:")
        for class_id, class_name in self.label_map.items():
            class_mask = self.labels == class_id
            class_clusters = cluster_labels[class_mask]
            cluster_distribution = Counter(class_clusters)
            
            print(f"   • {class_name}:")
            for cluster_id, count in sorted(cluster_distribution.items()):
                percentage = (count / len(class_clusters)) * 100
                print(f"     - Cluster {cluster_id}: {count} images ({percentage:.1f}%)")
        
        # Recommendations
        print(f"\n5. IMPROVEMENT RECOMMENDATIONS:")
        if silhouette_avg < 0.3:
            print("   • Consider changing the number of clusters")
        if ari < 0.3:
            print("   • Add more relevant features")
        if avg_purity < 0.7:
            print("   • Try different clustering algorithms")
        
        if any(purity < 0.5 for purity in cluster_purity.values()):
            print("   • Clusters contain mixed landscape types")
            print("   • More specific features are needed for better separation")
        
        print("="*60)
    
    def visualize_results(self, cluster_labels, features_scaled):
        """Enhanced visualization with multiple plots and 3D visualization"""
        print("\nCreating enhanced visualizations...")
        
        if ENHANCED_VIZ_AVAILABLE:
            # Use the enhanced visualization methods
            create_2d_scatter_plots(self, cluster_labels, features_scaled)
            create_3d_scatter_plot(self, cluster_labels, features_scaled)
            create_cluster_analysis_plots(self, cluster_labels)
            create_confusion_matrix(self, cluster_labels)
        else:
            # Fallback to basic visualization if enhanced methods are not available
            self._create_basic_visualizations(cluster_labels, features_scaled)
    
    def _create_basic_visualizations(self, cluster_labels, features_scaled):
        """Basic visualization method (fallback)"""
        print("\nCreating basic visualizations...")
        
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
        
        # Add value labels on bars
        for bar, size in zip(bars, cluster_sizes):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                          str(size), ha='center', va='bottom')
        
        # Plot 4: Cluster purity
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
        
        # Add value labels on bars
        for bar, purity in zip(bars2, purity_values):
            axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'{purity:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('clustering_results_improved.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def show_cluster_examples(self, cluster_labels, examples_per_cluster=2):
        """Show example images from clusters"""
        print("\nExample images from clusters:")
        
        unique_clusters = sorted(set(cluster_labels))
        
        for cluster_id in unique_clusters:
            print(f"\n--- Cluster {cluster_id} ---")
            
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
    """Main function with focus on 4 selected classes"""
    print("=== EARTH SURFACE IMAGE CLUSTERING (VGG16) ===")
    print("Focusing on 4 selected classes: AnnualCrop, HerbaceousVegetation, Industrial, Forest")
    print("=" * 60)
    
    if not VGG16_AVAILABLE:
        print("VGG16 is not available! Please install TensorFlow and Keras.")
        return
    
    # Initialize with more samples per class for better results
    clusterer = SimpleEarthSurfaceClustering(max_samples_per_class=50)
    
    # Load data (only selected classes)
    clusterer.load_sample_images("train.csv")
    
    # Process images
    clusterer.process_images()
    
    # Clustering
    print("\n" + "="*60)
    print("FOCUSED CLUSTERING ON 4 CLASSES")
    print("="*60)
    
    # Set n_clusters=4 to match our 4 classes
    cluster_labels, kmeans_model, features_scaled = clusterer.perform_clustering(n_clusters=4, optimize_params=True)
    
    # Detailed quality analysis
    clusterer.analyze_clustering_quality(cluster_labels)
    
    # Visualization
    clusterer.visualize_results(cluster_labels, features_scaled)
    
    # Example images
    clusterer.show_cluster_examples(cluster_labels, examples_per_cluster=4)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Results saved to: clustering_results_improved.png")

if __name__ == "__main__":
    main()