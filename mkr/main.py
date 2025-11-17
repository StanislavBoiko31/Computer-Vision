import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import time

image_path = 'dj-paine-JGcdjvwA14Q-unsplash.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


height, width = image_rgb.shape[:2]
if width > 800:
    scale = 800 / width
    new_width = int(width * scale)
    new_height = int(height * scale)
    image_rgb = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)

print(f"Оригінальний розмір зображення: {image_rgb.shape}")


print("\n=== Застосування попередньої обробки ===")

PREPROCESSING_CONFIG = {
    'apply_gaussian_blur': True,      
    'gaussian_kernel_size': 5,        
    'gaussian_sigma': 1.5,            
    
    'apply_bilateral_filter': False,   
    'bilateral_d': 9,                 
    'bilateral_sigma_color': 75,      
    'bilateral_sigma_space': 75,      
    
    'apply_contrast_enhancement': True, 
    'contrast_alpha': 1.2,            
    'contrast_beta': 10,              
    
    'apply_adaptive_histogram': False, 
    'clahe_clip_limit': 2.0,          
    'clahe_tile_size': (8, 8),        
    
    'apply_morphology': False,        
    'morphology_operation': 'closing', 
    'morphology_kernel_size': 3,      
    
    'color_space': 'LAB',             
    'apply_histogram_equalization': False,  
    
    'apply_median_filter': False,     
    'median_kernel_size': 5,          
}

image_original = image_rgb.copy()

if PREPROCESSING_CONFIG['apply_gaussian_blur']:
    kernel_size = PREPROCESSING_CONFIG['gaussian_kernel_size']
    sigma = PREPROCESSING_CONFIG['gaussian_sigma']
    image_processed = cv2.GaussianBlur(image_rgb, (kernel_size, kernel_size), sigma)
    print(f"✓ Застосовано Gaussian Blur (kernel={kernel_size}, sigma={sigma})")
else:
    image_processed = image_rgb.copy()

if PREPROCESSING_CONFIG['apply_bilateral_filter']:
    d = PREPROCESSING_CONFIG['bilateral_d']
    sigma_color = PREPROCESSING_CONFIG['bilateral_sigma_color']
    sigma_space = PREPROCESSING_CONFIG['bilateral_sigma_space']
    image_processed = cv2.bilateralFilter(image_processed, d, sigma_color, sigma_space)
    print(f"✓ Застосовано Bilateral Filter (d={d}, sigma_color={sigma_color}, sigma_space={sigma_space})")

if PREPROCESSING_CONFIG['apply_contrast_enhancement']:
    alpha = PREPROCESSING_CONFIG['contrast_alpha']
    beta = PREPROCESSING_CONFIG['contrast_beta']
    image_processed = cv2.convertScaleAbs(image_processed, alpha=alpha, beta=beta)
    print(f"✓ Покращено контраст (alpha={alpha}, beta={beta})")

color_space = PREPROCESSING_CONFIG['color_space']
if color_space == 'LAB':
    image_processed = cv2.cvtColor(image_processed, cv2.COLOR_RGB2LAB)
    print(f"✓ Конвертовано в LAB кольоровий простір")
elif color_space == 'HSV':
    image_processed = cv2.cvtColor(image_processed, cv2.COLOR_RGB2HSV)
    print(f"✓ Конвертовано в HSV кольоровий простір")
elif color_space == 'LUV':
    image_processed = cv2.cvtColor(image_processed, cv2.COLOR_RGB2LUV)
    print(f"✓ Конвертовано в LUV кольоровий простір")
else:
    print(f"✓ Використовується RGB кольоровий простір")

if PREPROCESSING_CONFIG['apply_median_filter']:
    kernel_size = PREPROCESSING_CONFIG['median_kernel_size']
    image_processed = cv2.medianBlur(image_processed, kernel_size)
    print(f"✓ Застосовано медіанний фільтр (kernel={kernel_size})")

if PREPROCESSING_CONFIG['apply_adaptive_histogram'] and color_space == 'RGB':
    if len(image_processed.shape) == 3:
        if image_processed.dtype != np.uint8:
            image_processed = np.clip(image_processed, 0, 255).astype(np.uint8)
        
        clahe = cv2.createCLAHE(
            clipLimit=PREPROCESSING_CONFIG['clahe_clip_limit'],
            tileGridSize=PREPROCESSING_CONFIG['clahe_tile_size']
        )
        processed_channels = []
        for i in range(image_processed.shape[2]):
            processed_channels.append(clahe.apply(image_processed[:,:,i]))
        image_processed = np.stack(processed_channels, axis=2)
    else:
        if image_processed.dtype != np.uint8:
            image_processed = np.clip(image_processed, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(
            clipLimit=PREPROCESSING_CONFIG['clahe_clip_limit'],
            tileGridSize=PREPROCESSING_CONFIG['clahe_tile_size']
        )
        image_processed = clahe.apply(image_processed)
    print(f"✓ Застосовано адаптивне вирівнювання гістограми (CLAHE)")

if PREPROCESSING_CONFIG['apply_histogram_equalization'] and color_space == 'LAB':
    lab = image_processed.copy()
    lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
    image_processed = lab
    print(f"✓ Застосовано вирівнювання гістограми для L каналу")

if PREPROCESSING_CONFIG['apply_morphology']:
    kernel_size = PREPROCESSING_CONFIG['morphology_kernel_size']
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    operation = PREPROCESSING_CONFIG['morphology_operation']
    
    if operation == 'opening':
        image_processed = cv2.morphologyEx(image_processed, cv2.MORPH_OPEN, kernel)
        print(f"✓ Застосовано морфологічне відкриття (kernel={kernel_size})")
    elif operation == 'closing':
        image_processed = cv2.morphologyEx(image_processed, cv2.MORPH_CLOSE, kernel)
        print(f"✓ Застосовано морфологічне закриття (kernel={kernel_size})")
    elif operation == 'gradient':
        image_processed = cv2.morphologyEx(image_processed, cv2.MORPH_GRADIENT, kernel)
        print(f"✓ Застосовано морфологічний градієнт (kernel={kernel_size})")

if color_space != 'RGB':
    if color_space == 'LAB':
        image_processed_rgb = cv2.cvtColor(image_processed, cv2.COLOR_LAB2RGB)
    elif color_space == 'HSV':
        image_processed_rgb = cv2.cvtColor(image_processed, cv2.COLOR_HSV2RGB)
    elif color_space == 'LUV':
        image_processed_rgb = cv2.cvtColor(image_processed, cv2.COLOR_LUV2RGB)
    else:
        image_processed_rgb = image_processed
else:
    image_processed_rgb = image_processed

image_processed_rgb = np.clip(image_processed_rgb, 0, 255).astype(np.uint8)

print(f"Розмір обробленого зображення: {image_processed_rgb.shape}\n")

image_rgb = image_processed_rgb

height, width = image_rgb.shape[:2]
pixels = image_rgb.reshape(-1, 3)

y_coords, x_coords = np.mgrid[0:height, 0:width]
coords = np.column_stack([x_coords.ravel(), y_coords.ravel()])

coords_normalized = coords / [width, height]


spatial_weight = 0.3
features = np.hstack([
    pixels / 255.0,  
    coords_normalized * spatial_weight
])

print(f"Кількість пікселів: {len(features)}")
print(f"Розмірність ознак: {features.shape[1]}")

def visualize_clustering(image, labels, title, ax, draw_contours=True):
    height, width = image.shape[:2]
    labels_2d = labels.reshape(height, width)
    
    segmented = image.copy().astype(np.float32)
    
    unique_labels = np.unique(labels)
    cluster_colors = {}
    
    for label in unique_labels:
        if label == -1:  
            mask = labels_2d == label
            cluster_colors[label] = np.array([128, 128, 128], dtype=np.uint8)
        else:
            mask = labels_2d == label
            cluster_pixels = image[mask]
            if len(cluster_pixels) > 0:
                mean_color = cluster_pixels.mean(axis=0).astype(np.uint8)
                cluster_colors[label] = mean_color
                alpha = 0.6
                segmented[mask] = segmented[mask] * (1 - alpha) + mean_color * alpha
    
    segmented = segmented.astype(np.uint8)
    
    if draw_contours:
        n_valid_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        contour_colors = plt.cm.tab20(np.linspace(0, 1, max(n_valid_clusters, 1)))
        
        color_idx = 0
        for label in unique_labels:
            if label == -1:
                continue
            mask = (labels_2d == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            color_rgb = (contour_colors[color_idx][:3] * 255).astype(np.uint8)
            color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
            
            cv2.drawContours(segmented, contours, -1, color_bgr, 3)
            dark_color = (int(max(0, color_bgr[0] - 50)), 
                         int(max(0, color_bgr[1] - 50)), 
                         int(max(0, color_bgr[2] - 50)))
            cv2.drawContours(segmented, contours, -1, dark_color, 1)
            
            color_idx += 1
    
    ax.imshow(segmented)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(labels == -1) if -1 in unique_labels else 0
    info_text = f"Кластерів: {n_clusters}"
    if n_noise > 0:
        info_text += f"\nШум: {n_noise} пікселів"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

fig, axes = plt.subplots(3, 3, figsize=(20, 18))
preprocessing_info = []
if PREPROCESSING_CONFIG['apply_gaussian_blur']:
    preprocessing_info.append(f"Gaussian Blur")
if PREPROCESSING_CONFIG['apply_bilateral_filter']:
    preprocessing_info.append(f"Bilateral")
if PREPROCESSING_CONFIG['apply_median_filter']:
    preprocessing_info.append(f"Median")
if PREPROCESSING_CONFIG['apply_contrast_enhancement']:
    preprocessing_info.append(f"Contrast↑")
if PREPROCESSING_CONFIG['apply_adaptive_histogram']:
    preprocessing_info.append(f"CLAHE")
if PREPROCESSING_CONFIG['apply_histogram_equalization']:
    preprocessing_info.append(f"HistEq")
if PREPROCESSING_CONFIG['apply_morphology']:
    preprocessing_info.append(f"Morph({PREPROCESSING_CONFIG['morphology_operation']})")
if PREPROCESSING_CONFIG['color_space'] != 'RGB':
    preprocessing_info.append(f"{PREPROCESSING_CONFIG['color_space']}")
title_suffix = f" ({', '.join(preprocessing_info)})" if preprocessing_info else ""
fig.suptitle(f'Порівняння алгоритмів кластеризації з обведенням кластерів{title_suffix}', 
             fontsize=18, fontweight='bold')

axes[0, 0].imshow(image_original)
axes[0, 0].set_title('Оригінальне зображення', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

if any([PREPROCESSING_CONFIG['apply_gaussian_blur'], 
        PREPROCESSING_CONFIG['apply_bilateral_filter'],
        PREPROCESSING_CONFIG['apply_contrast_enhancement'],
        PREPROCESSING_CONFIG['apply_adaptive_histogram'],
        PREPROCESSING_CONFIG['apply_morphology'],
        PREPROCESSING_CONFIG['apply_median_filter'],
        PREPROCESSING_CONFIG['apply_histogram_equalization'],
        PREPROCESSING_CONFIG['color_space'] != 'RGB']):
    axes[0, 1].imshow(image_rgb)
    axes[0, 1].set_title('Оброблене зображення', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    algorithm_positions = {
        'kmeans': (0, 2),
        'meanshift': (1, 0),
        'dbscan': (1, 1),
        'agg': (1, 2),
        'gmm': (2, 0),
        'spectral': (2, 1)
    }
else:
    algorithm_positions = {
        'kmeans': (0, 1),
        'meanshift': (0, 2),
        'dbscan': (1, 0),
        'agg': (1, 1),
        'gmm': (1, 2),
        'spectral': (2, 0)
    }

results = {}

print("\n=== K-Means кластеризація ===")
n_clusters = 5  
start_time = time.time()

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(features)

kmeans_time = time.time() - start_time
print(f"Час виконання: {kmeans_time:.2f} секунд")
print(f"Кількість кластерів: {n_clusters}")

visualize_clustering(image_rgb, kmeans_labels, f'K-Means (k={n_clusters})', axes[algorithm_positions['kmeans']])
results['K-Means'] = {'time': kmeans_time, 'clusters': n_clusters, 'labels': kmeans_labels}

print("\n=== Mean Shift кластеризація ===")
sample_size = min(5000, len(features))
sample_indices = np.random.choice(len(features), sample_size, replace=False)
features_sample = features[sample_indices]

start_time = time.time()

meanshift = MeanShift(bandwidth=None, bin_seeding=True, n_jobs=-1)
meanshift_labels_sample = meanshift.fit_predict(features_sample)

meanshift_time = time.time() - start_time
print(f"Час виконання: {meanshift_time:.2f} секунд")
n_clusters_ms = len(np.unique(meanshift_labels_sample))
print(f"Знайдено кластерів: {n_clusters_ms}")

nn = NearestNeighbors(n_neighbors=1)
nn.fit(features_sample)
_, indices = nn.kneighbors(features)
meanshift_labels = meanshift_labels_sample[indices.flatten()]

visualize_clustering(image_rgb, meanshift_labels, 
                    f'Mean Shift (кластерів: {n_clusters_ms})', 
                    axes[algorithm_positions['meanshift']])
results['Mean Shift'] = {'time': meanshift_time, 'clusters': n_clusters_ms, 'labels': meanshift_labels}

print("\n=== DBSCAN кластеризація ===")
sample_size_dbscan = min(10000, len(features))
sample_indices_dbscan = np.random.choice(len(features), sample_size_dbscan, replace=False)
features_sample_dbscan = features[sample_indices_dbscan]

start_time = time.time()

eps = 0.15  
min_samples = 50  

dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
dbscan_labels_sample = dbscan.fit_predict(features_sample_dbscan)

dbscan_time = time.time() - start_time
print(f"Час виконання: {dbscan_time:.2f} секунд")
n_clusters_dbscan = len(set(dbscan_labels_sample)) - (1 if -1 in dbscan_labels_sample else 0)
n_noise = np.sum(dbscan_labels_sample == -1)
print(f"Знайдено кластерів: {n_clusters_dbscan}")
print(f"Точок шуму: {n_noise}")

nn_dbscan = NearestNeighbors(n_neighbors=1)
nn_dbscan.fit(features_sample_dbscan)
_, indices_dbscan = nn_dbscan.kneighbors(features)
dbscan_labels = dbscan_labels_sample[indices_dbscan.flatten()]

visualize_clustering(image_rgb, dbscan_labels, 
                    f'DBSCAN (eps={eps}, min_samples={min_samples})', 
                    axes[algorithm_positions['dbscan']])
results['DBSCAN'] = {'time': dbscan_time, 'clusters': n_clusters_dbscan, 'labels': dbscan_labels, 'noise': n_noise}

print("\n=== Agglomerative Clustering кластеризація ===")
n_clusters_agg = 6
start_time = time.time()

sample_size_agg = min(8000, len(features))
sample_indices_agg = np.random.choice(len(features), sample_size_agg, replace=False)
features_sample_agg = features[sample_indices_agg]

agg_clustering = AgglomerativeClustering(n_clusters=n_clusters_agg, linkage='ward')
agg_labels_sample = agg_clustering.fit_predict(features_sample_agg)

agg_time = time.time() - start_time
print(f"Час виконання: {agg_time:.2f} секунд")
print(f"Кількість кластерів: {n_clusters_agg}")

nn_agg = NearestNeighbors(n_neighbors=1)
nn_agg.fit(features_sample_agg)
_, indices_agg = nn_agg.kneighbors(features)
agg_labels = agg_labels_sample[indices_agg.flatten()]

visualize_clustering(image_rgb, agg_labels, 
                    f'Agglomerative (k={n_clusters_agg}, ward)', 
                    axes[algorithm_positions['agg']])
results['Agglomerative'] = {'time': agg_time, 'clusters': n_clusters_agg, 'labels': agg_labels}

print("\n=== Gaussian Mixture Model кластеризація ===")
n_components_gmm = 5
start_time = time.time()

gmm = GaussianMixture(n_components=n_components_gmm, random_state=42, max_iter=100)
gmm_labels = gmm.fit_predict(features)

gmm_time = time.time() - start_time
print(f"Час виконання: {gmm_time:.2f} секунд")
print(f"Кількість компонентів: {n_components_gmm}")

visualize_clustering(image_rgb, gmm_labels, 
                    f'Gaussian Mixture (n={n_components_gmm})', 
                    axes[algorithm_positions['gmm']])
results['GMM'] = {'time': gmm_time, 'clusters': n_components_gmm, 'labels': gmm_labels}

print("\n=== Spectral Clustering кластеризація ===")
n_clusters_spec = 5
start_time = time.time()

sample_size_spec = min(6000, len(features))
sample_indices_spec = np.random.choice(len(features), sample_size_spec, replace=False)
features_sample_spec = features[sample_indices_spec]

spectral = SpectralClustering(n_clusters=n_clusters_spec, random_state=42, n_jobs=-1, affinity='rbf', gamma=1.0)
spectral_labels_sample = spectral.fit_predict(features_sample_spec)

spectral_time = time.time() - start_time
print(f"Час виконання: {spectral_time:.2f} секунд")
print(f"Кількість кластерів: {n_clusters_spec}")

nn_spec = NearestNeighbors(n_neighbors=1)
nn_spec.fit(features_sample_spec)
_, indices_spec = nn_spec.kneighbors(features)
spectral_labels = spectral_labels_sample[indices_spec.flatten()]

visualize_clustering(image_rgb, spectral_labels, 
                    f'Spectral Clustering (k={n_clusters_spec})', 
                    axes[algorithm_positions['spectral']])
results['Spectral'] = {'time': spectral_time, 'clusters': n_clusters_spec, 'labels': spectral_labels}

occupied_positions = set([
    (0, 0),  
    algorithm_positions['kmeans'],
    algorithm_positions['meanshift'],
    algorithm_positions['dbscan'],
    algorithm_positions['agg'],
    algorithm_positions['gmm'],
    algorithm_positions['spectral']
])
if any([PREPROCESSING_CONFIG['apply_gaussian_blur'], 
        PREPROCESSING_CONFIG['apply_bilateral_filter'],
        PREPROCESSING_CONFIG['apply_contrast_enhancement'],
        PREPROCESSING_CONFIG['color_space'] != 'RGB']):
    occupied_positions.add((0, 1))  

for i in range(3):
    for j in range(3):
        if (i, j) not in occupied_positions:
            axes[i, j].axis('off')

plt.tight_layout()
plt.savefig('clustering_comparison.png', dpi=150, bbox_inches='tight')
print("\n=== Результати збережено у файл 'clustering_comparison.png' ===")

print("\n" + "="*80)
print("ПОРІВНЯЛЬНА ТАБЛИЦЯ")
print("="*80)
print(f"{'Алгоритм':<20} {'Кластерів':<12} {'Час (сек)':<12} {'Особливості'}")
print("-"*80)

for name, data in results.items():
    clusters = data['clusters']
    time_val = data['time']
    if name == 'K-Means':
        features_desc = 'Фіксована кількість, швидкий'
    elif name == 'Mean Shift':
        features_desc = 'Автоматичне визначення кількості'
    elif name == 'DBSCAN':
        noise = data.get('noise', 0)
        features_desc = f'Виявляє шум ({noise} пікс.), довільна форма'
    elif name == 'Agglomerative':
        features_desc = 'Ієрархічна, ward linkage'
    elif name == 'GMM':
        features_desc = 'Ймовірнісна модель, м\'які кластери'
    elif name == 'Spectral':
        features_desc = 'На основі спектра графа'
    else:
        features_desc = '-'
    
    print(f"{name:<20} {clusters:<12} {time_val:<12.2f} {features_desc}")

print("="*80)
print("\nВізуалізація:")
print("- Кластери виділені кольором з прозорістю 60%")
print("- Кожен кластер обведений унікальним яскравим кольором")
print("- Шум (для DBSCAN) показано сірим кольором")

plt.show()

