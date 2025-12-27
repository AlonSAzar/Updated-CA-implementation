import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def analyze_pca(phenotype_cache, transitions):
    # 1. Prepare Data
    # Convert dictionary values to a matrix
    # phenotype_cache is {hash: np.array([0, 1, 0...])}

    unique_hashes = list(phenotype_cache.keys())
    # Flatten images to 1D vectors
    data_matrix = np.array([phenotype_cache[h].flatten() for h in unique_hashes])

    # 2. Run PCA
    print("Running PCA on Phenotype Morphospace...")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(data_matrix)

    # 3. Calculate "Probability" (Frequency) for coloring
    # Count how often each phenotype appeared as a CHILD or PARENT
    frequencies = {h: 0 for h in unique_hashes}
    for (h_parent, h_child), count in transitions.items():
        frequencies[h_parent] += count
        frequencies[h_child] += count

    colors = [np.log10(frequencies[h] + 1) for h in unique_hashes]

    # 4. Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=colors, cmap='viridis', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Log10 Frequency (Probability)')

    plt.title(f"PCA of Phenotype Space (Rule 168)\nExplained Variance: {pca.explained_variance_ratio_}")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} var)")
    plt.grid(True, alpha=0.3)
    plt.show()

# You can call this function at the end of your main() using the existing data!
# analyze_pca(phenotype_cache, transitions)