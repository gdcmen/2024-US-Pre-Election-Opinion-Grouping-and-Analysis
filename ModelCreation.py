import pandas as pd
import numpy as np
import re

# Data paths
preprocessed_reddit_path = "/content/drive/MyDrive/Colab Notebooks/Social Network Mining/Preprocessing/clean_reddit_dataset.csv"

# Load the data
df = pd.read_csv(preprocessed_reddit_path)

df.head()

from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

# Load a sentence embedding model (e.g., Sentence-BERT)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed the processed_text
df['text_embedding'] = df['processed_text'].apply(lambda x: model.encode(x))

# Expand the embedding into separate columns for easier manipulation
embedding_matrix = np.vstack(df['text_embedding'].values)
embedding_df = pd.DataFrame(embedding_matrix, columns=[f'embedding_{i}' for i in range(embedding_matrix.shape[1])])

# Combine all features: embeddings, compound, upvotes, and comments
df_combined = pd.concat([embedding_df, df[['compound', 'Upvotes_Normalized', 'Comments_Normalized']].reset_index(drop=True)], axis=1)

# Standardize the combined features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_combined)

# Now, df_scaled is ready for clustering
print("Data ready for clustering:")
print(df_scaled)

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Initialize variables to track the best model and parameters for each algorithm
best_score_kmeans = -1
best_kmeans_params = {}
best_kmeans_model = None

best_score_dbscan = -1
best_dbscan_params = {}
best_dbscan_model = None

# Loop over different values of eps and min_samples for DBSCAN
for eps in np.arange(0.3, 1.0, 0.1):  # Adjust range as needed
    for min_samples in range(2, 6):  # Adjust range as needed
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(df_scaled)

        # Only calculate silhouette score if we have more than one cluster
        if len(set(dbscan_labels)) > 1:
            dbscan_silhouette = silhouette_score(df_scaled, dbscan_labels)
            if dbscan_silhouette > best_score_dbscan:
                best_score_dbscan = dbscan_silhouette
                best_dbscan_model = dbscan
                best_dbscan_params = {'eps': eps, 'min_samples': min_samples}

# Loop over different values of n_clusters for KMeans
for n_clusters in range(2, 10):  # Adjust range as needed
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans_labels = kmeans.fit_predict(df_scaled)
    kmeans_silhouette = silhouette_score(df_scaled, kmeans_labels)

    if kmeans_silhouette > best_score_kmeans:
        best_score_kmeans = kmeans_silhouette
        best_kmeans_model = kmeans
        best_kmeans_params = {'n_clusters': n_clusters}

# Print the best-performing models and parameters
print("Best DBSCAN Model:")
print(f"Silhouette Score: {best_score_dbscan}")
print(f"Parameters: {best_dbscan_params}\n")

print("Best KMeans Model:")
print(f"Silhouette Score: {best_score_kmeans}")
print(f"Parameters: {best_kmeans_params}\n")

# Visualize the best models for each algorithm
# Perform clustering with best DBSCAN model
dbscan_labels = best_dbscan_model.fit_predict(df_scaled)

# Perform clustering with best KMeans model
kmeans_labels = best_kmeans_model.fit_predict(df_scaled)

# Reduce the data to 2D for visualization
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Plot DBSCAN Clusters
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
unique_labels_dbscan = set(dbscan_labels)
for label in unique_labels_dbscan:
    label_name = 'Noise' if label == -1 else f'Cluster {label}'
    plt.scatter(df_pca[dbscan_labels == label, 0], df_pca[dbscan_labels == label, 1],
                label=label_name, s=50, alpha=0.7)

plt.title(f"DBSCAN Clustering (eps={best_dbscan_params['eps']}, min_samples={best_dbscan_params['min_samples']})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()

# Plot KMeans Clusters
plt.subplot(1, 2, 2)
unique_labels_kmeans = set(kmeans_labels)
for label in unique_labels_kmeans:
    plt.scatter(df_pca[kmeans_labels == label, 0], df_pca[kmeans_labels == label, 1],
                label=f'Cluster {label}', s=50, alpha=0.7)

# Plot KMeans cluster centers
centers_pca = pca.transform(best_kmeans_model.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], s=200, c='red', marker='X', label='Centers', edgecolor='k')
plt.title(f"KMeans Clustering (n_clusters={best_kmeans_params['n_clusters']})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()

plt.suptitle("Comparison of DBSCAN and KMeans Clustering")
plt.show()

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Initialize variables to track the best model and parameters
best_kmeans_score = -1
best_kmeans_model = None
best_kmeans_labels = None
best_n_clusters = 0
best_num_topics = 0
results = []  # Store results for plotting

for num_topics in range(2, 8):  # Adjust range as needed

    # Step 1: Preprocess text
    # CountVectorizer will handle tokenization, lowercase, and remove stop words
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['processed_text'])

    # Step 2: Apply LDA for Topic Modeling
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    topic_distributions = lda.fit_transform(X)

    # Step 3: Use Topic Distributions as Features for Clustering
    # Normalize topic distributions
    scaler = StandardScaler()
    topic_distributions_scaled = scaler.fit_transform(topic_distributions)

    # Test different values of k for KMeans
    for n_clusters in range(2, 6):  # Adjust range as needed
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans_labels = kmeans.fit_predict(topic_distributions_scaled)
        silhouette_avg = silhouette_score(topic_distributions_scaled, kmeans_labels)

        # Store the result for plotting
        results.append({
            'num_topics': num_topics,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_avg
        })

        # Update best model if current silhouette score is the best
        if silhouette_avg > best_kmeans_score:
            best_kmeans_score = silhouette_avg
            best_kmeans_model = kmeans
            best_kmeans_labels = kmeans_labels
            best_n_clusters = n_clusters
            best_num_topics = num_topics
            best_topic_distributions_scaled = topic_distributions_scaled

# Print the best KMeans model and its silhouette score
print(f"Best KMeans Model with n_clusters={best_n_clusters} and num_topics={best_num_topics}")
print(f"Silhouette Score: {best_kmeans_score}")

# Assign clusters to the original data
df['Cluster'] = best_kmeans_labels

# Display the clustered data
print("Clustered Data:")
print(df[['processed_text', 'Cluster']])

# Plot the silhouette scores for different configurations
results_df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
for num_topics in results_df['num_topics'].unique():
    subset = results_df[results_df['num_topics'] == num_topics]
    plt.plot(subset['n_clusters'], subset['silhouette_score'], marker='o', label=f'Topics: {num_topics}')

plt.title("Silhouette Score for Different Numbers of Topics and Clusters")
plt.xlabel("Number of Clusters (n_clusters)")
plt.ylabel("Silhouette Score")
plt.legend(title="Number of Topics")
plt.grid(True)
plt.show()

from sklearn.metrics import silhouette_score, davies_bouldin_score

# Step 3: Calculate Silhouette Score
silhouette_avg = silhouette_score(best_topic_distributions_scaled, best_kmeans_labels)
print(f"Silhouette Score for KMeans: {silhouette_avg}")

# Step 4: Calculate Davies-Bouldin Index
davies_bouldin_avg = davies_bouldin_score(best_topic_distributions_scaled, best_kmeans_labels)
print(f"Davies-Bouldin Index for KMeans: {davies_bouldin_avg}")

pca = PCA(n_components=2)
topic_distributions_pca = pca.fit_transform(best_topic_distributions_scaled)

# Plot the clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(topic_distributions_pca[:, 0], topic_distributions_pca[:, 1],
                      c=best_kmeans_labels, cmap='viridis', s=50, alpha=0.7)
plt.title(f"KMeans Clustering (n_clusters={n_clusters})\nSilhouette Score: {silhouette_avg:.2f}, Davies-Bouldin Index: {davies_bouldin_avg:.2f}")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

# Add cluster centers if available
centers_pca = pca.transform(best_kmeans_model.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], s=200, c='red', marker='X', edgecolor='k', label='Centers')

# Add legend for clusters
legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
plt.gca().add_artist(legend1)
plt.legend(['Centers'], loc='upper right')

plt.show()


# Code to save and load the model

from joblib import dump, load

# Save the best KMeans model to an .h5 file
dump(best_kmeans_model, '/content/drive/MyDrive/Colab Notebooks/Social Network Mining/ModelCreation/best_kmeans_model.h5')

print("KMeans model saved as best_kmeans_model.h5")

# Load the KMeans model
best_kmeans_model = load('/content/drive/MyDrive/Colab Notebooks/Social Network Mining/ModelCreation/best_kmeans_model.h5')

print("KMeans model loaded successfully")
