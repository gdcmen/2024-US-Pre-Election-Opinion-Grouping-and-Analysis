import pandas as pd
import numpy as np
import re

# Data paths
preprocessed_reddit_path = "/content/drive/MyDrive/Colab Notebooks/Social Network Mining/Preprocessing/clean_reddit_dataset.csv"

# Load the data
df = pd.read_csv(preprocessed_reddit_path)

df.head()

from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Initialize variables to track the best model and parameters
best_kmeans_score = -1
best_kmeans_model = None
best_kmeans_labels = None
best_n_clusters = 0
best_num_topics = 0
results = []  # Store results for plotting
best_vectorizer = None

for num_topics in range(2, 8):  # Adjust range as needed

    # Step 1: Preprocess text
    # CountVectorizer will handle tokenization, lowercase, and remove stop words
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['cleaned_text'])

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
            best_vectorizer = vectorizer
            best_lda = lda
            best_scaler = scaler

# Print the best KMeans model and its silhouette score
print(f"Best KMeans Model with n_clusters={best_n_clusters} and num_topics={best_num_topics}")
print(f"Silhouette Score: {best_kmeans_score}")

# Assign clusters to the original data
df['Cluster'] = best_kmeans_labels

# Display the clustered data
print("Clustered Data:")
print(df[['cleaned_text', 'Cluster']])

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

print(f"Cluster Distribution in Training Data: {np.bincount(best_kmeans_labels)}")

# Print the best KMeans model and its silhouette score
print(f"Best KMeans Model with n_clusters={best_n_clusters} and num_topics={best_num_topics}")
print(f"Silhouette Score: {best_kmeans_score}")

# Assign clusters to the original data
df['Cluster'] = best_kmeans_labels

# Display 10 examples from each cluster
print("\nExamples from each cluster:\n")
for cluster in range(best_n_clusters):
    print(f"Cluster {cluster}:")
    cluster_examples = df[df['Cluster'] == cluster].head(2)  # Get the first 10 examples from this cluster
    for i, row in cluster_examples.iterrows():
        print(f"- {row['cleaned_text']}")
    print("\n" + "-" * 50 + "\n")

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
topic_distributions_pca = pca.fit_transform(best_topic_distributions_scaled)

# Plot the clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(topic_distributions_pca[:, 0], topic_distributions_pca[:, 1],
                      c=best_kmeans_labels, cmap='viridis', s=50, alpha=0.7)
plt.title(f"KMeans Clustering (n_clusters={best_n_clusters})\nSilhouette Score: {silhouette_avg:.2f}, Davies-Bouldin Index: {davies_bouldin_avg:.2f}")
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

from joblib import dump, load

# Save the best KMeans model to an .h5 file
dump(best_kmeans_model, '/content/drive/MyDrive/Colab Notebooks/Social Network Mining/ModelCreation/best_kmeans_model.h5')
# Save the trained vectorizer
dump(vectorizer, '/content/drive/MyDrive/Colab Notebooks/Social Network Mining/ModelCreation/vectorizer.pkl')
# Save the LDA model
dump(best_lda, '/content/drive/MyDrive/Colab Notebooks/Social Network Mining/ModelCreation/lda.pkl')
# Save the Scaler model
dump(best_scaler, '/content/drive/MyDrive/Colab Notebooks/Social Network Mining/ModelCreation/scaler.pkl')

print("KMeans model saved as best_kmeans_model.h5")
print("Vectorizer saved successfully!")
print("LDA model saved successfully!")
print("Scaler model saved successfully!")
