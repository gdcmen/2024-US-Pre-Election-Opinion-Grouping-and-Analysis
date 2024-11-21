# 2024-US-Pre-Election-Opinion-Grouping-and-Analysis

## Project Overview
This project analyzes public opinions about the 2024 US elections using unsupervised machine learning on social media data. We developed a hybrid model combining Topic Modeling (LDA) and KMeans Clustering to identify distinct opinion groups. Sentiment Analysis was also applied to measure public sentiment on key topics, revealing insights into common themes and stances.
- For this project, we used the Silhouette Score and the Davies-Bouldin Index measurements:
-   `Silhouette Score`: 0.81
-   `Davies-Bouldin Index`: 0.21

## Project Goals
1. Identify Key Themes in public discourse about the 2024 elections.
2. Group Public Opinions into distinct clusters to uncover diverse viewpoints.
3. Analyze Sentiments within clusters to evaluate public attitudes on critical election topics.

## Key Insights
### Impact of Clusters and Topics
We experimented with various numbers of clusters and topics:

1. Fewer Clusters: Higher Silhouette Scores, leading to more distinct groupings but broader themes.
2. Three Clusters: Slightly lower Silhouette Scores but provided deeper insights into specific conversations.
3. Larger Numbers of Clusters: Enabled finer segmentation of opinions but with diminishing returns on interpretability.

The following graph illustrates the trade-off between the number of clusters and Silhouette Score across different numbers of topics:

![image](https://github.com/user-attachments/assets/ceca256e-f5f6-4f8d-9b16-65909b182756)


## Skills and Tools
Python: Data manipulation, modeling, and analysis.
NLP Techniques: Topic Modeling (LDA), Sentiment Analysis.
Clustering: KMeans for grouping opinions.
Visualization: Matplotlib, Seaborn for data insights.
Libraries: scikit-learn, NLTK, Gensim, and Sentence Transformers.

## Dataset
This project uses anonymized social media data (reddit), preprocessed to include the following features:

Text Data: Public opinions on the 2024 US elections.
Sentiment Scores: Sentiment analysis of opinions.
Topic Distributions: Topic weights derived from LDA modeling.

## Methodology

### 1. Data Preprocessing:
  - Text cleaning and preprocessing.
  - Tokenization and vectorization with CountVectorizer.

### 2. Topic Modeling:
  - Applied Latent Dirichlet Allocation (LDA) to extract key topics from the text data.

### 3. Clustering:
  - Used KMeans Clustering on the LDA topic distributions to group similar opinions.
  - Experimented with different numbers of clusters and topics for optimal results.

### 4. Sentiment Analysis:
  - Integrated sentiment analysis scores to assess the emotional tone within each cluster.
  - This measure, while it was very valuable to understand the data, was not used to create the ML model. Poor results where obtained when using it.

### 5. Evaluation:
  - Evaluated clustering performance using Silhouette Score and Davies-Bouldin Index.
  - Visualized the relationship between the number of topics and clusters to determine the best configuration.

## Results
  - Identified Clusters: Key opinion groups were identified, revealing diverse perspectives on the election.
  - Sentiment Insights: Sentiment analysis highlighted varying attitudes towards specific topics within each cluster.

