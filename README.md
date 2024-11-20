# 2024-US-Pre-Election-Opinion-Grouping-and-Analysis

## Project Overview
. This project analyzes social media and survey data to group public opinions about the 2024 US elections. Using unsupervised machine learning techniques, we developed a model combining Topic Modeling (LDA) and KMeans Clustering to categorize different opinion groups. Additionally, Sentiment Analysis was integrated to gauge public sentiment on key election topics, providing insights into common themes and stances.
- For this project, we used the Silhouette Score and the Davies-Bouldin Index measurements:
-   `Silhouette Score`: 0.81
-   `Davies-Bouldin Index`: 0.21

![image](https://github.com/user-attachments/assets/086defec-2b08-4347-9443-4387cf0adb8b)


## Project Goals
Identify Key Themes in public discourse related to the 2024 elections.
Group Opinions into clusters to uncover distinct viewpoints.
Analyze Sentiment within each opinion group to understand public attitudes.

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

![image](https://github.com/user-attachments/assets/ceca256e-f5f6-4f8d-9b16-65909b182756)
