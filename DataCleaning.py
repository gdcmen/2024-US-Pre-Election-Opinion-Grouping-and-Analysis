import praw
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import csv

# Set up Reddit API credentials

reddit_file = "/content/drive/MyDrive/Colab Notebooks/Social Network Mining/reddit_data.csv"

# Check if authenticated and print the username
try:
    user = reddit.user.me()
    if user == None:
      print("Authenticated as Read Only!")
    else:
      print(f"Authenticated as: {user}")
except Exception as e:
    print(f"Authentication failed: {e}")

# Function to fetch Reddit posts and perform sentiment analysis
def fetch_reddit_posts(subreddit_name, keyword, max_posts=100):
    # Initialize sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    with open(reddit_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Post ID', 'Title', 'Selftext', 'Upvotes', 'Comments', 'Sentiment'])

        # Fetch posts from the given subreddit
        subreddit = reddit.subreddit(subreddit_name)
        posts = subreddit.search(keyword, limit=max_posts)

        post_count = 0
        for post in posts:
            post_id = post.id
            title = post.title
            selftext = post.selftext if post.selftext else "No content"
            upvotes = post.score
            comments = post.num_comments

            # Sentiment analysis on the title + selftext
            sentiment = analyzer.polarity_scores(title + " " + selftext)['compound']

            # Save the post details to a list and write to CSV
            writer.writerow([post_id, title, selftext, upvotes, comments, sentiment])
            post_count += 1

            if post_count >= max_posts:
                break

        print(f"Fetched {post_count} posts from r/{subreddit_name} with the keyword '{keyword}'")
        print(f"Data stored in {reddit_file}")

# Create a list of potential subreddit names

subreddits = ['politics', 'USPolitics', 'Conservative', 'Republican', 'Liberal', 'Democrats', 'Progressive', 'Election2024']
keywords = ["Election2024", "US election", "Vote", "Presidential election", "Kamala Harris", "Donald Trump", "Radical left", "Alt-right", "Radical Right", "Abortion"]

# Fetch posts about the US elections from a specific subreddit
for sr in subreddits:
  for k in keywords:
    fetch_reddit_posts(sr, k, max_posts=1000)

"""#DATA PREPROCESSING"""

import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = "/content/drive/MyDrive/Colab Notebooks/Social Network Mining/Preprocessing/reddit_data.csv"
reddit_data = pd.read_csv(file_path)

# Handling missing values
reddit_data['Title'] = reddit_data['Title'].fillna('')
reddit_data['Selftext'] = reddit_data['Selftext'].fillna('')

# Define a basic text cleaning function
def basic_clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    tokens = text.split()  # Tokenize the text
    return ' '.join(tokens)

# Apply the basic cleaning function to the 'Title' and 'Selftext' columns
reddit_data['Cleaned_Title'] = reddit_data['Title'].apply(basic_clean_text)
reddit_data['Cleaned_Selftext'] = reddit_data['Selftext'].apply(basic_clean_text)

# Normalize 'Upvotes' and 'Comments' using MinMaxScaler
scaler = MinMaxScaler()

reddit_data[['Upvotes', 'Comments']] = reddit_data[['Upvotes', 'Comments']].apply(pd.to_numeric, errors='coerce')
reddit_data[['Upvotes', 'Comments']] = reddit_data[['Upvotes', 'Comments']].fillna(0)

reddit_data[['Upvotes_Normalized', 'Comments_Normalized']] = scaler.fit_transform(reddit_data[['Upvotes', 'Comments']])

# Save the cleaned and processed data
reddit_data.to_csv('cleaned_reddit_data.csv', index=False)

file_path = "/content/drive/MyDrive/Colab Notebooks/Social Network Mining/reddit_data.csv"
no_clean_reddit_data = pd.read_csv(file_path)

no_clean_reddit_data.head()

# Display the cleaned dataset
cleaned_reddit_data = reddit_data[['Post ID', 'Cleaned_Title', 'Cleaned_Selftext', 'Upvotes_Normalized', 'Comments_Normalized', 'Sentiment']]
cleaned_reddit_data.head()

cleaned_reddit_data.shape

"""Save the model into a new dataset"""

cleaned_reddit_data.to_csv('/content/drive/MyDrive/Colab Notebooks/Social Network Mining/Preprocessing/claned_reddit_data.csv', index=False)

cleaned_reddit_data.head()

"""# USING NLTK FOR SENTIMENT ANALYSIS"""

!pip install nltk

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
nltk.download('vader_lexicon')

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Social Network Mining/Preprocessing/reddit_data.csv')
df.drop(df[df['Selftext'].str.lower() == 'no content'].index, inplace=True)

df.head()

for row in df.itertuples(index=True, name='Pandas'):
    print(f"Index: {row.Index}, Title: {row.Title}, Selftext: {row.Selftext}, Vader Sentiment: {row.Sentiment}")
    break


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('vader_lexicon')


def clean_text(text):
    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Step 3: Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # Step 4: Tokenization
    words = nltk.word_tokenize(text)

    # Step 5: Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Step 6: Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Step 7: Join words back into a cleaned string
    cleaned_text = ' '.join(words)

    return cleaned_text

sentiment_analyzer = SentimentIntensityAnalyzer()

# Combine 'Title' and 'Selftext' into a new column 'processed_text'
df['processed_text'] = df['Title'] + ' ' + df['Selftext']
# Drop rows with NaN or non-string values in 'processed_text'
df = df.dropna(subset=['processed_text'])
df['processed_text'] = df['processed_text'].astype(str)  # Convert all to strings
df['cleaned_text'] = df['processed_text'].apply(clean_text)

# Initialize empty lists to store sentiment scores
nltk_sentiments = []
compounds = []

# Apply Vader Analysis
for row in df.itertuples(index=True, name='Pandas'):
    # Get the processed_text for the current row, converting non-string values to empty strings
    text = row.processed_text if isinstance(row.processed_text, str) else ""

    # Get the sentiment scores for the current row
    scores = sentiment_analyzer.polarity_scores(text)

    # Append scores to the lists
    nltk_sentiments.append(scores)
    compounds.append(scores['compound'])

# Assign the results back to the DataFrame
df['NLTK_sentiment'] = nltk_sentiments
df['compound'] = compounds

print(type(text))
print(type(scores), "Scores:", scores)

print('Sentiment Analysis finished!')

# Convert columns to numeric, setting errors='coerce' to handle non-numeric values by converting them to NaN
df['Sentiment'] = pd.to_numeric(df['Sentiment'], errors='coerce')
df['compound'] = pd.to_numeric(df['compound'], errors='coerce')

# Drop any rows with NaN values in either column (optional)
df.dropna(subset=['Sentiment', 'compound'], inplace=True)

# Calculate the absolute difference between the two columns
df['difference'] = abs(df['Sentiment'] - df['compound'])

# Calculate the mean of the differences
average_difference = df['difference'].mean()

print("Average Difference between Sentiment and Compound:", average_difference)
df['difference']
print(df)

"""# Results Analysis

Some difference between sentiment models have been noted, after some research and measuring the data, it has been decided to go with NLTK for sentiment analysis
"""

from sklearn.preprocessing import MinMaxScaler

# Normalize 'Upvotes' and 'Comments' using MinMaxScaler
scaler = MinMaxScaler()

df[['Upvotes', 'Comments']] = df[['Upvotes', 'Comments']].apply(pd.to_numeric, errors='coerce')
df[['Upvotes', 'Comments']] = df[['Upvotes', 'Comments']].fillna(0)

df[['Upvotes_Normalized', 'Comments_Normalized']] = scaler.fit_transform(df[['Upvotes', 'Comments']])

df.head()

final_df = df[["processed_text", "Upvotes_Normalized", "Comments_Normalized", "compound"]]

final_df.head()

df.to_csv('/content/drive/MyDrive/Colab Notebooks/Social Network Mining/Preprocessing/full_reddit_dataset.csv', index=False)
final_df.to_csv('/content/drive/MyDrive/Colab Notebooks/Social Network Mining/Preprocessing/clean_reddit_dataset.csv', index=False)
