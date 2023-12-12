import pandas as pd
from textblob import TextBlob

# Read the 'books.csv' file into a DataFrame
df = pd.read_csv('books.csv')

# Drop rows where 'original_title' is null
df = df.dropna(subset=['original_title'])

# Function to get sentiment polarity using TextBlob
def get_sentiment(title):
    analysis = TextBlob(str(title))
    return analysis.sentiment.polarity

# Apply the function to the 'original_title' column and create a new 'sentiment' column
df['sentiment'] = df['original_title'].apply(get_sentiment)

# Count the number of titles with positive sentiment (cutoff >= 0)
positive_titles_count = df[df['sentiment'] >= 0].shape[0]

# Display the result
print(f"The number of titles with positive sentiment is: {positive_titles_count}")
