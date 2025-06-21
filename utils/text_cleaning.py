# utils/text_cleaning.py

import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download stopwords safely
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Initialize the stemmer
stemmer = PorterStemmer()

def clean_text(text):
    """
    Cleans input text by:
    - Lowercasing
    - Removing punctuation
    - Removing numbers
    - Removing stopwords
    - Applying stemming
    """
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Optional: test it from VS Code directly
if __name__ == "__main__":
    sample = "U.S. Economy [report] is growing 5% in 2024, experts say!"
    print("Original:", sample)
    print("Cleaned :", clean_text(sample))
