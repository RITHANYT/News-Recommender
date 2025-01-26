import nltk
import os

# Define the path for downloading the NLTK data
nltk_data_path = os.path.expanduser('~/nltk_data')  # Default path for NLTK data
nltk.data.path.append(nltk_data_path)

# Check if the 'punkt' tokenizer is available, otherwise download it
try:
    nltk.data.find('tokenizers/punkt')
    print("Punkt tokenizer found!")
except LookupError:
    print("Punkt tokenizer not found. Downloading...")
    nltk.download('punkt', download_dir=nltk_data_path)
    print("Punkt tokenizer downloaded successfully!")
