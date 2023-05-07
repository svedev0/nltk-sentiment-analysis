from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Read text from input file
with open('input_file.txt', 'r') as file:
    text = file.read()

# Tokenize text into words
words = word_tokenize(text)

# Remove stop words from the words list
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words]

# User-supplied keywords
keywords = input('important,keywords,to,search,for')
keywords = keywords.split(',')

# Find the important content based on keywords
important_content = []
for word in filtered_words:
    if word.lower() in keywords:
        important_content.append(word)

# Print the important content
print('Important content:')
print(' '.join(important_content))
