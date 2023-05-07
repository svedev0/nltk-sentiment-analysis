import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download required NLTK datasets
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# Load dataset
data_1 = """I hope this email finds you well. I just wanted to remind you that 
the documentation for the project was provided to everyone two weeks ago. It 
seems that some parts of it have not been thoroughly reviewed, as we have 
encountered some avoidable mistakes. Please take the time to read and 
understand the documentation before proceeding with any further work. Thank 
you for your cooperation."""

data_2 = """I hope this email finds you well. I wanted to follow up with you 
regarding the project's documentation. I noticed that there have been some 
recent changes made, and it is essential that everyone on the team is up to 
date on the latest information. It would be greatly appreciated if you could 
take some time to review the documentation thoroughly. This will ensure that 
we are all on the same page and can work together more efficiently towards 
achieving our project goals. Please let me know if you have any questions or 
concerns about the documentation, and I will be happy to assist you. Thank 
you for your attention to this matter."""

dataset = [data_1, data_2]  # Dataset
labels = ["Positive", "Negative"]  # Sentiment labels (e.g., positive or negative)


# Preprocess text data
def preprocess(text: str) -> str:
    # Tokenize text
    tokens = word_tokenize(text.lower())

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join tokens back into string
    preprocessed_text = " ".join(lemmatized_tokens)
    return preprocessed_text


def main() -> None:
    # Preprocess data
    preprocessed_dataset = [preprocess(data) for data in dataset]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        preprocessed_dataset, labels, test_size=0.2, shuffle=True, random_state=42
    )

    # Vectorize text data
    vectorizer = TfidfVectorizer()
    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)

    # Train a Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train_vectors, y_train)

    # Predict sentiment of test data
    y_pred = clf.predict(X_test_vectors)

    # Evaluate performance of classifier
    accuracy = accuracy_score(y_train, y_pred)
    # print('Accuracy:', accuracy)

    # Print sentiment labels for test data
    for i in range(len(y_test)):
        print("Sentiment for data_1:", y_test[i])
        print("Sentiment for data_2:", y_train[i])
        print("Predicted overall sentiment\n" + "of entire dataset:", y_pred[i])


if __name__ == "__main__":
    main()
