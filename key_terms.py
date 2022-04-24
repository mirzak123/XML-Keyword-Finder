from nltk import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from lxml import etree
from sklearn.feature_extraction.text import TfidfVectorizer
import string


def get_lemmatized_tokens(tokens):
    lemmatized_tokens = []
    lemmatizer = WordNetLemmatizer()

    for token in tokens:
        lemma = lemmatizer.lemmatize(token)
        lemmatized_tokens.append(lemma)

    return lemmatized_tokens


def get_removed_punctuation(tokens):
    new_tokens = []
    for token in tokens:
        new_token = token.rstrip(string.punctuation)
        if new_token in stopwords.words('english') or new_token == '':
            continue
        new_tokens.append(new_token)

    return new_tokens


def main():
    xml_file = 'news.xml'
    root = etree.parse(xml_file).getroot()

    dataset = []  # store texts from all news stories
    headings = []  # store all news headings

    for news in root[0]:
        head = news[0].text + ':'
        headings.append(head)

        tokens = sorted(word_tokenize(news[1].text.lower()), reverse=True)
        tokens = get_removed_punctuation(get_lemmatized_tokens(tokens))

        nouns = [token for token in tokens if pos_tag([token])[0][1] == 'NN']  # nltk pos_tag each word and check if they are nouns
        dataset.append(" ".join(nouns))

    vectorizer = TfidfVectorizer(input='content', use_idf=True, analyzer='word', ngram_range=(1, 1))
    tfidf_matrix = vectorizer.fit_transform(dataset).toarray()  # toarray() makes the tfidf matrix easier to work with
    terms = vectorizer.get_feature_names_out()  # entire dataset vocabulary

    tfidf_scores = []  # list of lists that stores all words and their tfidf scores as a tuple
    for i, heading in enumerate(headings):
        tfidf_scores.append([])  # create new list that corresponds to the heading by index
        for j, score in enumerate(tfidf_matrix[i]):
            if score != 0:  # disregard meaningless words
                tfidf_scores[i].append((terms[j], score))

        tfidf_scores[i].sort(key=(lambda x: (x[1], x[0])), reverse=True)  # sort all words from highest to lowest tfidf scores

    for i, heading in enumerate(headings):
        print(heading)
        print(*[x for x, y in tfidf_scores[i]][:5], end='\n\n')  # output 5 keywords for each news story


if __name__ == "__main__":
    main()
