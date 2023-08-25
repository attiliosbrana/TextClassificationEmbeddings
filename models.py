from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class Model:
    def load_embeddings(self, train_data1, train_data2):
        raise NotImplementedError

    def classify(self, test_data, embeddings, category1, category2):
        raise NotImplementedError


class BAIModel(Model):
    def load_embeddings(self, train_data1, train_data2):
        model = SentenceTransformer("BAAI/bge-large-en")
        embeddings1 = model.encode(train_data1, normalize_embeddings=True)
        embeddings2 = model.encode(train_data2, normalize_embeddings=True)

        # Compute average embeddings
        avg_embedding1 = np.mean(embeddings1, axis=0)
        avg_embedding2 = np.mean(embeddings2, axis=0)

        return avg_embedding1, avg_embedding2

    def classify(self, test_data, embeddings, category1, category2):
        model = SentenceTransformer("BAAI/bge-large-en")
        results = []
        for sentence in test_data:
            sentence_embedding = model.encode([sentence], normalize_embeddings=True)[0]
            similarities = cosine_similarity([sentence_embedding], embeddings)
            if similarities[0][0] > similarities[0][1]:
                results.append(category1)
            else:
                results.append(category2)
        return results


class GloveModel(Model):
    def load_embeddings(self, train_data1, train_data2):
        model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d"
        )
        embedding1 = model.encode([" ".join(train_data1)])[0]
        embedding2 = model.encode([" ".join(train_data2)])[0]
        return embedding1, embedding2

    def classify(self, test_data, embeddings, category1, category2):
        model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d"
        )
        results = []
        for sentence in test_data:
            sentence_embedding = model.encode([sentence])[0]
            cosine_to_cat1 = cosine_similarity([sentence_embedding], [embeddings[0]])
            cosine_to_cat2 = cosine_similarity([sentence_embedding], [embeddings[1]])
            if cosine_to_cat1 > cosine_to_cat2:
                results.append(category1)
            else:
                results.append(category2)
        return results


class TFIDFModel(Model):
    def load_embeddings(self, train_data1, train_data2):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(train_data1 + train_data2)
        return vectorizer, tfidf_matrix, len(train_data1), len(train_data2)

    def classify(self, test_data, embeddings, category1, category2):
        vectorizer, tfidf_matrix_train, len_train_data1, _ = embeddings

        tfidf_matrix_test = vectorizer.transform(test_data)

        results = []
        for index, sentence in enumerate(test_data):
            cosine_to_cat1 = cosine_similarity(
                tfidf_matrix_test[index], tfidf_matrix_train[:len_train_data1]
            ).mean()
            cosine_to_cat2 = cosine_similarity(
                tfidf_matrix_test[index], tfidf_matrix_train[len_train_data1:]
            ).mean()

            if cosine_to_cat1 > cosine_to_cat2:
                results.append(category1)
            else:
                results.append(category2)
        return results
