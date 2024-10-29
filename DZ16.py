
import re
import nltk
from nltk.corpus import conll2002
from sklearn.metrics import accuracy_score
import spacy
from spacy .training import Example
from gensim.models import Word2Vec, FastText

nltk.download('conll2002')
nltk_train_data = conll2002.iob_sents('esp.train')
nltk_test_data = conll2002.iob_sents('esp.testb')
nlp_spacy = spacy.blank("es")

unique_tags = set(tag for sent in nltk_train_data for _, tag, _ in sent)

tagger = nlp_spacy.add_pipe("tagger")
for tag in unique_tags:
    tagger.add_label(tag)

def prepare_sentences(nltk_data):
    sentences = []
    for sent in nltk_data:
        tokens = [token for token, _, _ in sent]
        sentences.append(tokens)
    return sentences

train_sentences = prepare_sentences(nltk_train_data)

model_word2vec_cbow = Word2Vec(sentences=train_sentences, vector_size=100, window=5, min_count=1, sg=0)
model_word2vec_sg = Word2Vec(sentences=train_sentences, vector_size=100, window=5, min_count=1, sg=1)
model_fasttext = FastText(sentences=train_sentences, vector_size=100, window=5, min_count=1)

def prepare_training_data(nltk_data, max_skipped=10):
    examples = []
    skipped_count = 0

    for sent in nltk_data:
        tokens = [token for token, _, _ in sent]
        tags = [tag for _, tag, _ in sent]

        doc = nlp_spacy.make_doc(" ".join(tokens))

        if len(doc) != len(tags):
            print(f"Несумісність у реченні: {tokens}")
            print(f"Кількість токенів: {len(doc)}, Кількість тегів: {len(tags)}")
            skipped_count += 1
            if skipped_count <= max_skipped:
                print(f"Пропущено речення через невідповідність довжин: {tokens} | Токени: {len(doc)}, Теги: {len(tags)}")
            continue

        examples.append(Example.from_dict(doc, {"tags": tags}))

    if skipped_count >= max_skipped:
        print(f"Загальна кількість пропущених речень через невідповідність довжин: {skipped_count}")

    return examples

train_examples = prepare_training_data(nltk_train_data)

optimizer = nlp_spacy.begin_training()
for i in range(10):
    for example in train_examples:
        nlp_spacy.update([example], drop=0.5)

def evaluate_tagger(nlp, test_data, lib_name="spaCy"):
    true_tags = []
    pred_tags = []
    skipped_count = 0

    for sent in test_data:
        tokens = [token for token, _, _ in sent]
        sentence_tags = [tag for _, tag, _ in sent]

        doc = nlp(" ".join(tokens))
        if len(doc) != len(sentence_tags):
            print(f"Пропущено речення під час оцінки: {tokens}")
            print(f"Кількість токенів: {len(doc)}, Кількість тегів: {len(sentence_tags)}")
            skipped_count += 1
            continue

        true_tags.extend(sentence_tags)
        pred_tags.extend([token.tag_ for token in doc])

    print(f"Пропущено речень під час оцінки через невідповідність довжин: {skipped_count}")

    accuracy = accuracy_score(true_tags, pred_tags)
    print(f"Точність PoS-теггера {lib_name}: {accuracy:.4f}")

evaluate_tagger(nlp_spacy, nltk_test_data, lib_name="spaCy")

print("Ембедінги згенеровані для порівняння моделей:")
print("Word2Vec CBOW: готово")
print("Word2Vec Skip-gram: готово")
print("FastText: готово")

"""
"""
