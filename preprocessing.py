import re
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import json
import os 

class Preprocessor:
    def __init__(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        kamus_path = os.path.join(current_dir, 'kamus_slang.json')

        try:
            with open(kamus_path, 'r') as f:
                self.kamus_slang = json.load(f)
            print("Kamus slang berhasil dimuat.")
        except Exception as e:
            print(f"ERROR: Gagal memuat 'kamus_slang.json' dari {kamus_path}: {e}")
            self.kamus_slang = {}

        stopwords_id = set(stopwords.words('indonesian'))
        stopwords_en = set(stopwords.words('english'))
        custom_stopwords = {
            'di', 'ke', 'ya', 'eh', 'he', 'nya', 'nih', 'sih', 'si', 'tau', 'tuh',
            'dong', 'kok', 'wow', 'om', 'kak', 'bang', 'bro', 'cici', 'kakak', 'ka'
        }
        self.list_stopwords_final = stopwords_id.union(stopwords_en).union(custom_stopwords)
        self.list_stopwords_final.discard('tidak')
        self.list_stopwords_final.discard('aku')

        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        print("Preprocessor (Offline) siap.")

    def cleanse(self, text):
        text = re.sub(r'(@\w+|#\w+|https?://\S+)', '', text)
        text = re.sub(r'(\w)yg', r'\1 yang', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\b([a-zA-Z])\s+(?=[a-zA-Z]\b)', r'\1', text)
        text = re.sub(r'(\w)\1{1,}', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def normalize_slang(self, tokens):
        return [self.kamus_slang.get(word, word) for word in tokens]

    def remove_stopwords(self, tokens):
        return [word for word in tokens if word not in self.list_stopwords_final]

    def filter_length(self, tokens):
        return [word for word in tokens if len(word) > 1]

    def stem_tokens(self, tokens):
        if not tokens:
            return []
        return [self.stemmer.stem(word) for word in tokens]

    def preprocess_text(self, text):
        if not isinstance(text, str):
            return []
        text_clean = self.cleanse(text.lower())
        tokens = wordpunct_tokenize(text_clean)
        tokens_normalized = self.normalize_slang(tokens)
        tokens_stopped = self.remove_stopwords(tokens_normalized)
        tokens_filtered = self.filter_length(tokens_stopped)
        tokens_stemmed = self.stem_tokens(tokens_filtered)
        return tokens_stemmed