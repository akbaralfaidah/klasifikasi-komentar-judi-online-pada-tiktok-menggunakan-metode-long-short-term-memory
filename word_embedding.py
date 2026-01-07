import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

class WordEmbedding:
    def __init__(self):
        self.config = {
            'max_length': 50,
            'embedding_dim': 300,
            'vocab_size': 10000
        }
        self.tokenizer = None 
        print("WordEmbedding (Offline) siap.")

    def load_tokenizer(self, filepath='tokenizer.json'):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                tokenizer_data = json.load(f)
                # Keras memuat dari string json, bukan dari file
                self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_data)
            print(f"Tokenizer berhasil dimuat dari {filepath}")
        except Exception as e:
            print(f"ERROR: Gagal memuat tokenizer dari {filepath}: {e}")
            raise e 

    def get_sequences(self, texts_as_strings):
        if self.tokenizer is None:
            raise ValueError("Tokenizer belum dimuat. Panggil load_tokenizer() dulu.")
            
        seq = self.tokenizer.texts_to_sequences(texts_as_strings)
        return pad_sequences(
            seq, 
            maxlen=self.config['max_length'], 
            padding='post', 
            truncating='post'
        )