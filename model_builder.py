import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

class ModelBuilder:
    def __init__(self, preprocessor, word_embedding):
        self.preprocessor = preprocessor
        self.word_embedding = word_embedding
        self.model = None

    def load_model(self, model_path):
        try:
            self.model = tf.keras.models.load_model(model_path, compile=False)
            print(f"Model berhasil dimuat dari {model_path}")
            return True
        except Exception as e:
            print(f"ERROR: Gagal memuat model dari {model_path}: {e}")
            return False

    def classify_text(self, text_input):
        if self.model is None:
            raise ValueError("Model belum dimuat. Panggil load_model() dulu.")
            
        print(f"Running : {text_input[:20]}...")
        tokens = self.preprocessor.preprocess_text(text_input)
        text_str = ' '.join(tokens)
        
        sequences_padded = self.word_embedding.get_sequences([text_str])
        
        try:
            score = self.model.predict(sequences_padded, verbose=0)
            return float(score[0][0]) 
        except Exception as e:
            print(f"ERROR saat prediksi tunggal: {e}")
            return 0.0

    def classify_batch(self, processed_text_list):
        if self.model is None:
            raise ValueError("Model belum dimuat. Panggil load_model() dulu.")
            
        sequences_padded = self.word_embedding.get_sequences(processed_text_list)
        
        try:
            scores_batch = self.model.predict(sequences_padded, verbose=0)
            return scores_batch.flatten()
        except Exception as e:
            print(f"ERROR saat prediksi batch: {e}")
            return np.array([0.0] * len(processed_text_list))