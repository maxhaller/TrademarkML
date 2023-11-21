import fasttext
import spacy_universal_sentence_encoder
import numpy as np

class ItemSimilarity():

    def __init__(self):
        self.nlp = spacy_universal_sentence_encoder.load_model('en_use_lg')
        self.ft = fasttext.load_model('D://wiki.en.bin')

    def get_item_similarity_google_use(self, earlier_items, contested_item):
        earlier_list = earlier_items.split(';')
        sim = 0
        c_emb = self.nlp(contested_item)
        for earlier in earlier_list:
            e_emb = self.nlp(earlier)
            curr_sim = c_emb.similarity(e_emb)
            if sim < curr_sim:
                sim = curr_sim
        return sim

    def get_item_similarity_fasttext(self, earlier_items, contested_item):
        earlier_list = earlier_items.split(';')
        sim = 0
        c_emb = self.ft.get_sentence_vector(contested_item)
        for earlier in earlier_list:
            e_emb = self.ft.get_sentence_vector(earlier)
            curr_sim = self._cos_sim(e_emb, c_emb)
            if sim < curr_sim:
                sim = curr_sim
        return sim

    @staticmethod
    def _cos_sim(a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)