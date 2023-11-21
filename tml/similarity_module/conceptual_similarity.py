import openai

from strsimpy import Levenshtein
from strsimpy import Cosine
from strsimpy import LongestCommonSubsequence
from nltk.corpus import wordnet


class ConceptualSimilarity():

    def __init__(self):
        self.all_words = list(set(wordnet.words()))

    def get_conceptual_similarity(self, s1: str, s2: str, sim_measure: str):
        c1 = self._get_closest_concept(s=s1, sim=sim_measure)
        c2 = self._get_closest_concept(s=s2, sim=sim_measure)
        return self._compute_conceptual_similarity(s1=c1, s2=c2)


    def _compute_conceptual_similarity(self, s1: str, s2: str):
        syns1 = wordnet.synsets(s1)[0]
        syns2 = wordnet.synsets(s2)[0]
        return syns1.wup_similarity(syns2)

    def _get_closest_concept(self, s: str, sim: str):
        s = s.lower()
        min_distance = 100000000000
        min_distance_word = ''
        for w in self.all_words:
            if sim == 'lev':
                d = Levenshtein().distance(s, w)
            if sim == 'cos':
                d = Cosine(2).distance(s, w)
            if sim == 'lcs':
                d = LongestCommonSubsequence().distance(s, w)
            if d < min_distance:
                min_distance = d
                min_distance_word = w
            if d == 0:
                break
        return min_distance_word


    '''
    @staticmethod
    def _openai_comparison(s1: str, s2: str):
        openai.api_key = "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"
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are provided two words. Your job is to provide a the score for their conceptual similarity (between 0 and 1). Conceptual similarity is the semantic similarity between the concepts both words are related to. Only provide the score (numeric), do not reply text."},
                {"role": "user", "content": f"{s1}, {s2}"}
            ],
            temperature=0
        )
        return float(completion.choices[0].message.content)
    '''