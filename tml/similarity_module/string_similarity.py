from strsimpy import *
from strsimpy.jaro_winkler import JaroWinkler
from strsimpy.overlap_coefficient import OverlapCoefficient


class StringSimilarity:

    def __init__(self, s1, s2):
        if self._contains_irregular_capitalization(s=s1):
            self.s1 = s1
        else:
            self.s1 = s1.lower()
        if self._contains_irregular_capitalization(s=s2):
            self.s2 = s2
        else:
            self.s2 = s2.lower()

    @staticmethod
    def _contains_irregular_capitalization(s):
        if all([(not c.islower()) for c in s]):
            return False # all uppercase
        if all([(not c.isupper()) for c in s]):
            return False # all lowercase
        for word in s.split(' '):
            if word[0].islower() and not all([(not c.isupper()) for c in word]):
                return True
            if word[0].isupper() and not all([(not c.islower()) for c in word]):
                return True
        return False

    def lev(self):
        return Levenshtein().distance(self.s1, self.s2)

    def normalized_lev(self):
        return NormalizedLevenshtein().similarity(self.s1, self.s2)

    def damerau(self):
        return Damerau().distance(self.s1, self.s2)

    def osa(self):
        return OptimalStringAlignment().distance(self.s1, self.s2)

    def jw(self):
        return JaroWinkler().similarity(self.s1, self.s2)

    def lcs(self):
        return LongestCommonSubsequence().distance(self.s1, self.s2)

    def metric_lcs(self):
        return MetricLCS().distance(self.s1, self.s2)

    def two_n_gram(self):
        return NGram(2).distance(self.s1, self.s2)

    def three_n_gram(self):
        return NGram(3).distance(self.s1, self.s2)

    def four_n_gram(self):
        return NGram(4).distance(self.s1, self.s2)

    def two_q_gram(self):
        return QGram(2).distance(self.s1, self.s2)

    def three_q_gram(self):
        return QGram(3).distance(self.s1, self.s2)

    def four_q_gram(self):
        return QGram(4).distance(self.s1, self.s2)

    def two_cosine(self):
        return Cosine(2).similarity(self.s1, self.s2)

    def three_cosine(self):
        return Cosine(3).similarity(self.s1, self.s2)

    def four_cosine(self):
        return Cosine(4).similarity(self.s1, self.s2)

    def two_jaccard(self):
        return Jaccard(2).similarity(self.s1, self.s2)

    def three_jaccard(self):
        return Jaccard(3).similarity(self.s1, self.s2)

    def four_jaccard(self):
        return Jaccard(4).similarity(self.s1, self.s2)

    def sorensen(self):
        try:
            return SorensenDice().similarity(self.s1, self.s2)
        except ZeroDivisionError:
            return 0

    def two_overlap(self):
        try:
            return OverlapCoefficient(2).similarity(self.s1, self.s2)
        except ZeroDivisionError:
            return 0

    def three_overlap(self):
        try:
            return OverlapCoefficient(3).similarity(self.s1, self.s2)
        except ZeroDivisionError:
            return 0

    def four_overlap(self):
        try:
            return OverlapCoefficient(4).similarity(self.s1, self.s2)
        except ZeroDivisionError:
            return 0

    def sift(self):
        return SIFT4().distance(self.s1, self.s2)