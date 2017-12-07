import string
from collections import Counter
from operator import itemgetter


import numpy as np
import pandas as pd


class Characters:
    """
        all lowercase for now
    """
    _accentuated_range = set(range(224, 256))

    accentuated = ''.join(map(chr, _accentuated_range))
    accentuated_set = set(accentuated)

    ascii = string.ascii_lowercase
    ascii_set = set(ascii)

    digits_set = set(string.digits)

    all = ascii + string.digits + accentuated 
    all_set = set(all)


class Words(pd.DataFrame):
    class COLUMNS:
        WORD = 'word'
        LABEL = 'label'
        
    class LABELS:
        TRUE = 'true'
        FALSE = 'false'
        #TRUE_VALUE = True?? to use in masks??

    def __init__(self, fn_words='words.txt', sep=' ', columns=[COLUMNS.WORD], fn_labels ='wordsclass.npy'):
        """
            file is a list of words separated by a space
        """
        super().__init__(pd.read_csv(fn_words, sep=sep, encoding='latin-1').columns, columns=columns)
        training_labels = np.load(fn_labels)
        self[Words.COLUMNS.LABEL] = training_labels

        self.series_true = self.make_series(self[Words.COLUMNS.LABEL])
        self.series_false = self.make_series(self[Words.COLUMNS.LABEL]==False)

        self.df_true_false = pd.DataFrame(
            { Words.LABELS.TRUE : self.series_true,
              Words.LABELS.FALSE : self.series_false
            }).reset_index(drop=True)

        self.df_true = self.make_df(self.series_true)
        self.df_false = self.make_df(self.series_false)

    def make_series(self, mask):
        return self[mask][Words.COLUMNS.WORD].sort_values().reset_index(drop=True)

    def make_df(self, series):
        """
            DataFrame with words as index and characters on columns, character couts on cells
        """
        df = pd.DataFrame(index = series, columns = list(Characters.all))
        for w in df.index:
            df.loc[w] = pd.Series(Counter(w))

        df.fillna(0, inplace=True)
        df.drop(df.columns[(df == 0).all(axis=0)], axis=1, inplace=True)

        return df


class Explorer:
    """
        designed to explore a column labeled True or False
        column is an iterable of words
        and the goal is to based on those words or the characters that compose them
        differentiate columns with different labels
    """

    def __init__(self, words):
        self.words = words
        #self.
        #x['counter'] = [Counter(e) for e in x.word]

        for index, label in zip((0, -1), ('first', 'last')):
            setattr(self, f"chr_set_{label}", set(map(itemgetter(index), self.words)))

        self.chr_set_all = {c for w in words for c in w}


class CharacterSet:
    """
        designed to describe a character set
    """
    def __init__(self, s, name):
        self.set = s
        self.name = name
        self.length = len(self.set)
        self.accentuated = self.set - Characters.ascii_set 
        self.ascii = self.set - Characters.accentuated_set
        self.ascii_all = self.accentuated == {}
        self.ascii_p = len(self.ascii) / len(self.set)
        self.digits = self.set & Characters.digits_set
        
    def report(self, types = (set, bool, int, float,)):
        print()
        print('set related to', self.name)

        for k,v in self.__dict__.items():
            if type(v) in types:
                print(k, v) 


class CharacterSets:
    """
        designed to compare differences between character sets
    """
    def __init__(self, true, false, name):
        self.name = name
        self.true = true
        self.false = false
        self.intersection = self.true & self.false
        self.union = self.true | self.false
        self.true_minus_false = self.true - self.false
#        self.true_minus_false_accentuated = self.true_minus_false - Characters.ascii_set
        self.false_minus_true = self.false - self.true
 #       self.false_minus_true_accentuated = self.false_minus_true - Characters.ascii_set
        self.symmetric_difference = self.true.symmetric_difference(self.false)
        self.true_issubset = self.true.issubset(self.false)
        self.true_issuperset = self.true.issuperset(self.false)

        self.true_ = CharacterSet(self.true, Words.LABELS.TRUE)
        self.false_ = CharacterSet(self.false, Words.LABELS.FALSE)


#        self.intersection_length = len(self.intersection)

    def report(self, types = (set, bool, int, float,)):
        print()
        print('sets related to', self.name)

        for k,v in self.__dict__.items():
            if type(v) in types:
                print(k, v) 
            elif isinstance(v, CharacterSet):
                v.report()

        print();print()

class Comparator:
    """
        designed to compare attributes of explorers for True and False Labels
    """
    def __init__(self, true, false):
        
        self.true = true
        self.false = false
        self.chr_first = CharacterSets(self.true.chr_set_first, self.false.chr_set_first, "chr_first")
        self.chr_last = CharacterSets(self.true.chr_set_last, self.false.chr_set_last, "chr_last")
        self.chr_all = CharacterSets(self.true.chr_set_all, self.false.chr_set_all, "chr_all")

    def report(self):
        self.chr_first.report() 
        self.chr_last.report() 
        self.chr_all.report() 

def driver():
    w = Words()
    c = Comparator(Explorer(w.series_true), Explorer(w.series_false))
    return c
        

if __name__ == '__main__':
    c = driver()
    c.report()

