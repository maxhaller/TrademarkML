from tml.tml import TrademarkML

import pandas as pd

data_path='../TMSIM-500/data.csv',
#stats_path='../dataset stats',
#img_path='../TMSIM-500/Dataset Images',
#letter_path='../weighted_levenshtein/characters',
#weights_path='../weighted_levenshtein'

trademarkml = TrademarkML()

full_df = pd.read_csv('../TMSIM-500/data.csv', sep=',', encoding='utf-8')
#word_df, figurative_df = trademarkml.get_word_and_fig_mark_set(full_df)
#word_df = trademarkml.compute_features(df=word_df)
word_df = pd.read_csv('./features_full.csv', sep=',', encoding='utf-8')
word_df = word_df[full_df['Type'] == 'figurative'].reset_index()

train_word_df, test_word_df, train_idx, test_idx = trademarkml.train_test_split(df=word_df, id_col='Case ID')

x_train, y_train = trademarkml.get_x_y_from_df(df=train_word_df, y_col='Outcome')
x_test, y_test = trademarkml.get_x_y_from_df(df=test_word_df, y_col='Outcome')
trademarkml.fit(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, train_idx=train_idx, word_mark_df=word_df, set='figurative')

word_df = pd.read_csv('./features_full.csv', sep=',', encoding='utf-8')
word_df = word_df[full_df['Type'] == 'word'].reset_index()

train_word_df, test_word_df, train_idx, test_idx = trademarkml.train_test_split(df=word_df, id_col='Case ID')

x_train, y_train = trademarkml.get_x_y_from_df(df=train_word_df, y_col='Outcome')
x_test, y_test = trademarkml.get_x_y_from_df(df=test_word_df, y_col='Outcome')
trademarkml.fit(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, train_idx=train_idx, word_mark_df=word_df, set='word')