from tml.tml import TrademarkML

import pandas as pd

data_path='../TMSIM-500/data.csv'
stats_path='../dataset stats'
img_path='../TMSIM-500/Dataset Images'
#letter_path='../weighted_levenshtein/characters',
#weights_path='../weighted_levenshtein'

trademarkml = TrademarkML()

full_df = pd.read_csv('../TMSIM-500/data.csv', sep=',', encoding='utf-8')
trademarkml.export_dataset_statistics(dataset=full_df, img_path=img_path, stats_path=stats_path)