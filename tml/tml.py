from tml.statistics_module.dataset_exploration import export_statistics
from tml.similarity_module.string_similarity import StringSimilarity
from tml.similarity_module.character_similarity import CharacterSimilarity
from tml.similarity_module.phentic_encoding import PhoneticEncoding
from tml.similarity_module.conceptual_similarity import ConceptualSimilarity

from glob import glob

import pandas as pd
from tqdm import tqdm
from tml.TrademarkML import TrademarkML