import numpy as np
import time

from sklearn.pipeline import Pipeline

from tml.similarity_module.item_similarity import ItemSimilarity
from tml.similarity_module.phentic_encoding import PhoneticEncoding
from tml.similarity_module.string_similarity import StringSimilarity
from tml.similarity_module.conceptual_similarity import ConceptualSimilarity
from tml.statistics_module.dataset_exploration import export_statistics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler
from sklearn.decomposition import PCA

import pandas as pd
import pickle


def get_visual_similarity_method_calls(s1: str, s2:str):
    sim = StringSimilarity(s1=s1, s2=s2)
    return [(getattr(sim, m), m) for m in dir(sim) if callable(getattr(sim, m)) if not m.startswith('_')]


class TrademarkML:

    def __init__(self):
        pass


    def compute_features(self, df: pd.DataFrame,
                         tm1_col: str = 'Contested Trademark',
                         tm2_col: str = 'Earlier Trademark',
                         contested_item_col: str = 'Contested Goods and Services',
                         earlier_items_col: str = 'Earlier Goods and Services'):
        vis_cache = {}
        aur_enc_cache = {}
        conc_cache = {}
        conceptual_sim = ConceptualSimilarity()
        for i, row in df.iterrows():
            tm1 = row[tm1_col]
            tm2 = row[tm2_col]
            for feature in self._get_visual_similarity_method_calls(s1=tm1, s2=tm2):
                if tm1 not in vis_cache:
                    vis_cache[tm1] = {}
                if tm2 not in vis_cache[tm1]:
                    vis_cache[tm1][tm2] = {}
                if feature[1] not in vis_cache[tm1][tm2]:
                    vis_cache[tm1][tm2][feature[1]] = feature[0]()
                df.loc[i, f'visual_{feature[1]}'] = vis_cache[tm1][tm2][feature[1]]

            for feature in self._get_phonetic_encoding_method_calls():
                if tm1 not in aur_enc_cache:
                    aur_enc_cache[tm1] = {}
                if feature[1] not in aur_enc_cache[tm1]:
                    aur_enc_cache[tm1][feature[1]] = feature[0](tm1)
                if tm2 not in aur_enc_cache:
                    aur_enc_cache[tm2] = {}
                if feature[1] not in aur_enc_cache[tm2]:
                    aur_enc_cache[tm2][feature[1]] = feature[0](tm2)

                tm1_enc = aur_enc_cache[tm1][feature[1]]
                tm2_enc = aur_enc_cache[tm2][feature[1]]
                for vis_feature in self._get_visual_similarity_method_calls(tm1_enc, tm2_enc):
                    if tm1_enc not in vis_cache:
                        vis_cache[tm1_enc] = {}
                    if tm2_enc not in vis_cache[tm1_enc]:
                        vis_cache[tm1_enc][tm2_enc] = {}
                    if vis_feature[1] not in vis_cache[tm1_enc][tm2_enc]:
                        vis_cache[tm1_enc][tm2_enc][vis_feature[1]] = vis_feature[0]()
                    df.loc[i, f'aural_{feature[1]}_{vis_feature[1]}'] = vis_cache[tm1_enc][tm2_enc][vis_feature[1]]

            for method in ['lev', 'cos', 'lcs']:
                # cache
                if tm1 not in conc_cache:
                    conc_cache[tm1] = {}
                if tm2 not in conc_cache[tm1]:
                    conc_cache[tm1][tm2] = {}
                if method not in conc_cache[tm1][tm2]:
                    conc_cache[tm1][tm2][method] = conceptual_sim.get_conceptual_similarity(tm1, tm2, method)

                df.loc[i, f'conceptual_{method}_wordnet'] = conc_cache[tm1][tm2][method]
            
            earlier_items = row[earlier_items_col]
            contested_item = row[contested_item_col]

            for feature in self._get_item_similarity_method_calls():
                df.loc[i, f'item_{feature[1]}'] = feature[0](earlier_items, contested_item)

        return df


    @staticmethod
    def get_word_and_fig_mark_set(df: pd.DataFrame):
        word_df = df.loc[df['Type'] == 'word']
        figurative_df = df.loc[df['Type'] == 'figurative']
        return word_df, figurative_df

    @staticmethod
    def export_dataset_statistics(**kwargs):
        df = kwargs.get('dataset', None)
        data_path = kwargs.get('data_path', None)
        img_path = kwargs.get('img_path', None)
        stats_path = kwargs.get('stats_path', None)
        if data_path:
            df = pd.read_csv(data_path, sep=',', encoding='utf-8')

        export_statistics(df, img_path, stats_path)

    @staticmethod
    def train_test_split(df: pd.DataFrame, id_col: str):
        splitter = GroupShuffleSplit(test_size=.20, n_splits=1, random_state=42)
        split = splitter.split(df, groups=df[id_col])
        train_idx, test_idx = next(split)
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]
        return df_train, df_test, train_idx, test_idx

    @staticmethod
    def get_x_y_from_df(df: pd.DataFrame, y_col: str):
        feature_cols = [c for c in df.columns if c not in ['outcome', 'ID', 'Type', 'index']]
        return df[feature_cols], df[y_col]

    @staticmethod
    def _get_visual_similarity_method_calls(s1: str, s2:str):
        sim = StringSimilarity(s1=s1, s2=s2)
        return [(getattr(sim, m), m) for m in dir(sim) if callable(getattr(sim, m)) if not m.startswith('_')]

    @staticmethod
    def _get_phonetic_encoding_method_calls():
        pe = PhoneticEncoding()
        return [(getattr(pe, m), m) for m in dir(pe) if callable(getattr(pe, m)) if not m.startswith('_')]

    @staticmethod
    def _get_item_similarity_method_calls():
        itsim = ItemSimilarity()
        return [(getattr(itsim, m), m) for m in dir(itsim) if callable(getattr(itsim, m)) if not m.startswith('_')]

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame, word_mark_df: pd.DataFrame, train_idx: np.ndarray, set: str):
        cols = x_train.columns

        train_set_indices = x_train.index
        test_set_indices = x_test.index

        x_train = x_train.reset_index()
        x_test = x_test.reset_index()

        if set == 'word':
            vis_features = [c for c in cols if c.startswith('visual') and 'vgg' not in c and 'resnet' not in c]
        else:
            vis_features = [c for c in cols if c.startswith('vgg') or c.startswith('resnet')]

        aur_features = [c for c in cols if c.startswith('aural')]
        con_features = [c for c in cols if c.startswith('conceptual')]
        it_features = [c for c in cols if c.startswith('item')]

        pipeline = Pipeline([('scaling', RobustScaler()), ('pca', PCA(n_components=1, random_state=42))])

        x_train.loc[:, 'vis_pca'] = pd.Series(pipeline.fit_transform(X=x_train[vis_features])[:, 0])
        x_test.loc[:, 'vis_pca'] = pd.Series(pipeline.transform(X=x_test[vis_features])[:, 0])
        vis_features.append('vis_pca')

        m2_features = [c for c in cols if 'metaphone2' in c]
        x_train.loc[:, 'm2_pca'] = pd.Series(pipeline.fit_transform(X=x_train[m2_features])[:, 0])
        x_test.loc[:, 'm2_pca'] = pd.Series(pipeline.transform(X=x_test[m2_features])[:, 0])
        aur_features.append('m2_pca')

        m3_features = [c for c in cols if 'metaphone3' in c]
        x_train.loc[:, 'm3_pca'] = pd.Series(pipeline.fit_transform(X=x_train[m3_features])[:, 0])
        x_test.loc[:, 'm3_pca'] = pd.Series(pipeline.transform(X=x_test[m3_features])[:, 0])
        aur_features.append('m3_pca')

        vis_features.append('none')
        aur_features.append('none')
        con_features.append('none')
        it_features.append('none')

        x_train.set_index(train_set_indices)
        x_test.set_index(test_set_indices)

        print(x_test)

        print(vis_features)
        print(aur_features)
        print(con_features)
        print(it_features)

        binarizer = LabelBinarizer()
        y_train = binarizer.fit_transform(y_train).ravel()
        y_test = binarizer.transform(y_test).ravel()

        svm = LinearSVC(random_state=42, dual='auto')
        svm_grid = {'C': [0.01, 0.1, 1, 10, 100]}

        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        rf_grid = {'max_depth': [25, 50, 75],
                   'max_features': ['log2', 'sqrt'],
                   'n_estimators': [15, 20, 50]}

        models = [
            {
                'name': 'rf',
                'clf': rf,
                'grid': rf_grid
            },             
            {
                'name': 'svm',
                'clf': svm,
                'grid': svm_grid
            }
        ]

        scalers = [
            'none',
            'normalizer',
            'standardscaler'
        ]

        for model in models:
            m_name = model['name']
            counter = 1
            result = ''
            best_acc = 0
            best_iteration = 0
            best_gridsearch = None
            for v in vis_features:
                for a in aur_features:
                    for c in con_features:
                        for i in it_features:
                            if m_name == 'svm' or m_name == 'mlp':
                                for scaler in scalers:
                                    cols = []
                                    if v != 'none':
                                        cols.append(v)
                                    if a != 'none':
                                        cols.append(a)
                                    if c != 'none':
                                        cols.append(c)
                                    if i != 'none':
                                        cols.append(i)

                                    if len(cols) > 0:
                                        print(f'{m_name} - {counter}: {scaler}, {cols}')
                                        split = GroupShuffleSplit(n_splits=5, train_size=.8, random_state=42).split(X=x_train, y=y_train, groups=word_mark_df.loc[train_idx, 'Case ID'])
                                        gridsearch = GridSearchCV(model['clf'], cv=split, param_grid=model['grid'])

                                        sc = None
                                        if scaler == 'normalizer':
                                            sc = Normalizer()
                                        if scaler == 'standardscaler':
                                            sc = StandardScaler()

                                        if scaler != 'none':
                                            x_train_scaled = sc.fit_transform(x_train[cols])
                                            x_test_scaled = sc.transform(x_test[cols])
                                        else:
                                            x_train_scaled = x_train[cols]
                                            x_test_scaled = x_test[cols]

                                        gridsearch.fit(x_train_scaled, y_train)
                                        y_pred = gridsearch.predict(x_test_scaled)
                                        acc = accuracy_score(y_pred=y_pred, y_true=y_test)
                                        precision = precision_score(y_pred=y_pred, y_true=y_test)
                                        recall = recall_score(y_pred=y_pred, y_true=y_test)
                                        auc = roc_auc_score(y_score=y_pred, y_true=y_test)
                                        f1 = f1_score(y_pred=y_pred, y_true=y_test)
                                        result += f'\n{counter}: {v}, {a}, {c}, {i}, {scaler}\n\naccuracy: {acc}\nprecision: {precision}\nrecall: {recall}\nroc: {auc}\nf1: {f1}\n@@@@@@@@@\n\nbest params: {gridsearch.best_params_}\n\n'
                                        if best_acc < acc:
                                            best_acc = acc
                                            best_iteration = counter
                                            best_gridsearch = gridsearch
                                        print(f'{counter}: {acc}')
                                        counter += 1
                            else:
                                cols = []
                                if v != 'none':
                                    cols.append(v)
                                if a != 'none':
                                    cols.append(a)
                                if c != 'none':
                                    cols.append(c)
                                if i != 'none':
                                    cols.append(i)

                                if len(cols) > 0:
                                    print(f'{m_name} - {counter}: {cols}')
                                    split = GroupShuffleSplit(n_splits=5, train_size=.8, random_state=42).split(X=x_train, y=y_train, groups=word_mark_df.loc[train_idx, 'Case ID'])
                                    gridsearch = GridSearchCV(model['clf'], cv=split, param_grid=model['grid'])
                                    x_train_scaled = x_train[cols]
                                    x_test_scaled = x_test[cols]
                                    gridsearch.fit(x_train_scaled, y_train)
                                    y_pred = gridsearch.predict(x_test_scaled)
                                    acc = accuracy_score(y_pred=y_pred, y_true=y_test)
                                    precision = precision_score(y_pred=y_pred, y_true=y_test)
                                    recall = recall_score(y_pred=y_pred, y_true=y_test)
                                    auc = roc_auc_score(y_score=y_pred, y_true=y_test)
                                    f1 = f1_score(y_pred=y_pred, y_true=y_test)
                                    result += f'\n{counter}: {v}, {a}, {c}, {i}\n\naccuracy: {acc}\nprecision: {precision}\nrecall: {recall}\nroc: {auc}\nf1: {f1}\n@@@@@@@@@\n\nbest params: {gridsearch.best_params_}\n\n'
                                    if best_acc < acc:
                                        best_acc = acc
                                        best_iteration = counter
                                        best_gridsearch = gridsearch
                                    print(f'{counter}: {acc}')
                                    counter += 1

            with open(f'tml_results_{m_name}_{set}_final_5-fold.txt', 'w') as file:
                file.write(f'best iteration: {best_iteration}\n\n\n' + result)

            with open(f'tml_best_{m_name}_{set}_final_5-fold.pickle', 'wb') as pickle_file:
                pickle.dump(best_gridsearch.best_estimator_, pickle_file)


#%%
