import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import cv2
import matplotlib.pylab as pylab

from glob import glob


def add_count_to_dict(d: dict, df: pd.DataFrame, key: str, types: list[str], outcomes: list[str], target_value: str):
    if key not in d:
        d[key] = {}
    for t in types:
        if t not in d[key]:
            d[key][t] = {}
        sub_df = df.loc[df['Type'] == t]
        d[key][t]['total'] = len(sub_df)
        for outcome in outcomes:
            d[key][t][outcome] = len(sub_df.loc[sub_df[target_value] == outcome])
    return d


def get_number_of_samples_per_class_and_type(df: pd.DataFrame):
    stats = add_count_to_dict({}, df, 'comparison-level', ['word', 'figurative'], outcomes=['upheld', 'rejected'], target_value='Outcome')
    stats = add_count_to_dict(stats, df.drop_duplicates(subset=['Case ID']), 'tm-level', ['word', 'figurative'], outcomes=['upheld', 'rejected', 'partially upheld'], target_value='Opposition Outcome')
    return stats


def store_statistics(functions: [], df: pd.DataFrame, target_path: str):
    for f in functions:
        result = f(df)
        title = '_'.join(f.__name__.split('_')[1:])
        if type(result) == plt.Figure:
            result.savefig(f'{target_path}/{title}.png', bbox_inches='tight', pad_inches=0.3)
        else:
            with open(f'{target_path}/{title}.txt', 'w') as file:
                file.write(json.dumps(result))
        plt.clf()
        plt.cla()


def get_correlation_matrix(df: pd.DataFrame):
    corr_cols = df[['Type', 'Visual Similarity', 'Aural Similarity', 'Conceptual Similarity', 'Degree of Attention', 'Distinctiveness', 'Opposition Outcome', 'Item Similarity', 'Outcome']]
    corr_cols.loc[corr_cols['Type'] == 'word', 'Type'] = 1
    corr_cols.loc[corr_cols['Type'] == 'figurative', 'Type'] = 0
    corr_cols.loc[corr_cols['Outcome'] == 'upheld', 'Outcome'] = 1
    corr_cols.loc[corr_cols['Outcome'] == 'rejected', 'Outcome'] = 0
    corr_cols.loc[corr_cols['Opposition Outcome'] == 'upheld', 'Opposition Outcome'] = 2
    corr_cols.loc[corr_cols['Opposition Outcome'] == 'partially upheld', 'Opposition Outcome'] = 1
    corr_cols.loc[corr_cols['Opposition Outcome'] == 'rejected', 'Opposition Outcome'] = 0

    plt.rcParams['axes.grid'] = False
    corr = corr_cols.corr()

    # Create a mask
    mask = np.triu(np.ones_like(corr, dtype=bool))
    np.fill_diagonal(mask, False)

    plt.figure(figsize=(11,8))
    sns.set_theme(style='white')
    h = sns.heatmap(corr, mask=mask, center=0, annot=True,
                    fmt='.2f', square=True, cmap=sns.diverging_palette(316, 270, as_cmap=True))

    h.set_yticklabels(corr_cols.columns, rotation="horizontal")
    return h.figure


def get_class_distribution(df: pd.DataFrame):
    plot = df['Opposition Outcome'].value_counts().plot.pie(ylabel='', title="", legend=False,
                                                            autopct='%1.1f%%', wedgeprops={'linewidth': 3, 'edgecolor': 'white'},
                                                            colors=['#FFCE68', '#9789f3', '#FF87B1'], startangle=0)
    return plot.get_figure()


def get_detailed_class_distribution(df: pd.DataFrame):
    plot = df['Outcome'].value_counts().plot.pie(ylabel='', title="", legend=False,
                                                 autopct='%1.1f%%', wedgeprops={'linewidth': 3, 'edgecolor': 'white'},
                                                 colors=['#9789f3', '#FF87B1'], startangle=0)
    return plot.get_figure()


def get_missing_values_per_variable(df: pd.DataFrame):
    missing_values = {}
    df_size = len(df)
    for col in df.columns:
        missing_values[col] = {}
        abs_missing_values = df[col].isna().sum()
        missing_values[col]['absolute'] = int(abs_missing_values)
        missing_values[col]['relative'] = float(abs_missing_values) / float(df_size)
    return missing_values


def get_character_distribution_case_sensitive(df: pd.DataFrame):
    word_marks = df.loc[df['Type'] == 'word']
    alphabet = {}
    contested_marks_list = word_marks['Contested Trademark']
    earlier_marks_list = word_marks['Earlier Trademark']
    all_word_marks = list(set(contested_marks_list + earlier_marks_list))

    for m in all_word_marks:
        for c in m:
            if c not in alphabet:
                alphabet[c] = 1
            else:
                alphabet[c] = alphabet[c] + 1

    alphabet = {k: v for k, v in sorted(alphabet.items(), key=lambda item: item[1], reverse=True)}
    fig = plt.figure(figsize=(11,4))
    plt.bar(alphabet.keys(), alphabet.values(), color='#9789f3')
    plt.xlabel('Character')
    plt.ylabel('Occurrences')
    plt.xticks(fontsize=8)
    return fig


def get_character_distribution_case_insensitive(df: pd.DataFrame):
    word_marks = df.loc[df['Type'] == 'word']
    alphabet = {}
    contested_marks_list = word_marks['Contested Trademark']
    earlier_marks_list = word_marks['Earlier Trademark']
    all_word_marks = list(set(contested_marks_list + earlier_marks_list))

    for m in all_word_marks:
        for c in m:
            c = c.lower()
            if c not in alphabet:
                alphabet[c] = 1
            else:
                alphabet[c] = alphabet[c] + 1

    alphabet = {k: v for k, v in sorted(alphabet.items(), key=lambda item: item[1], reverse=True)}
    fig = plt.figure(figsize=(11,4))
    plt.bar(alphabet.keys(), alphabet.values(), color='#9789f3')
    plt.xlabel('Character')
    plt.ylabel('Occurrences')
    plt.xticks(fontsize=8)
    return fig


def get_word_mark_statistics(df: pd.DataFrame):
    word_marks = df.loc[df['Type'] == 'word']
    contested_marks_list = word_marks['Contested Trademark']
    earlier_marks_list = word_marks['Earlier Trademark']
    all_word_marks = list(set(contested_marks_list + earlier_marks_list))
    word_mark_lengths = [len(w) for w in all_word_marks]
    w_stats = {
        'mean': float(np.mean(word_mark_lengths)),
        'median': int(np.median(word_mark_lengths)),
        'std': float(np.std(word_mark_lengths)),
        'min': int(np.min(word_mark_lengths)),
        'max': int(np.max(word_mark_lengths)),
        'var': float(np.var(word_mark_lengths)),
        '.25-q': int(np.quantile(word_mark_lengths, q=.25)),
        '.75-q': int(np.quantile(word_mark_lengths, q=.75))
    }
    return w_stats


def get_word_mark_statistics_boxplot(df: pd.DataFrame):
    word_marks = df.loc[df['Type'] == 'word']
    contested_marks_list = word_marks['Contested Trademark']
    earlier_marks_list = word_marks['Earlier Trademark']
    all_word_marks = list(set(contested_marks_list + earlier_marks_list))
    word_mark_lengths = [len(w) for w in all_word_marks]
    plt.figure(figsize=(11,4))
    sns.set_theme(style='white')
    h = sns.boxplot(x=word_mark_lengths)
    h.set_xticks(range(min(word_mark_lengths), max(word_mark_lengths)+1, 5), labels=range(min(word_mark_lengths), max(word_mark_lengths)+1, 5))
    return h.figure


def get_word_mark_character_positions_case_insensitive(df: pd.DataFrame):
    word_marks = df.loc[df['Type'] == 'word']
    character_positions_insensitive = {}
    contested_marks_list = word_marks['Contested Trademark']
    earlier_marks_list = word_marks['Earlier Trademark']
    all_word_marks = list(set(contested_marks_list + earlier_marks_list))
    for m in all_word_marks:
        for i, c in enumerate(m):
            c = c.lower()
            if c not in character_positions_insensitive:
                character_positions_insensitive[c] = [i]
            else:
                character_positions_insensitive[c].append(i)
    character_positions_insensitive = {k: v for k, v in sorted(character_positions_insensitive.items(), key=lambda item: np.std(item[1]), reverse=True)}
    plt.figure(figsize=(11,4))
    sns.set_theme(style='white')
    h = sns.boxplot(data=character_positions_insensitive)
    return h.figure


def get_word_mark_character_positions_case_sensitive(df: pd.DataFrame):
    word_marks = df.loc[df['Type'] == 'word']
    character_positions = {}
    contested_marks_list = word_marks['Contested Trademark']
    earlier_marks_list = word_marks['Earlier Trademark']
    all_word_marks = list(set(contested_marks_list + earlier_marks_list))
    for m in all_word_marks:
        for i, c in enumerate(m):
            if c not in character_positions:
                character_positions[c] = [i]
            else:
                character_positions[c].append(i)
    character_positions = {k: v for k, v in sorted(character_positions.items(), key=lambda item: np.std(item[1]), reverse=True)}
    plt.figure(figsize=(11,4))
    sns.set_theme(style='white')
    h = sns.boxplot(data=character_positions)
    return h.figure


def get_variable_distribution(df: pd.DataFrame):
    columns = ['Visual Similarity', 'Aural Similarity', 'Conceptual Similarity', 'Degree of Attention', 'Distinctiveness', 'Item Similarity']
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.ylabel('Score')
    plt.xlabel('Factor')
    sns.boxplot(data = df[columns])
    return fig


def get_image_aspect_ratios(img_path: str, target_path: str):
    plt.figure(figsize=(11, 4))
    plt.xlabel('Width (px)')
    plt.ylabel('Height (px)')

    for im in glob(f'{img_path}/*'):
        im_path = im.replace('\\', '/')
        img = cv2.imread(im_path)
        (h, w) = img.shape[:2]
        plt.plot(w, h, 'b.', markersize=2)

    plt.savefig(f'{target_path}/image_aspect_ratios.png', bbox_inches='tight', pad_inches=0.3)


def export_statistics(df: pd.DataFrame, img_path: str, target_path: str):
    rcParams.update({'figure.autolayout': True})
    params = {'legend.fontsize': 'large',
              'axes.labelsize': 'large',
              'axes.titlesize':'large',
              'xtick.labelsize':'large',
              'ytick.labelsize':'large',
              'font.size': 18}
    pylab.rcParams.update(params)
    store_statistics([
        get_number_of_samples_per_class_and_type,
        get_class_distribution,
        get_detailed_class_distribution,
        get_correlation_matrix,
        get_missing_values_per_variable,
        get_character_distribution_case_sensitive,
        get_character_distribution_case_insensitive,
        get_word_mark_statistics,
        get_word_mark_statistics_boxplot,
        get_word_mark_character_positions_case_sensitive,
        get_word_mark_character_positions_case_insensitive,
        get_variable_distribution
    ], df, target_path)
    get_image_aspect_ratios(img_path, target_path)