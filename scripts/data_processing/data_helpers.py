"""
Data helper functions.
"""
import pandas as pd
import re
import os
def assign_label_by_cutoff_pct(data, label_var='gender', min_pct=0.75):
    """
    Assign one label per value based on
    cutoff value; e.g. assign "MALE" to name
    if >=X% of name belongs to "MALE".

    :param data:
    :param label_var:
    :param min_pct:
    :return:
    """
    data.sort_values('count_pct', inplace=True, ascending=False)
    cutoff_data = data[data.loc[:, 'count_pct'] >= min_pct]
    label = 'UNK'
    if (cutoff_data.shape[0] > 0):
        label = cutoff_data.iloc[0, :].loc[label_var]
    return label

def load_name_gender_data(name_data_dir):
    """
    Load per-name gender data from
    directory of Social Security
    birth name records.

    :param name_data_dir:
    :return:
    """
    # let's get all names from likely periods of birth for comments, i.e. 1930-2000
    name_data_matcher = re.compile('yob19[3-9][0-9]')
    name_data_files = list(filter(lambda x: name_data_matcher.search(x), os.listdir(name_data_dir)))
    name_data_files = list(map(lambda x: os.path.join(name_data_dir, x), name_data_files))
    name_data = pd.concat(list(map(lambda x: pd.read_csv(x, sep=',', header=None, index_col=False), name_data_files)),
                          axis=0)
    name_data.columns = ['name', 'gender', 'count']
    # group by name, get raw count
    name_count_data = name_data.groupby(['name', 'gender']).apply(
        lambda x: x.loc[:, 'count'].sum()).reset_index().rename(columns={0: 'count'})
    # # get gender percent
    name_gender_data = name_count_data.groupby('name').apply(
        lambda x: x.assign(**{'count_pct': x.loc[:, 'count'] / x.loc[:, 'count'].sum()}).drop(['count', ],
                                                                                              axis=1)).reset_index(
        drop=True)

    min_gender_pct = 0.75
    name_gender_label_data = []
    for name_i, data_i in name_gender_data.groupby('name'):
        label_i = assign_label_by_cutoff_pct(data_i, label_var='gender', min_pct=min_gender_pct)
        name_gender_label_data.append([name_i, label_i])
    name_gender_label_data = pd.DataFrame(name_gender_label_data, columns=['name', 'gender'])
    # lowercase for consistency
    name_gender_label_data = name_gender_label_data.assign(
        **{'name': name_gender_label_data.loc[:, 'name'].apply(lambda x: x.lower())})
    return name_gender_label_data

def extract_name(text, camel_matcher):
    """
    Extract name from raw text. Assume either "first_name last_name" or "FirstnameLastname".

    :param text:
    :param camel_matcher:
    :return: name
    """
    name = text
    text_tokens = text.split(' ')
    if(len(text_tokens) > 0):
        name = text_tokens[0]
    elif(camel_matcher.search(text) is not None):
        name = camel_matcher.search(text).group(0)
    name = name.lower()
    return name