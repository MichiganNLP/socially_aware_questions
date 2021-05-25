"""
Extract author data from comments:

- location
- prior comment count
- prior comment length

Tested here: scripts/data_processing/label_author_status.ipynb
"""
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import os
from data_helpers import round_date_to_day
import geocoder
from nltk.tokenize import WordPunctTokenizer
from data_helpers import clean_text_matchers
import re
from pandarallel import pandarallel

def geocode_country(text):
    """
    Find best match for text in OSM database.

    :param text: location text
    :return: country code of location match
    """
    text_loc_country = 'UNK'
    try:
        text_loc = geocoder.osm(text, method='geocode')
        text_loc_data = text_loc.response.json()
        if (len(text_loc_data) > 0):
            text_loc_country = text_loc_data[0]['address']['country_code']
    except Exception as e:
        print(f'geocoding error {e}')
    return text_loc_country

def compute_prior_comment_count(full_data, date_var='date_day', author_var='userID'):
    """
    Compute prior number of comments/day per author.
    """
    comment_count_per_author = []
    for (author_i, date_i), data_i in full_data.groupby([author_var, date_var]):
        prior_data_i = full_data[(full_data.loc[:, 'userID']==author_i) &
                                 (full_data.loc[:, date_var] <= date_i)]
        date_range_i = (prior_data_i.loc[:, date_var].max() - prior_data_i.loc[:, date_var].min()).days + 1 # "smooth" to avoid div-by-0
        comment_count_i = prior_data_i.shape[0] / date_range_i
        comment_count_per_author.append([author_i, date_i, comment_count_i])
    comment_count_per_author = pd.DataFrame(comment_count_per_author, columns=[author_var, date_var, 'prior_comment_count'])
    return comment_count_per_author
def compute_prior_comment_length(full_data, tokenizer, text_var='commentBody', date_var='date_day', author_var='userID'):
    """
    Compute prior length of comment per author.
    """
    comment_len_per_author = []
    # get tokens
    full_data = full_data.assign(**{
        'comment_tokens' : full_data.loc[:, text_var].apply(tokenizer.tokenize)
    })
    for (author_i, date_i), data_i in full_data.groupby([author_var, date_var]):
        prior_data_i = full_data[(full_data.loc[:, 'userID']==author_i) &
                                 (full_data.loc[:, date_var] <= date_i)]
        comment_len_i = prior_data_i.loc[:, 'comment_tokens'].apply(lambda x: len(x))
        mean_comment_len_i = comment_len_i.mean()
        comment_len_per_author.append([author_i, date_i, mean_comment_len_i])
    comment_len_per_author = pd.DataFrame(comment_len_per_author, columns=[author_var, date_var, 'prior_comment_len'])
    return comment_len_per_author

def main():
    parser = ArgumentParser()
    parser.add_argument('comment_data_dir') # ../../data/nyt_comments/
    args = vars(parser.parse_args())
    comment_data_dir = args['comment_data_dir']

    # load comments
    month_year_pairs = [
        ('Jan', '2017'),
        ('Feb', '2017'),
        ('March', '2017'),
        ('April', '2017'),
        ('Jan', '2018'),
        ('Feb', '2018'),
        ('March', '2018'),
        ('April', '2018'),
    ]

    comment_data = []
    comment_data_cols = ['articleID', 'commentBody', 'commentID', 'commentType', 'createDate',
                         'depth', 'parentID', 'recommendedFlag',
                         'reportAbuseFlag', 'sectionName',
                         'userID', 'userDisplayName', 'userLocation']
    for month_i, year_i in month_year_pairs:
        comment_data_file_i = os.path.join(comment_data_dir, f'Comments{month_i}{year_i}.csv')
        comment_data_i = pd.read_csv(comment_data_file_i, sep=',', index_col=False, usecols=comment_data_cols)
        comment_data.append(comment_data_i)
    comment_data = pd.concat(comment_data, axis=0)
    # add date
    comment_data = comment_data.assign(**{
        'date_day': comment_data.loc[:, 'createDate'].apply(lambda x: round_date_to_day(x))
    })
    # print(datetime.fromtimestamp(comment_data.loc[:, 'createDate'].iloc[0]))

    ## comment count
    date_var = 'date_day'
    author_var = 'userID'
    comment_count_per_author = compute_prior_comment_count(comment_data, date_var=date_var, author_var=author_var)

    ## comment length
    # clean HTML`
    text_var = 'commentBody'
    matcher_pairs = [
        (re.compile('<.+>'), ' '),  # HTML
    ]
    word_tokenizer = WordPunctTokenizer()
    comment_data = comment_data.assign(**{
        text_var: comment_data.loc[:, text_var].apply(lambda x: clean_text_matchers(x, word_tokenizer, matcher_pairs))
    })
    comment_len_per_author = compute_prior_comment_length(comment_data, word_tokenizer,
                                                          text_var=text_var, date_var=date_var,
                                                          author_var=author_var)
    # combine
    author_var = 'userID'
    round_date_var = 'date_day'
    combined_comment_author_data = pd.merge(comment_count_per_author, comment_len_per_author,
                                            on=[author_var, round_date_var])
    # log-transform
    comment_vars = ['prior_comment_count', 'prior_comment_len']
    for comment_var in comment_vars:
        combined_comment_author_data = combined_comment_author_data.assign(**{
            f'log_{comment_var}': np.log(combined_comment_author_data.loc[:, comment_var])
        })
    # convert to bins
    comment_cutoff_pct = 50
    comment_cutoff_count = [
        np.percentile(combined_comment_author_data.loc[:, 'log_prior_comment_count'].values, comment_cutoff_pct)]
    comment_cutoff_len = [
        np.percentile(combined_comment_author_data.loc[:, 'log_prior_comment_len'].values, comment_cutoff_pct)]
    combined_comment_author_data = combined_comment_author_data.assign(**{
        'prior_comment_count_bin': np.digitize(combined_comment_author_data.loc[:, 'log_prior_comment_count'],
                                               comment_cutoff_count),
        'prior_comment_len_bin': np.digitize(combined_comment_author_data.loc[:, 'log_prior_comment_len'],
                                             comment_cutoff_len),
    })

    ## location
    comment_location_data = comment_data.loc[:, ['userID', 'userLocation']]
    author_locations = comment_location_data.loc[:, 'userLocation'].unique()
    # print(author_locations)
    # geocode all locations with OSM
    # DO NOT use parallel w/ too many workers => rate limit
    num_workers = 2
    pandarallel.initialize(progress_bar=True, nb_workers=num_workers)
    ## WARNING: takes ~ 12 hours for 40K locations
    author_location_countries = author_locations.parallel_apply(geocode_country)

    author_location_data = pd.DataFrame([
        author_locations,
        author_location_countries,
    ], index=['userLocation', 'location_country']).transpose()
    # fix common mistakes
    # pd.set_option('display.max_rows', 100)
    # top_k_locations = comment_data.loc[:, 'userLocation'].value_counts().head(100).index.tolist()
    # display(author_location_data[author_location_data.loc[:, 'userLocation'].isin(top_k_locations)])
    gold_location_countries = {
        'New York City': 'us',
        'NY NY': 'us',
        'Earth': 'UNK',
        '</br>': 'UNK',
        'CA': 'us',
        'PA': 'us',
        'MA': 'us',
        'Here': 'UNK',
        'DC': 'us',
    }
    author_location_data = author_location_data.assign(**{
        'location_country': author_location_data.apply(
            lambda x: gold_location_countries[x.loc['userLocation']] if x.loc['userLocation'] in gold_location_countries else x.loc['location_country'], axis=1)
    })
    # bin into US vs. non-US
    author_region_data = pd.DataFrame([
        ['us'],
        ['US'],
    ], index=['location_country', 'location_region']).transpose()
    # add UNK countries
    unk_countries = list(set(author_location_data.loc[:, 'location_country'].unique()) - set(['us']))
    unk_region_data = pd.DataFrame([unk_countries, ['non_US', ] * len(unk_countries)],
                                   index=['location_country', 'location_region']).transpose()
    author_region_data = pd.concat([
        author_region_data,
        unk_region_data,
    ], axis=0)
    author_location_data = pd.merge(author_location_data, author_region_data, on='location_country')
    # add author ID
    author_location_data = pd.merge(author_location_data.loc[:, ['userLocation', 'location_country', 'location_region']],
                                    comment_data.loc[:, ['userID', 'userLocation']].drop_duplicates('userID'),
                                    on='userLocation')
    # add locations to author data
    combined_comment_author_data = pd.merge(combined_comment_author_data, author_location_data, on='userID')

    ## save
    out_dir = args['out_dir']
    # fix author ID
    combined_comment_author_data = combined_comment_author_data.assign(**{
        'userID' : combined_comment_author_data.loc[:, 'userID'].astype(int)
    })
    comment_author_data_file = os.path.join(out_dir, 'author_comment_social_data.tsv')
    combined_comment_author_data.to_csv(comment_author_data_file, sep='\t', index=False)

if __name__ == '__main__':
    main()