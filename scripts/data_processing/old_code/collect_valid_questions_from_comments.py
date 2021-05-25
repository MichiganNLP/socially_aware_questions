"""
Collect valid questions from comments
based on (1) semantic overlap with post;
(2) clarification question structure.
More analysis here: look_for_valid_questions_in_comments.ipynb
"""
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('comment_dir')
    parser.add_argument('post_file')
    parser.add_argument('--out_dir', default='../../data/reddit_data/')

if __name__ == '__main__':
    main()