SUBMISSION_DIR=/local2/lbiester/pushshift/submissions/
SUBREDDITS=('AmItheAsshole' 'legaladvice' 'pcmasterrace' 'Advice' 'personalfinance')
OUT_DIR='../../data/reddit_data/'
SUBMISSION_DATES=(2018 1 2019 12)
python collect_submissions_from_subreddits.py $SUBMISSION_DIR --subreddits "${SUBREDDITS[@]}" --out_dir $OUT_DIR --submission_dates "${SUBMISSION_DATES[@]}"