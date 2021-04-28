AUTHOR_DIR=../../data/reddit_data/author_data/
OUT_DIR=../../data/reddit_data/author_data/
EMBED_TYPE='subreddit'
#EMBED_TYPE='text'
(python generate_author_embeddings.py $AUTHOR_DIR $OUT_DIR --embed_type $EMBED_TYPE)&
PID=$!
MAX_MEMORY=30000000000 # 30G
prlimit --pid $PID --as=$MAX_MEMORY