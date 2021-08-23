#!/bin/bash
#SBATCH --job-name=extract_different_author_questions_per_post
#SBATCH --mail-user=ianbstew@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --time=10:00:00
#SBATCH --account=mihalcea1
#SBATCH --output=/home/%u/logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --mem-per-cpu=80g

OUT_DIR=../../data/reddit_data/
python extract_different_author_questions_per_post.py $OUT_DIR
