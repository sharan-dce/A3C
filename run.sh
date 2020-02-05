#!/usr/bin/env bash

# rm  ./bin/checkpoints/* \
#     ./bin/logs/* \
#     ./bin/gifs/*
python3 __main__.py \
--learning_rate 0.0005 \
--environment Breakout-v4 \
--optimizer rms_prop \
--gamma 0.99 \
--checkpoint_dir ./bin/checkpoints/ \
--log_dir ./bin/logs \
--threads 4 \
--checkpoint_save_interval 1 \
--target_update_interval 128 \
--gifs_dir ./bin/gifs \
--gifs_save_interval 1 \
--checkpoint_path ./bin/checkpoints/AC_1
