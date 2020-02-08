#!/usr/bin/env bash

# rm  ./bin/checkpoints/* \
#     ./bin/logs/* \
#     ./bin/gifs/*
python3 __main__.py \
--learning_rate 0.01 \
--gradient_clipping 5.0 \
--environment Breakout-v0 \
--gamma 0.99 \
--checkpoint_dir ./bin/checkpoints/ \
--log_dir ./bin/logs \
--threads 2 \
--critic_coefficient 0.35 \
--checkpoint_save_interval 1 \
--update_intervals 5 \
--gifs_dir ./bin/gifs \
--gifs_save_interval 1 \
--render
# --checkpoint_path ./bin/checkpoints/AC_1
