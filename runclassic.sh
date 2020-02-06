#!/usr/bin/env bash

# rm  ./bin/checkpoints/* \
#     ./bin/logs/* \
#     ./bin/gifs/*
python3 __main__.py \
--learning_rate 0.01 \
--environment CartPole-v1 \
--gradient_clipping 5.0 \
--gamma 0.99 \
--log_dir ./bin1/logs \
--threads 1 \
--update_intervals 5 \
--render \
--critic_coefficient 0.1
# --checkpoint_path ./bin/checkpoints/AC_1
