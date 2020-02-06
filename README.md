# Asynchronous Advantage Actor Critic (A3C)
Paper Link: https://arxiv.org/abs/1602.01783  
Asynchronously updates Policy and Value Nets by training episodes in parallel  

## Running
To create logs, checkpoints and gifs directories, run  
bash ./refresh.sh  
Then,
chmod 777 ./run.sh  
./run  
## Change Config
In run.sh, all arguments passed to the __main__.py script are listed. Modify them as you wish. The environment argument expects a standard gym environment which can be built using gym.make(). Parameters have been shared between the Policy and Value nets. I am using 4 threads for my testing, but use 16 if possible (my lappy cant take more :P).    
Will add samples once fine tuning and training is done.
