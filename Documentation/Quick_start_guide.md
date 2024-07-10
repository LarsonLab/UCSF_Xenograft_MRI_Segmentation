Quick start guide for adapting torch_main to run different models, losses, optimizers etc. 

1) change weights and metrics save paths uder global variables 
2) change model name to desired model at commented #2
3) change model in the next line below to desired model call 
4) adjust epochs, and learning rate in subsequent lines if necessary 
5) fill in scheduler and loss names if changes in the scheduler or loss used in a given training run 
6) change weights save path to desired weights path under commented #3