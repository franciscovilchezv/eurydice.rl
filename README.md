# Eurydice 🤖🎓🎵

This project allows humans to teach a computer how to play music using Reinforcement Learning.

## Installation

You may want to install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) for running this project.

### Create your conda enviroment

`conda create --name <environment_name> --file spec-file.txt`

<!-- 
File generated with:
conda list --explicit > spec-file.txt
 -->

### Activate your conda environment

`conda activate <environment_name>`

### Pip packages

Since some dependencies are not stored in the conda environment, we need to install them with `pip`.

`pip install -r requirements.txt`

<!-- 
File generated with:
pip freeze > requirements.txt
 -->

## Execute

This section explains how to run each of the functionalities available. Code is available in the [source](./source) directory.

```
$ cd source
$ python run_music_generation.py --help
  usage: run_music_generation.py [-h] [--epsilon EPSILON] [--discount DISCOUNT] 
  [--episodes EPISODES] [--model MODEL] [--step STEP] [--interactive_mode]
  [--aprox_q_learning] [--batch_size BATCH_SIZE] [--results]

  optional arguments:
    -h, --help            show this help message and exit
    --epsilon EPSILON     randomness in action
    --discount DISCOUNT   learning rate
    --episodes EPISODES   number of training episodes
    --model MODEL         loads and persists model in file
    --step STEP           visualize results after a number of steps
    --interactive_mode    interact with user for learning
    --aprox_q_learning    use NN for aproximate q-learning
    --batch_size BATCH_SIZE
                          size of each NN batch
    --results             plays the best result so far
```

### Automated Q-learning

Run Q-learning algorithm with an automated reward which tries to teach the computer the [descending C major scale](https://www.allaboutmusictheory.com/major-scale/c-major-scale/).

![](https://www.mymusictheory.com/images/stories/grade2/5/c-desc.jpg)

`python run_music_generation.py --step 500 --episodes 2000`

![](https://user-images.githubusercontent.com/9200906/101970071-30b39080-3bf6-11eb-9c68-ab14af598f03.gif)

### Automated Q-learning saving learning into a file

You can save the training into a file in order to continue training later.

`python run_music_generation.py --step 500 --episodes 2000 --model <model>`

![](https://user-images.githubusercontent.com/9200906/101970072-314c2700-3bf6-11eb-8fd8-e13cd34750ab.gif)

### Interactive Q-learning saving learning into a file

Instead of using the automated policy, you can give the reward by yourself to the computer. You may want to use the `--model` for saving the progress that you have done so far.

`python run_music_generation.py --interactive_mode --model <model>`

![](https://user-images.githubusercontent.com/9200906/101970074-327d5400-3bf6-11eb-8b04-776c8956b950.gif)

### Play optimal policy from a file

You can play optimal policy from a file with training progress.

`python run_music_generation.py --model <model> --results`

![](https://user-images.githubusercontent.com/9200906/101970076-33ae8100-3bf6-11eb-990f-3d17419e2a31.gif)

###### Play the lick

No one wants to hear [The Lick](https://youtu.be/krDxhnaKD7Q?t=63) again, but you can do it reproducing the results we have in `thelick` model.

`python run_music_generation.py --model thelick --results`

### Automated Deep Q-learning

An approach using Neural Network for storing the rewards was created. Since results are not very promising yet, only automated testing is available for now.

`python run_music_generation.py --aprox_q_learning --step 500 --episodes 5000 --epsilon 0.3`

![](https://user-images.githubusercontent.com/9200906/101970075-3315ea80-3bf6-11eb-832b-e5a91687f95e.gif)