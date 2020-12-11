# Eurydice 🤖🎓🎵

This project allows humans to teach a computer how to play music using Reinforcement Learning.

## Installation

You need to install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) for running this project.

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

This section explains how to run each of the functionalities available.

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

![](./docs/examples/qlearningauto.gif)

### Automated Q-learning w/ saving into a model

You can save the training into a file in order to continue training later.

`python run_music_generation.py --step 500 --episodes 2000 --model <model>`

![](./docs/examples/qlearningautomodel.gif)

### Interactive Q-learning w/ saving into model

Instead of using the automated policy, you can five the reward by yourself to the computer. You may want to use the `--model` for saving the progress that you have done so far.

`python run_music_generation.py --interactive_mode --model <model>`

![](./docs/examples/qlearninginteractivemodel.gif)

### Play optimal policy from a model

You can play optimal policy from a model.

`python run_music_generation.py --model <model> --results`

![](./docs/examples/qlearningmodelresults.gif)

### Automated Deep Q-learning

An approach using Neural Network for storing the rewards was created. Since results are not very promising yet, only automated testing is available for now.

`python run_music_generation.py --aprox_q_learning --step 500 --episodes 5000 --epsilon 0.3`

![](./docs/examples/deeplearningauto.gif)