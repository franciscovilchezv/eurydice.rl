from environments.music_world import MusicWorld
from visualizer.music_visualizer import InteractiveComposer

from argparse import ArgumentParser

def main():
  parser: ArgumentParser = ArgumentParser()
  parser.add_argument('--epsilon', type=float, default=0.1, help="randomness in action")
  parser.add_argument('--discount', type=float, default=1, help="learning rate")
  parser.add_argument('--episodes', type=int, default=5000, help="number of training episodes")
  parser.add_argument('--model', type=str, default="", help="loads and persists model in file")
  parser.add_argument('--step', type=int, default=100, help="visualize results after a number of steps")
  parser.add_argument('--interactive_mode', action="store_true", help="interact with user for learning")
  parser.add_argument('--aprox_q_learning', action="store_true", help="use NN for aproximate q-learning")
  parser.add_argument('--batch_size', type=int, default=100, help="size of each NN batch")
  parser.add_argument('--results', action="store_true", help="plays the best result so far")

  args = parser.parse_args()

  env : MusicWorld = MusicWorld(args.interactive_mode)

  viz: InteractiveComposer = InteractiveComposer(env, args.model)

  if(not args.results):
    if (args.aprox_q_learning):
      viz.deep_q_learning(args.epsilon, args.discount, args.batch_size, args.episodes, args.step)
    else:
      viz.q_learning(args.epsilon, args.discount, args.episodes, args.step)
  else:
    viz.greedy_policy_vis(8)

if __name__ == "__main__":
  main()