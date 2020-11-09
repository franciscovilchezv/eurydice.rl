from environments.music_world import MusicWorld
from visualizer.music_visualizer import InteractiveComposer

from argparse import ArgumentParser

def main():
  parser: ArgumentParser = ArgumentParser()
  parser.add_argument('--epsilon', type=float, default=0, help="randomness in action")
  parser.add_argument('--discount', type=float, default=1, help="learning rate")
  parser.add_argument('--episodes', type=int, default=5000, help="number of training episodes")
  parser.add_argument('--model', type=str, default="", help="loads and persists model in file")
  parser.add_argument('--step', type=int, default=100, help="visualize results after a number of steps")

  args = parser.parse_args()

  env : MusicWorld = MusicWorld()

  viz: InteractiveComposer = InteractiveComposer(env)

  viz.q_learning(args.epsilon, args.discount, args.episodes, args.model, args.step)

if __name__ == "__main__":
  main()