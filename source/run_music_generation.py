from environments.music_world import MusicWorld
from visualizer.music_visualizer import InteractiveComposer

from argparse import ArgumentParser

def main():
  parser: ArgumentParser = ArgumentParser()
  parser.add_argument('--epsilon', type=float, default=0, help="randomness in action")
  parser.add_argument('--discount', type=float, default=1, help="learning rate")
  args = parser.parse_args()

  env : MusicWorld = MusicWorld()

  viz: InteractiveComposer = InteractiveComposer(env)

  viz.q_learning(args.epsilon, args.discount)

if __name__ == "__main__":
  main()