from enum import Enum
import musicalbeeps

player = musicalbeeps.Player(volume = 0.3, mute_output = False)

class Symbol(Enum):
  NAN: int = -1

  def __bool__(self):
    return self != self.NAN

  def play(self):
    return

class Note(Enum):
  C4: int = 0
  D4: int = 1
  E4: int = 2
  F4: int = 3
  G4: int = 4
  A4: int = 5
  B4: int = 6
  C5: int = 7

  def __lt__(self, other):
    return self.value < other.value

  def __le__(self, other):
    return self.value <= other.value

  def __gt__(self, other):
    return self.value > other.value

  def __ge__(self, other):
    return self.value >= other.value

  def play(self):
    player.play_note(self.name, 0.4)

