from typing import List, Tuple
from constants.note import Note, Symbol

class CompositionState():
  def __init__(self, composition_notes: List[Note]):
    self.composition_notes = composition_notes

  def __hash__(self):
    return hash(tuple(self.composition_notes))

  def __eq__(self, other):
    return self.composition_notes == other.composition_notes

  def __repr__(self):
    return str(self.composition_notes)

  def __str__(self):
    return str(self.composition_notes)

  def transition_note(self, new_note: Note) -> List[Note]:
    assert (Symbol.NAN in self.composition_notes), "No available spot for more notes"

    composition = []
    note_added = False
    for note in self.composition_notes:
      if note == Symbol.NAN and not(note_added):
        composition.append(new_note)
        note_added = True        
      else:
        composition.append(note)
    
    return composition

  # TODO: change to reward based on user feedback
  def get_reward_for_transition(self, note: Note):
    # print("Request user reward for transition")
    prev_note = Symbol.NAN
    for composition_note in self.composition_notes:
      if (composition_note != Symbol.NAN):
        prev_note = composition_note

    if (prev_note == Symbol.NAN):
      return 0.0
    elif (prev_note > note):
      return 1.0
    
    return -100.0

class MusicWorld():
  def get_actions(self) -> List[Note]:
    # TODO: Let's start with just a major scale one octave
    return list(Note)

  def is_terminal(self, state: CompositionState) -> bool:
    return (not Symbol.NAN in state.composition_notes) # TODO: Size of composition

  def sample_transition(self, state: CompositionState, action: Note) -> Tuple[CompositionState, float]:
    next_composition_notes: List[int] = state.transition_note(action)
    reward: float = state.get_reward_for_transition(action)

    state_next: CompositionState = CompositionState(next_composition_notes)

    return state_next, reward