import numpy as np

class Spawner:
    """
    Handles time‐varying spawn periods for one pedestrian spawner.
    schedule: list of dicts, each with keys:
      - 'end'    : float, end time (inclusive)
      - 'period' : float, spawn period up to that end time
    """
    def __init__(self, pos, goal, schedule):
        self.pos        = np.array(pos)
        self.goal       = np.array(goal)
        self.schedule   = schedule
        self.last_spawn = 0.0

    def current_period(self, t):
        for entry in self.schedule:
            if t <= entry['end']:
                return entry['period']
        return self.schedule[-1]['period']
