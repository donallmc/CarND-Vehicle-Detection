import uuid
import math

class Vehicle:
    def __init__(self, box, parent_shape=(1280, 720)):
        self.box = box
        self.center = ((box[0][0] + box[1][0])/2, (box[0][1] + box[1][1])/2)
        self.parent_shape = parent_shape
        self.id = uuid.uuid4()
        self.frames_unseen = 0
        self.frames_seen = 0
        self.diffs = []
        self.recovered = 0

    def is_same(self, other):
        print("\t comparing other " + str(other.center) + " to this " + str(self.center))
        print(math.sqrt((other.center[0] - self.center[0])**2 + (other.center[1] - self.center[1])**2) < 64)
        return math.sqrt((other.center[0] - self.center[0])**2 + (other.center[1] - self.center[1])**2) < 64

    def reset_unseen(self):
        self.frames_unseen = 0

    def increase_unseen(self):
        self.frames_unseen += 1

    def increase_seen(self):
        self.frames_seen += 1
        
    def set_id(self, ident):
        self.id = ident

    def get_box(self):
        return ((int(self.center[0] - 48), int(self.center[1] - 48)), (int(self.center[0] + 48), int(self.center[1] + 48)))

    def update_box(self, box): #compute an average?
        new_center = ((box[0][0] + box[1][0])/2, (box[0][1] + box[1][1])/2)
        self.diffs.append((new_center[0] - self.center[0], new_center[1] - self.center[1]))
        self.box = box
        self.center = new_center

    def ghost_move(self):
        x_total = 0
        y_total = 0
        for diff in self.diffs:
            x_total += diff[0]
            y_total += diff[1]
        x_diff = x_total // len(self.diffs)
        y_diff = y_total // len(self.diffs)
        self.center = (self.center[0] + x_diff, self.center[1] + y_diff)        
