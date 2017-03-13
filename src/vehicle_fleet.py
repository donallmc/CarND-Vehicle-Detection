from vehicle import Vehicle

'''
Tracks vehicles seen on the road. 

Keeps a list of vehicles it can currently see and adds/removes them as necessary.
Should handle some false positives by only allowing vehicles to enter in certain positions
(for now, the position is bottom right of the image because I'm overfitting to the project
but it could be extended to bring in from all sides at specific boundaries (ideally based
on recognising the shape of the road).

It also will handle vehicle continuity. If a vehicle drops out unexpectedly it will continue
to display a best-guess for its current position until it's matched again.
'''
class VehicleFleet:
    def __init__(self, container_shape=(720, 1280)):
        self.tracking = {}
        self.width = container_shape[1]
        self.height = container_shape[0]

    def process_new_data(self, boxes):
        box_output = []
        false_positives = []
        accounted_for = []
        for box in boxes:
            match = self.match_existing(box)
            if match:
                self.update_vehicle(match, box)
                box_output.append(match.get_box())
                accounted_for.append(match.id)
            elif self.is_incoming(box):
                new_vehicle = Vehicle(box)
                self.tracking[new_vehicle.id] = new_vehicle
                box_output.append(new_vehicle.get_box())
                accounted_for.append(new_vehicle.id)
            else:
                false_positives.append(box)
        dropped = self.handle_dropped_vehicles(accounted_for)
        return box_output + dropped

    def update_vehicle(self, vehicle, box):
        vehicle.update_box(box)
        vehicle.increase_seen()

    #this could be made more elegant by supporting multiple matches
    def match_existing(self, box):
        match = None
        new = Vehicle(box)
        for v_id, vehicle in self.tracking.items():
            if vehicle.is_same(new):
                match = vehicle
        return match

    def is_incoming(self, box):
        center = ((box[0][0] + box[1][0])/2, (box[0][1] + box[1][1])/2)
        return (self.width - center[0]) < 400 and (self.width - center[0]) > 150

    def handle_dropped_vehicles(self, seen):
        box_output = []
        to_delete = []
        for v_id, vehicle in self.tracking.items():
            if v_id not in seen:
                #attempt recovery
                #the recovery criteria are pretty hacky. Ideally it would be based on a number of factors including movement,
                #the presence of low-quality recognitions where we expect the vehicle to be, road recognition (to account for vehicles
                #exiting), etc.
                if vehicle.frames_seen > 10 and vehicle.recovered < 12 and not self.is_incoming(vehicle.get_box()):
                    vehicle.recovered += 1
                    vehicle.ghost_move()
                    box_output.append(vehicle.get_box())
                else:
                    #no recovery. delete it
                    to_delete.append(v_id)
        for v_id in to_delete:
            del self.tracking[v_id]
        return box_output
                    
            
            
    
