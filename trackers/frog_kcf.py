import cv2


class FrogKCFTracker():

    def __init__(self, id):
        self.id = id
        self.tracker = cv2.TrackerKCF_create()
        self.current_track = None
        self.track_records = []
    
    def init_tracker(self, frame, frame_id, bbox):
        self.tracker.init(frame, bbox)
        track_record = {}
        track_record['frame_id'] = frame_id
        track_record['bbox'] = bbox
        self.track_records.append(track_record)
        self.current_track = track_record

    
    def update_tracker(self, frame, frame_id):
        ok, bbox = self.tracker.update(frame)
        track_record = {}
        track_record['frame_id'] = frame_id
        track_record['bbox'] = bbox
        self.track_records.append(track_record)
        self.current_track = track_record
        return self.id, ok, bbox
