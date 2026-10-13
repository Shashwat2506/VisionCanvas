class GestureRecognizer:
    GESTURE_MAP = {
        (0, 1, 0, 0, 0): "DRAW",
        (0, 1, 1, 0, 0): "SELECT",
        (0, 1, 1, 1, 0): "UNDO",
        (0, 1, 1, 1, 1): "REDO",
        (1, 0, 0, 0, 0): "SAVE",
    }

    def recognize(self, finger_states):
        if not finger_states:
            return "NONE"

        key = tuple(finger_states)
        return self.GESTURE_MAP.get(key, "NONE")