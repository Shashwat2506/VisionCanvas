class FingerDetector:
    TIP_IDS = [4, 8, 12, 16, 20]

    def get_finger_states(self, hand_landmarks):
        if not hand_landmarks:
            return []

        fingers = []

        # Thumb: compare x of tip (4) vs joint below it (3)
        if hand_landmarks[4].x < hand_landmarks[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other 4 fingers: compare y of tip vs y of PIP joint (2 below tip)
        for tip_id in self.TIP_IDS[1:]:
            if hand_landmarks[tip_id].y < hand_landmarks[tip_id - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers