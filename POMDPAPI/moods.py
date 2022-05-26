N_MOOD = 5


class MoodType(object):
    """
    Enumerates the OCC emotions
    """

    RELAXED = 5
    HAPPY = 1
    SAD = 2
    ANGRY = 3
    NEUTRAL = 4
    SURPRISED=0


class Mood:
    def __init__(self, type):

        self.bin_number = type
        self.m = 0.5

    def distance_to_observation(self, observation):
        p, a, d = self.PAD_values()
        return (
            (observation.P - p) ** 2
            + (observation.A - a) ** 2
            + (observation.D - d) ** 2
        )


    def distance_to(self, mood):
        if isinstance(mood, int):
            mood = Mood(mood)
        p, a, d = self.PAD_values()
        pe, ae, de = mood.PAD_values()
        return (pe - p) ** 2 + (ae - a) ** 2 + (de - d) ** 2

    def PAD_values(self):
        p, a,d = "Error", "Error", "Error"
        m = self.m
        match self.bin_number:
            case MoodType.HAPPY:
                p, a,d= 0.76, 0.48,0.35
            case MoodType.SAD:
                p, a,d = -0.63, 0.27,-0.33
            case MoodType.ANGRY:
                p, a,d = -0.43, 0.67,0.34
            case MoodType.RELAXED:
                p, a,d = m, -m,m
            case MoodType.NEUTRAL:
                p, a,d = 0, 0,0
            case MoodType.SURPRISED:
                p, a,d = 0.4, 0.67,-0.13
        return p, a,d

    def to_string(self):
        e = "Unknown"
        match self.bin_number:
            case MoodType.HAPPY:
                e = "Happy"
            case MoodType.SAD:
                e = "Sad"
            case MoodType.RELAXED:
                e = "Relaxed"
            case MoodType.ANGRY:
                e = "Angry"
            case MoodType.Neutral:
                e = "Neutral"
        return e

        def print(self):
            print(self.to_string())