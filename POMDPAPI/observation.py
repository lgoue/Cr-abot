from __future__ import print_function
from builtins import str
import numpy as np

N_IDEA_QUALITY = 5
INF = 1000000

class IdeaQuality(object):

    BAD=1
    MEDIUM = 2
    GOOD = 3
    NO = 4
    NA = 0



class Idea:
    def __init__(self, quality, bin_number=True):
        if bin_number:

            if quality == IdeaQuality.NO:

                self.quality = -2
            elif quality == IdeaQuality.NA:
                self.quality = 0
            else:
                self.quality = quality
            self.bin_number = quality
        else:

            if quality == -2:
                self.quality = -2
                self.bin_number = IdeaQuality.NO
            if quality == 0:
                self.quality = 0
                self.bin_number = IdeaQuality.NA
            else:
                self.quality = quality
                self.bin_number = quality +1

    def distance_to_observation(self, other_idea):
        if self.bin_number == IdeaQuality.NO:
            if other_idea.bin_number == IdeaQuality.NO:
                return 0
            else:
                return INF
        if self.bin_number == IdeaQuality.NA:
            if other_idea.bin_number == IdeaQuality.NA:
                return 0
            else:
                return INF
        if other_idea.bin_number == IdeaQuality.NO:
                return INF
        if other_idea.bin_number == IdeaQuality.NA:
                return INF
        else:

            return (self.quality - other_idea.quality)**2

    def to_string(self):
        match self.bin_number:

            case IdeaQuality.BAD:
                return "Bad Idea"
            case IdeaQuality.MEDIUM:
                return "MEDIUM_IDEA"
            case IdeaQuality.GOOD:
                return "Good Idea"
            case IdeaQuality.NO:
                return "No Idea"
            case IdeaQuality.NA:
                return "Idea quality not applicable"
        return 'Error : '+str(self.bin_number)
    def to_int(self):
        return self.bin_number

    def print(self):
        print(self.to_string())

class MoodObservation:

    def __init__(self, P, A):
        self.A = A
        self.P = P

    def distance_to(self, other_observation):
        return (
            abs(self.P - other_observation.P)
            + abs(self.A - other_observation.A)
        )

    def copy(self):
        return Observation(self.P, self.A)

    def __eq__(self, other_observation):
        return self.PAD == other_observation.PAD

    def __hash__(self):
        return self.PAD

    def print_observation(self):
        print(
            "Measured PAD = ("
            + str(self.P)
            + ","
            + str(self.A)
            + ","
        )

    def to_string(self):

        obs = (
            "Measured pleasure is"
            + str(self.P)
            + " measured arousal is"
            + str(self.A)
        )
        return obs
