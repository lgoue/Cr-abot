from __future__ import print_function
from builtins import str
import numpy as np
from moods import MoodType, Mood

INF = 1000000


class MoodObservation:

    def __init__(self, hap, sad,ang,sur,ne):
        self.hap = hap
        self.sad = sad
        self.ang = ang
        self.sur = sur
        self.ne = ne

        self.P,self.A,self.D = self.PAD_values()

    def distance_to(self, other_observation):
        return (
            abs(self.P - other_observation.P)**2
            + abs(self.A - other_observation.A)**2
            + abs(self.D - other_observation.D)**2
        )

    def PAD_values(self):

        hap_p,hap_a,hap_d = Mood(MoodType.HAPPY).PAD_values()
        sad_p,sad_a,sad_d = Mood(MoodType.SAD).PAD_values()
        ang_p,ang_a,ang_d = Mood(MoodType.ANGRY).PAD_values()
        sur_p,sur_a,sur_d = Mood(MoodType.SURPRISED).PAD_values()
        a =  self.sad*sad_a+self.ang*ang_a+self.sur*sur_a + self.hap*hap_a
        p =  self.sad*sad_p+self.ang*ang_p+self.sur*sur_p+ self.hap*hap_p
        d =  self.sad*sad_d+self.ang*ang_d+self.sur*sur_d+ self.hap*hap_d
        return p, a, d


    def copy(self):
        return MoodObservation(self.hap ,self.sad,self.ang,self.sur,self.ne)

    def __eq__(self, other_observation):
        return (self.hap == other_observation.hap) & (self.sad == other_observation.sad) & (self.ang == other_observation.ang) & (self.sur == other_observation.sur) & (self.ne == other_observation.ne)


    def print_observation(self):
        print(
            "Measured hap = ("
            + str(self.hap)
            + ", sad ="
            + str(self.sad)
            + ", ang ="
            + str(self.ang)
            + ", sur ="
            + str(self.sur)
            + ", ne ="
            + str(self.ne)
            + ")"
        )

    def to_string(self):

        obs = (
            "Measured hap = ("
            + str(self.hap)
            + ", sad ="
            + str(self.sad)
            + ", ang ="
            + str(self.ang)
            + ", sur ="
            + str(self.sur)
            + ", ne ="
            + str(self.ne)
            + ")"
        )
        return obs
