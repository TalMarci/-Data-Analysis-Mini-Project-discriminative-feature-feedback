from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from typing import List, Tuple
import random

"""
differet modelst for the assignment, Teacher for part1 and Teacher_v2 for part 2.
"""


# abstract models:
@dataclass
class FeedBack:
    X: np.array
    y: int
    c: List[tuple]

    def get_phi(self) -> tuple:
        return self.c[0][0], self.c[0][1]

    def get_not_phi(self) -> tuple:
        return self.c[0][0], 1-self.c[0][1]

@dataclass
class TeacherFeedBack:
    student_answer: bool
    feed_back: FeedBack



@dataclass
class TAFeedBack:
    teacher_feed_back: TeacherFeedBack
    ind_of_x_hat: int


class Teacher(ABC):
    X: np.array
    y: np.array

    def __init__(self, X: np.array, y: np.array):
        self.X = X
        self.y = y

    @staticmethod
    def _discriminative_feature(X: np.array, X_hat: np.array) -> List[tuple]:
        d = len(X)
        res = []
        for i in range(d):
            if X[i] == 1 - X_hat[i]:
                res.append(tuple([i, X[i]]))
        return res

    @abstractmethod
    def teach(self, X: np.array, y: int, l: tuple) -> TeacherFeedBack:
        pass


class TeacherAssistance(ABC):
    X: np.array
    y: np.array
    teacher: Teacher
    reputation_map: np.array
    alpha: float

    def __init__(self, X: np.array, y: np.array, teacher: Teacher, alpha: int):
        m = len(X)
        self.X = X
        self.y = y
        self.teacher = teacher
        self.alpha = alpha
        self.reputation_map = np.full(m, 1 / m)

    @abstractmethod
    def _get_probability_vec(self, X_indices) -> List[float]:
        pass

    @abstractmethod
    def _set_reputation(self, response, chosen_x_ind):
        pass

    def assist(self, X: np.array, y: int, L_l: List[tuple]) -> TAFeedBack:
        X_indices = [np.flatnonzero((self.X == l[0]).all(1))[0] for l in L_l]
        P = self._get_probability_vec(X_indices)
        chosen_l_ind = np.random.choice(len(L_l), p=P)
        response = self.teacher.teach(X, y, L_l[chosen_l_ind])
        self._set_reputation(response, chosen_l_ind)
        return TAFeedBack(response, chosen_l_ind)


# Implementations of the abstract models:
class SimpleTeacher(Teacher):

    def teach(self, X: np.array, y: int, l: tuple) -> TeacherFeedBack:
        if y == l[1]:
            return TeacherFeedBack(True, None)
        discriminative_features: List[tuple] = self._discriminative_feature(X, l[0])
        feedback = FeedBack( X, y, [random.choice(discriminative_features)])
        return TeacherFeedBack(False, feedback)


class DiscriminativeTeacher(Teacher):

    def _getP(self, feature, label):
        label_indices = np.where(self.y == label + 1)[0]
        term1 = len([ind for ind in label_indices if self.X[ind][feature] == 1])

        p = (term1 / len(label_indices)) * 100
        return p

    def _build_Pmatrix(self):
        num_of_labels = len(np.unique(self.y))
        num_of_features = self.X[0].shape[0]
        Pmatrix = np.zeros((num_of_features, num_of_labels))
        for i in range(num_of_features):
            for j in range(num_of_labels):
                Pmatrix[i, j] = self._getP(i, j)
        return Pmatrix

    def _most_discriminative(self, discriminative_features: list, y: int):
        most_ind = -1
        most_p = 0
        most_f = -1
        for i, f in discriminative_features:
            if self.p_matrix[i][y - 1] > most_p:  # y is the correct label
                most_ind = i;
                most_p = self.p_matrix[i][y - 1]
                most_f = f
        return (most_ind, most_f)

    def __init__(self, X: np.array, y: np.array):
        super().__init__(X, y)
        self.p_matrix = None  # just for init
        self.p_matrix = self._build_Pmatrix()

    def teach(self, X: np.array, y: int, l: tuple) -> tuple:
        if y == l[1]:
            return TeacherFeedBack(True, None)
        discriminative_features: List[tuple] = self._discriminative_feature(X, l[0])
        feedback = FeedBack(X, y, [self._most_discriminative(discriminative_features, y)])
        return TeacherFeedBack(False, feedback)


class ExampleTA(TeacherAssistance):

    def _get_probability_vec(self, X_indices) -> List[float]:
        P = [self.reputation_map[ind] for ind in X_indices]
        divisor = sum(P)
        P = [p / divisor for p in P]
        return P

    def _set_reputation(self, response, chosen_x_ind):
        self.reputation_map[chosen_x_ind] *= 1 / self.alpha if response is not None else self.alpha
