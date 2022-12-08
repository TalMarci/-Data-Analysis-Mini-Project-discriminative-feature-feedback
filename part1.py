from models import *
from data import *
from typing import List, Tuple
from matplotlib import pyplot as plt
import numpy as np


def satisfy_conj(C, x_t):
    satisfies: bool = True
    for t in C:
        satisfies &= x_t[t[0]] == t[1]
    return satisfies


def shuffle_data(X, y):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]


def run(X: np.array, y: np.array, teacher, title=None, name=None, plot=False):
    global satisfied
    x0, y0, X, y = X[0], y[0], X[1:, :], y[1:]  # todo check splice
    l0 = (x0, y0, [])
    L: List[tuple] = []
    mistakes_count = 0
    errors = []

    for i in range(X.shape[0]):
        satisfied = False
        for l in L:  # l=[X,y, c],
            if satisfy_conj(l[2], X[i]):
                satisfied = True
                res: TeacherFeedBack = teacher.teach(X[i], y[i], l)
                if res.student_answer is False:
                    mistakes_count += 1
                    conj = res.feed_back.get_not_phi()
                    l[2].append(conj)
                    res_string = f"label: {y[i]}\nChosen discriminative feature: {res.feed_back.get_phi()}\n"

                else:
                    res_string = f"label: {y[i]}\n"
                print(f"Example: {X[i]}\n"
                      f"Predicted label: {l[1]}\n"
                      f"Explanation example: {l[0]}\n"
                      f"Teacher response:\n{res_string}")
                errors.append(mistakes_count / (i + 1))
                break
        if satisfied: continue
        res = teacher.teach(X[i], y[i], l0)
        if res.student_answer is False:
            mistakes_count += 1
            L.append((res.feed_back.X, res.feed_back.y, res.feed_back.c,))
            res_string = f"label: {y[i]}\nChosen discriminative feature: {res.feed_back.get_phi()}\n"
        else:
            res_string = f"label: {y[i]}\n"
        print(f"Example: {X[i]}\n"
              f"Predicted label: {l0[1]}\n"
              f"Explanation example: {l0[0]}\n"
              f"Teacher response:\n{res_string}")
        errors.append(mistakes_count / (i + 1))
    print(f"The algorithm has mistaken {mistakes_count * 100 / (len(X) + 1)}% of the times.\n")
    print("decision list")
    for l in L:
        print(f"does X satisfies {l[2]} ? --> label = {l[1]}")
        print("       |\n       |\n       V")
    print("unknown label")

    # plot
    if plot:
        plt.plot(np.arange(X.shape[0]), errors, label="error", color="red")
        plt.title(title)
        plt.xlabel("iterations")
        plt.ylabel("Error")
        # plt.xticks(xlabels)
        # plt.legend()
        plt.savefig(f"/tmp/{name}.png")
        plt.clf()
        # plt.show()
    return errors[-1] * 100


def run_ten(data_getter, teacher):
    X, y = data_getter()
    err = 0
    for _ in range(10):
        X, y = shuffle_data(X, y)
        t = teacher(X, y)
        err += run(X, y, t)

    return err / 10


X, y = get_zoo_data()
X, y = shuffle_data(X, y)
dt = SimpleTeacher(X, y)
run(X, y, dt, "part 1: results of simple teacher on zoo data", "simple_zoo_pt1", plot=True)

X, y = get_zoo_data()
X, y = shuffle_data(X, y)
dt = DiscriminativeTeacher(X, y)
run(X, y, dt, "part 1: results of discriminative teacher on zoo data", "disc_zoo_pt1", plot=True)

X, y = get_nursery_data()
X, y = shuffle_data(X, y)
dt = SimpleTeacher(X, y)
run(X, y, dt, "part 1: results of simple teacher on nursery data", "simple_nursery_pt1", plot=True)

X, y = get_nursery_data()
X, y = shuffle_data(X, y)
dt = DiscriminativeTeacher(X, y)
run(X, y, dt, "part 1: results of Discriminative teacher on nursery data", "disc_nursery_pt1", plot=True)

# Average calculation, not necessary to run
# print(f"\nerror average of simple teacher on zoo data:  {run_ten(get_zoo_data, SimpleTeacher)}\n"
#       f"error average of discriminative teacher on zoo data:  {run_ten(get_zoo_data, DiscriminativeTeacher)}\n"
#       f"error average of simple teacher on nursery data:  {run_ten(get_nursery_data, SimpleTeacher)}\n"
#       f"error average of discriminative teacher on nursery data:  {run_ten(get_nursery_data, DiscriminativeTeacher)}\n")

print("Files names:")
print("simple_zoo_pt1")
print("disc_zoo_pt1")
print("simple_nursery_pt1")
print("disc_nursery_pt1")
