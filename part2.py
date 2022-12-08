from models import *
from data import *
from typing import List, Tuple
from matplotlib import pyplot as plt
import numpy as np
import os


def satisfy_conj(C, x_t):
    satisfies: bool = True
    for t in C:
        satisfies &= x_t[t[0]] == t[1]
    return satisfies


def shuffle_data(X, y):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]


def run_ten(data_getter, teacher):
    X, y = data_getter()
    err = 0
    for _ in range(10):
        X, y = shuffle_data(X, y)
        t = teacher(X, y)
        ta = ExampleTA(X, y, t, 1.4)

        err += run(X, y, ta)

    return err / 10


def run(X: np.array, y: np.array, ta: TeacherAssistance, title=None, name=None, plot=False):
    global satisfied
    x0, y0, X, y = X[0], y[0], X[1:, :], y[1:]  # todo do we really need to splice when using ta?
    l0 = (x0, y0, [])
    L: List[tuple] = []
    mistakes_count = 0
    errors = []

    for i in range(X.shape[0]):
        satisfied = False
        x_hat = []
        for l in L:  # l=[X,y, c],
            if satisfy_conj(l[2], X[i]):
                satisfied = True
                x_hat.append(l)
        if satisfied:
            res: TAFeedBack = ta.assist(X[i], y[i], x_hat)
            feed_back: feed_back = res.teacher_feed_back.feed_back
            if res.teacher_feed_back.student_answer is False:
                mistakes_count += 1
                x_hat[res.ind_of_x_hat][2].append(feed_back.get_not_phi())
                res_string = f"label: {y[i]}\nChosen discriminative feature: {feed_back.get_phi()}\n"

            else:
                res_string = f"label: {y[i]}\n"
            print(f"Example: {X[i]}\n"
                  f"Predicted label: {x_hat[res.ind_of_x_hat][1]}\n"
                  f"Explanation example: {x_hat[res.ind_of_x_hat][0]}\n"
                  f"Teacher response:\n{res_string}")
            errors.append(mistakes_count / (i + 1))
            x_hat.__delitem__(res.ind_of_x_hat)
            for l in x_hat:
                res: TeacherFeedBack = ta.teacher.teach(X[i], y[i], l)
                if res.student_answer is False:
                    l[2].append(res.feed_back.get_not_phi())

        else:
            res = ta.assist(X[i], y[i], [l0])  # could be replaced with empty list and have special behavior from ta
            feed_back: FeedBack = res.teacher_feed_back.feed_back
            if res.teacher_feed_back.student_answer is False:
                mistakes_count += 1
                L.append((feed_back.X, feed_back.y, feed_back.c,))
                res_string = f"label: {y[i]}\nChosen discriminative feature: {feed_back.get_phi()}\n"
            else:
                res_string = f"label: {y[i]}\n"
            print(f"Example: {X[i]}\n"
                  f"Predicted label: {l0[1]}\n"
                  f"Explanation example: {l0[0]}\n"
                  f"Teacher response:\n{res_string}")
            errors.append(mistakes_count / (i + 1))
    print(f"The algorithm has mistaken {mistakes_count * 100 / (len(X) + 1)}% of the times.")
    print("decision list\n")
    for l in L:
        print(f"does X satisfies {l[2]} ? --> label = {l[1]}\n")
        print("       |\n       |\n\n")
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
    return errors[-1] * 100


X, y = get_zoo_data()
X, y = shuffle_data(X, y)
st = SimpleTeacher(X, y)
ta = ExampleTA(X, y, st, 1.4)
run(X, y, ta, "part 2: results of simple teacher on zoo data", "simple_zoo_pt2", plot=True)

X, y = get_zoo_data()
X, y = shuffle_data(X, y)
dt = DiscriminativeTeacher(X, y)
ta = ExampleTA(X, y, dt, 1.4)
run(X, y, ta, "part 2: results of discriminative teacher on zoo data", "disc_zoo_pt2", plot=True)

X, y = get_nursery_data()
X, y = shuffle_data(X, y)
st = SimpleTeacher(X, y)
ta = ExampleTA(X, y, st, 1.4)
run(X, y, ta, "part 2: results of simple teacher on nursery data", "simple_nursery_pt2", plot=True)

X, y = get_nursery_data()
X, y = shuffle_data(X, y)
dt = DiscriminativeTeacher(X, y)
ta = ExampleTA(X, y, dt, 1.4)
run(X, y, ta, "part 2: results of Discriminative teacher on nursery data", "disc_nursery_pt2", plot=True)

X, y = get_mushrooms_data()
X, y = shuffle_data(X, y)
dt = SimpleTeacher(X, y)
ta = ExampleTA(X, y, st, 1.4)
run(X, y, ta, "part 2: results of simple teacher on zoo data", "simple_mushroom_pt2", plot=True)

X, y = get_mushrooms_data()
X, y = shuffle_data(X, y)
dt = DiscriminativeTeacher(X, y)
ta = ExampleTA(X, y, dt, 1.4)
run(X, y, ta, "part 2: results of discriminative teacher on mushroom data", "disc_mushroom_pt2", plot=True)

# Average calculation, not necessary to run
# print(f"\nerror average of simple teacher on zoo data:  {run_ten(get_zoo_data, SimpleTeacher)}\n"
#       f"error average of discriminative teacher on zoo data:  {run_ten(get_zoo_data, DiscriminativeTeacher)}\n"
#       f"error average of simple teacher on nursery data:  {run_ten(get_nursery_data, SimpleTeacher)}\n"
#       f"error average of discriminative teacher on nursery data:  {run_ten(get_nursery_data, DiscriminativeTeacher)}\n"
#       f"\nerror average of simple teacher on mushroom data:  {run_ten(get_mushrooms_data, SimpleTeacher)}\n"
#       f"error average of discriminative teacher on mushroom data:  {run_ten(get_mushrooms_data, DiscriminativeTeacher)}"
#       f"\n")


print("File names:")
print("simple_zoo_pt2")
print("disc_zoo_pt2")
print("simple_nursery_pt2")
print("disc_nursery_pt2")
print("simple_mushroom_pt2")

