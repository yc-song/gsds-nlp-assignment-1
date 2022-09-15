#!/usr/bin/env python

# Save parameters every a few SGD iterations as fail-safe
SAVE_PARAMS_EVERY =5000

import pickle
import glob
import random
import numpy as np
import os.path as op

def load_saved_params():
    """
    A helper function that loads previously saved parameters and resets
    iteration start.
    """
    st = 0
    for f in glob.glob("saved_params_*.npy"):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter > st):
            st = iter

    if st > 0:
        param0_file = "saved_params_%d_inside.npy" % st
        param1_file = "saved_params_%d_outside.npy" % st
        state_file = "saved_state_%d.pickle" % st

        param_0 = np.load(param0_file, allow_pickle=True)
        param_1 = np.load(param1_file, allow_pickle=True)
        with open(state_file, "rb") as f:
            state = pickle.load(f)


        params = (param_0, param_1)
        return st, params, state
    else:
        return st, None, None


def save_params(iter, params):
    for i, param in enumerate(params):
        name = "inside" if i ==0 else "outside"
        params_file = "saved_params_%d_%s.npy" % (iter, name)
        np.save(params_file, params[i])
    with open("saved_state_%d.pickle" % (iter), "wb") as f:
        pickle.dump(random.getstate(), f)


def sgd(f, x0, step, iterations, postprocessing=None, useSaved=False,
        PRINT_EVERY=10):
    """ Stochastic Gradient Descent

    Implement the stochastic gradient descent method in this function.

    Arguments:
    f -- the function to optimize, it should take a single
         argument and yield two outputs, a loss and the gradient
         with respect to the arguments
    x0 -- the initial point to start SGD from
    step -- the step size for SGD
    iterations -- total iterations to run SGD for
    postprocessing -- postprocessing function for the parameters
                      if necessary. In the case of word2vec we will need to
                      normalize the word vectors to have unit length.
    PRINT_EVERY -- specifies how many iterations to output loss

    Return:
    x -- the parameter value after SGD finishes
    """

    # Anneal learning rate every several iterations
    ANNEAL_EVERY = 20000

    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)

        if state:
            random.setstate(state)
    else:
        start_iter = 0

    x = list(x0)

    if not postprocessing:
        postprocessing = lambda x: x

    exploss = None
    for iter in range(start_iter + 1, iterations + 1):
        # You might want to print the progress every few iterations.

        loss = None
        loss, grad = f(x)
        x[0] = x[0] - step * grad[0]
        x[1] = x[1] - step * grad[1]

        x = postprocessing(x)
        if iter % PRINT_EVERY == 0:
            if not exploss:
                exploss = loss
            else:
                exploss = .95 * exploss + .05 * loss
            print("iter %d: %f" % (iter, exploss))

        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x)

        if iter % ANNEAL_EVERY == 0:
            step *= 0.5


    return x



