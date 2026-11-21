"""Reproduces adr/0017's os._exit()-vs-HDF5-consistency measurement.

Empirical check: does an abrupt process death mid-write corrupt an emcee
HDFBackend file (unreadable afterward), or just lose the in-progress step
(file still opens fine, reflecting only the last FULLY completed step)?

Tests four kill points within save_step's own write sequence, and two kill
mechanisms (os._exit() -- skips __exit__/finally entirely -- vs a normal
uncaught Python exception, which does NOT skip them), by actually forking a
child process, replicating HDFBackend.save_step's own internal sequence to
die partway through, and checking whether the parent can still open+read the
file afterward.

Needs emcee (not a project dependency -- pip install emcee):

    pip install emcee
    python adr/0017_hdf5_crash_experiment.py
"""
import os
import sys
import numpy as np
import emcee
from emcee.backends.hdf import HDFBackend

NWALKERS, NDIM = 8, 3


def make_state(step):
    coords = np.full((NWALKERS, NDIM), float(step))
    log_prob = np.full(NWALKERS, -float(step))
    accepted = np.ones(NWALKERS, dtype=bool)
    random_state = np.random.RandomState(step).get_state()
    return emcee.State(coords, log_prob=log_prob, blobs=None, random_state=random_state), accepted


def run_child(path, kill_point, kill_mechanism):
    """Runs in the forked child: does 5 normal steps, then on step 6 dies
    partway through save_step at kill_point, via kill_mechanism."""
    backend = HDFBackend(path, name="mcmc")
    backend.reset(NWALKERS, NDIM)
    backend.grow(6, None)
    for step in range(5):
        state, accepted = make_state(step)
        backend.save_step(state, accepted)

    # Patch save_step to die partway through, mirroring its real internal
    # structure (same open("a")/write-fields/close sequence).
    state, accepted = make_state(5)
    backend._check(state, accepted)

    def die():
        if kill_mechanism == "os_exit":
            os._exit(1)
        else:
            raise RuntimeError("simulated mid-write failure (normal exception)")

    with backend.open("a") as f:
        g = f[backend.name]
        iteration = g.attrs["iteration"]
        if kill_point == "before_chain_write":
            die()
        g["chain"][iteration, :, :] = state.coords
        if kill_point == "after_chain_before_logprob":
            die()
        g["log_prob"][iteration, :] = state.log_prob
        if kill_point == "after_logprob_before_iteration_attr":
            die()
        g["accepted"][:] += accepted
        for i, v in enumerate(state.random_state):
            g.attrs["random_state_{0}".format(i)] = v
        g.attrs["iteration"] = iteration + 1
        if kill_point == "after_full_write_before_close":
            die()
    # If we get here without dying, something's wrong with the test itself.
    os._exit(0)


def check_file_after(path):
    """Returns (openable, iteration_reported, chain_at_last_iter_is_sane)."""
    try:
        backend = HDFBackend(path, name="mcmc", read_only=True)
        it = backend.iteration
        chain = backend.get_value("chain")
        return True, it, chain.shape
    except Exception as exc:
        return False, None, repr(exc)


results = []
for kill_mechanism in ["os_exit", "normal_exception"]:
    for kill_point in ["before_chain_write", "after_chain_before_logprob",
                      "after_logprob_before_iteration_attr", "after_full_write_before_close"]:
        path = f"/tmp/emcee_crash_test_{kill_mechanism}_{kill_point}.h5"
        if os.path.exists(path):
            os.remove(path)

        pid = os.fork()
        if pid == 0:
            # child -- for "normal_exception", let the RuntimeError propagate
            # naturally all the way up (stack unwinding runs the with-block's
            # __exit__/f.close() along the way, same as a real uncaught
            # exception would); for "os_exit" it never gets the chance to.
            # Suppress the traceback noise but don't swallow/alter control flow.
            run_child(path, kill_point, kill_mechanism)
            os._exit(0)
        else:
            _, status = os.waitpid(pid, 0)
            openable, iteration, detail = check_file_after(path)
            results.append((kill_mechanism, kill_point, os.WEXITSTATUS(status) if os.WIFEXITED(status) else f"signal {os.WTERMSIG(status)}",
                           openable, iteration, detail))

print(f"{'mechanism':18s} {'kill_point':38s} {'child_exit':12s} {'openable':9s} {'iteration':10s} detail")
for mechanism, kill_point, exit_info, openable, iteration, detail in results:
    print(f"{mechanism:18s} {kill_point:38s} {str(exit_info):12s} {str(openable):9s} {str(iteration):10s} {detail}")
