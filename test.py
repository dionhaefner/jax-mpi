#!/usr/bin/env python

"""
Run with

$ mpirun -n <nproc> python test.py
"""

import jax
import jax.numpy as np
from jax_mpi.sum_inplace import sum_inplace_jax_primitive

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def test_nojit(a):
    return sum_inplace_jax_primitive(a, comm)


@jax.jit
def test_jit(a):
    return sum_inplace_jax_primitive(a, comm)


if __name__ == '__main__':
    arr = np.ones((2, 2))

    res = test_nojit(arr)
    if rank == 0:
        print('No JIT: ', end='')
        print('✓' if np.array_equal(res, arr * size) else '✗')

    res = test_jit(arr)
    if rank == 0:
        print('JIT: ', end='')
        print('✓' if np.array_equal(res, arr * size) else '✗')
