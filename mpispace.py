import odl
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
comm_size = comm.size


class MPISpace(odl.space.base_ntuples.FnBase):
    def __init__(self, size, dtype='float', impl='numpy'):
        self.local_size = size // comm_size + int(rank < (size % comm_size))
        starts = np.cumsum([size // comm_size + int(ranki < (size % comm_size))
                            for ranki in range(comm_size)]).tolist()
        self.index_start = np.array([0] + starts[:-1])
        self.local_start = self.index_start[rank]
        self.local_space = odl.rn(self.local_size, dtype=dtype, impl=impl)
        odl.space.base_ntuples.FnBase.__init__(self, size, dtype)

    def _multiply(self, x1, x2, out):
        self.local_space.multiply(x1.local_data, x2.local_data, out.local_data)

    def _divide(self, x1, x2, out):
        self.local_space.divide(x1.local_data, x2.local_data, out.local_data)

    def _lincomb(self, a, x1, b, x2, out):
        self.local_space.lincomb(a, x1.local_data,
                                 b, x2.local_data,
                                 out.local_data)

    def _inner(self, x1, x2):
        inners = np.empty(comm_size)
        inners[rank] = self.local_space.inner(x1.local_data, x2.local_data)
        return sum(comm.allreduce(inners))

    def element(self, inp=None):
        return MPISpaceElement(self, self.local_space.element(inp))

    def one(self):
        return MPISpaceElement(self, self.local_space.one())

    def zero(self):
        return MPISpaceElement(self, self.local_space.one())


class MPISpaceElement(odl.space.base_ntuples.FnBaseVector):
    def __init__(self, space, local_data):
        self.local_data = local_data
        odl.space.base_ntuples.FnBaseVector.__init__(self, space)

    def __getitem__(self, index):
        in_rank = np.nonzero(self.space.index_start <= index)[0][-1]
        if in_rank == rank:
            local_index = index - self.space.local_start
            result = np.array([self.local_data[local_index]])
        else:
            result = np.empty(1)
        comm.Bcast(result, root=in_rank)
        return self.space.field.element(result)

    def __setitem__(self, index, value):
        in_rank = np.nonzero(self.space.index_start <= index)[0][-1]
        if in_rank == rank:
            local_index = index - self.space.local_start
            self.local_data[local_index] = value

    def asarray(self):
        rcvbuffer = np.empty(self.space.size)
        comm.Allgatherv(self.local_data.asarray(), rcvbuffer)
        return rcvbuffer

n = 10**8
space = MPISpace(n)

el = space.one()
el[3] = 5
print(el[5])
print(np.asarray(el))
print(el.norm())

with odl.util.Timer('mpi'):
    print(el.norm())

if rank == 0:
    fn = odl.rn(n)
    fn_el = fn.one()

    with odl.util.Timer('fn'):
        print(fn_el.norm())
