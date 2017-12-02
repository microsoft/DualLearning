import os
def MapDeviceIds(comm):
    rank = comm.Get_rank()
    num_machine = comm.Get_size()
    os.system('python GPU_Usage.py ' + str(rank) + ' > record' + str(rank))
    comm.Barrier()
    if rank == 0:
        os.system('python AllocateGPU.py ' + str(num_machine) + ' > DirtyRecord')
    comm.Barrier()
    cardid = str(0)
    with open('DirtyRecord', 'r') as f:
        for idx, line in enumerate(f):
            if idx == rank:
                cardid = line.strip()
                break
            
    return cardid
