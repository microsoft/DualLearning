import re
import os
import socket
import sys

filename = r'.\gpu_usage_draft_'
default_gpu = 58 + 30



def GrabGPU(rank):
    cmdstr = '\"C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe\" > ' + filename + rank
    os.system(cmdstr)

def GetGPUUSage(rank):
    pattern = re.compile(r'(?P<num>[0-9]{1,5})MiB[\s]+/')
    id = 0
    GPUs = []
    fo = open(filename + rank, 'r')
    for line in fo:
        result = pattern.search(line)
        if result:
            if int(result.group("num")) < default_gpu:
                GPUs.append(id)
            id = id + 1
    fo.close()

    print len(GPUs)
    for gpu in GPUs:
        print gpu


if __name__ == '__main__':
    rank = sys.argv[1]
    GrabGPU(rank)
    print socket.gethostname()
    GetGPUUSage(rank)
    #os.system('del /q ' + filename + rank)
