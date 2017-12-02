import re, os, numpy, sys


filename = r'.\gpu_usage_draft'



def GrabGPU():
    cmdstr = '\"C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe\" > ' + filename
    os.system(cmdstr)

def GetGPUUSage():
    pattern = re.compile(r'(?P<num>[0-9]+)MiB[\s]+/')
    mem = []
    fo = open(filename, 'r')
    for line in fo:
        result = pattern.search(line)
        if result:
            mem.append(int(result.group('num')))
    fo.close()

    return numpy.array(mem).argsort()[0]

def print_script(cmd):
    GrabGPU()
    with open('worker.bat', 'w') as f:
        f.write('@echo off\nsetlocal ENABLEDELAYEDEXPANSION\n')  
        if len(cmd) == 1:
            f.write('set THEANO_FLAGS=device=gpu%d\n' % GetGPUUSage())  
            f.write('python ' + cmd[0]) 
        elif len(cmd) == 2:
            f.write('set THEANO_FLAGS=device=gpu' + cmd[1] + '\n')  
            f.write('python ' + cmd[0]) 
        

if __name__ == '__main__':
    print_script(sys.argv[1:])
    
    # os.system('del /q ' + filename + rank)
