from monitor import *


runner = monitor()
print 'valid classifier', runner.valid_S2L()
print 'valid CLM:', runner.valid_L2S()