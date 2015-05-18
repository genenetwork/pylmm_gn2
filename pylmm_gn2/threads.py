
numThreads = None

def setNumThreads(num):
   global numThreads
   numThreads = num

def single():
   global numThreads
   return numThreads is not None and numThreads == 1

def multi():
   return not single()
