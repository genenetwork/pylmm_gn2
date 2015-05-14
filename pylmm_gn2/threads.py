
numThreads = None

def single():
   global numThreads
   return numThreads is not None and numThreads != 1

def multi():
   return not single()
