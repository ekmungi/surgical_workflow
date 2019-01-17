import numpy as np
import os, sys
import pandas as pd
import time, textwrap


if __name__ == "__main__":
    
    

    while True:
        os.system("cat /proc/cpuinfo | grep \"MHz\" > /home/anant/Downloads/proc.txt")
        data = pd.read_csv('/home/anant/Downloads/proc.txt', sep=":", header=None)
        freq = np.round(data.values[:,1].transpose().astype(np.double)/1000, 2)

        sys.stdout.flush()
        sys.stdout.write("{0} \r".format(textwrap.fill(str(freq), 300)))
        
        
        time.sleep(1)

