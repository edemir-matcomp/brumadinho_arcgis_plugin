import sys, time, os

'''

print('OS ENVIRON BEFORE\n')
for i in os.environ['PATH'].split(";"):
    print(i)


from os.path import join, abspath, dirname
base_path = dirname(dirname(abspath(__file__)))
os.environ['PATH'] = '%s%s' % (
    os.environ['PATH'],
    join(base_path, 'Library', 'bin'),
)

print('OS ENVIRON\n')
for i in os.environ['PATH'].split(";"):
    print(i)
#print(os.environ['PATH'])

'''

list_lib = [r'C:\Users\edemir\anaconda3\envs\arc105', \
     r'C:\Users\edemir\anaconda3\envs\arc105\Library\mingw-w64\bin', \
     r'C:\Users\edemir\anaconda3\envs\arc105\Library\usr\bin', \
     r'C:\Users\edemir\anaconda3\envs\arc105\Library\bin', \
     r'C:\Users\edemir\anaconda3\envs\arc105\Scripts', \
     r'C:\Users\edemir\anaconda3\envs\arc105\bin', \
     r'C:\Users\edemir\anaconda3\condabin']

for i in list_lib:
    os.environ['PATH'] = '%s;%s' % (os.environ['PATH'], i)


print('OS ENVIRON AFTER\n')
for i in os.environ['PATH'].split(";"):
    print(i)
    
#print(list_lib)

import rasterio


#print('\nSYS PATH\n')
#print(sys.path)
input()
#time.sleep(30)

#import rasterio
#print(sys.version)
#print('acabou')

#print(rasterio.__version__)



