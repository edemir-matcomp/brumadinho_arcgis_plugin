import os
from skimage.io import imread, imsave
import numpy as np

def recursive_glob(rootdir=".", suffix=".tif"):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot,filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

# A function that returns the number of pixels from class:
def myFunc(e):
  return e[1]

list_images = recursive_glob('/mnt/DADOS_GRENOBLE_1/edemir/BRUMADINHO/1_PREPARE_DATA/results/1_PREPARE_DATA/raw_data/masks')

InRaster_Reference = '/mnt/DADOS_GRENOBLE_1/edemir/BRUMADINHO/0_CREATE_MASK/results/mask.tif'

# Get number of classes
mask = imread(InRaster_Reference)

list_classes = np.unique(mask)[:-1].astype(int)
num_classes = len(list_classes)

print(list_classes)


#list_classes = list(range(11))

dict_paths = {}
dict_sum = []

# Initialize empty lists
for i in list_classes:
    dict_paths[i] = []
    dict_sum.append(0)

for img_path in list_images:

    # Read Mask Image
    img = imread(img_path)

    # Count each label pixel in image
    labels, count = np.unique(img,return_counts=True)

    #print(labels)
    #print(count)
    # Fill dict with patches that contains at least one class
    for l, c in zip(labels, count):
        if l != 254.0:
            dict_paths[l].append((img_path, c))

# Count total frequency of each class
for key in dict_paths:
    for pair in dict_paths[key]:
        dict_sum[key] += pair[1]
    #print(key, dict_sum[key])

sorted_classes = np.argsort(dict_sum)
#print(np.argsort(dict_sum))
#print(np.sort(dict_sum))

K_FOLD = 5
dict_fold = {}
#summed = [0,0,0,0,0]
summed = {}
for id_fold in range(K_FOLD):
    dict_fold[id_fold] = []
    
for i in list_classes:
    summed[i] = [0,0,0,0,0]





# Usar set

last_list = []
#Iteração partindo da classe de menor numero de amostras
for label in sorted_classes:

    print("NUMBER OF SAMPLES FROM CLASS {}: {}".format(label, len(dict_paths[label])))
    # Para cada imagem que possui essa classe, distribuir em folds diferentes
    # A cada iteração, verificar a quantidade de pixels da classe no fold
    # Sempre escolher fold que tenho a menor quantidade de amostras
    
    # Remove samples already used, look for path
    if len(last_list) != 0:
        for sample in last_list:
            for item in dict_paths[label]:
                if item[0] == sample[0]:
                    try:
                        dict_paths[label].remove(item)
                    except:
                        continue
        
    
    # Sort by counter
    dict_paths[label].sort(reverse=True, key=myFunc)
        
    for index, res in enumerate(dict_paths[label]):
    
        #Check for every folds
        next_fold = np.argmin(summed[label])
      
        # Set on fold
        dict_fold[next_fold].append(res)
        
        summed[label][next_fold] += res[1]
        
        
    #TODO: Idealmente deveria mudar o contador da outra classe que nao esta sendo analisada mas ocorreu
    
    # Keep last list to remove in next iteration
    last_list += dict_paths[label].copy()
    
    
#mylist = list(dict.fromkeys(last_list))
#print('sem repeticao\n', len(mylist))

#import pprint
#for i in mylist:
#    print(i)
   
#print(len(last_list))
#exit(1)
    
    
    #for index, res in enumerate(tmp_list_sorted):
         #print(index, res)
    #    pos = index % 5
    #    dict_fold[pos] = res[1]
    
    #for id_fold in range(K_FOLD):
    #    print("FOLD {}".format(id_fold))
    #    print(dict_fold[id_fold])
        
    #for i in summed:
    #    print(i/sum(summed))
    
    #exit(1)
#'''

#Save files
fp = open('stratified_folds_jan','w+')
for id_fold in range(K_FOLD):
    for img_path in dict_fold[id_fold]:
        fp.write('{},{}\n'.format(img_path[0],id_fold))
        


import pprint

new_summed = {}
for i in list_classes:
    new_summed[i] = [0,0,0,0,0]

for id_fold in range(K_FOLD):
    #print("FOLD {}".format(id_fold))
    #print(len(dict_fold[id_fold]))
    
    #pprint.pprint(dict_fold[id_fold])
    
    #Compute Again Proportions
    
    for img_path in dict_fold[id_fold]:
        
        # Read Mask Image
        print(img_path)
        img = imread(img_path[0])

        # Count each label pixel in image
        labels, count = np.unique(img,return_counts=True)
        
        # Fill dict with patches that contains at least one class
        for l, c in zip(labels, count):
            if l != 254.0:
                new_summed[l][id_fold] += c
                #dict_paths[l].append((img_path, c))
        
    
        
        #print(sorted_classes)
        #for label in sorted_classes:
        #    print(summed[label][id_fold])
        #    #print(summed[label][id_fold]/sum(summed[label]))
        #'''


# Summary of Split
for id_fold in range(K_FOLD):
    print('K-FOLD {}'.format(id_fold))
    for i in list_classes:
        #print(new_summed[i][id_fold])
        print(new_summed[i][id_fold]/sum(new_summed[i]))

        
# Summary of Split
for id_fold in range(K_FOLD):
    print('K-FOLD {}'.format(id_fold))
    for i in list_classes:
        #print(new_summed[i][id_fold])
        print(new_summed[i][id_fold])
    
    
#print(new_summed)




    
