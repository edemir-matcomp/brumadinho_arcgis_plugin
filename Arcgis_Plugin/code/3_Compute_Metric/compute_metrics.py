import os, time

mydir = os.path.split(os.path.abspath(__file__))[0]
homepath = os.path.expanduser(os.getenv('USERPROFILE'))
#print('Meu diretorio: {}'.format(mydir))
#print('Minha Home: {}'.format(homepath))

list_lib = [r'{}\anaconda3\envs\arc105'.format(homepath), \
     r'{}\anaconda3\envs\arc105\Library\mingw-w64\bin'.format(homepath), \
     r'{}\anaconda3\envs\arc105\Library\usr\bin'.format(homepath), \
     r'{}\anaconda3\envs\arc105\Library\bin'.format(homepath), \
     r'{}\anaconda3\envs\arc105\Scripts'.format(homepath), \
     r'{}\anaconda3\envs\arc105\bin'.format(homepath), \
     r'{}\anaconda3\condabin'.format(homepath)]


#list_lib = [r'C:\Users\edemir\anaconda3\envs\arc105', \
#     r'C:\Users\edemir\anaconda3\envs\arc105\Library\mingw-w64\bin', \
#     r'C:\Users\edemir\anaconda3\envs\arc105\Library\usr\bin', \
#     r'C:\Users\edemir\anaconda3\envs\arc105\Library\bin', \
#     r'C:\Users\edemir\anaconda3\envs\arc105\Scripts', \
#     r'C:\Users\edemir\anaconda3\envs\arc105\bin', \
#     r'C:\Users\edemir\anaconda3\condabin']

for i in list_lib:
    os.environ['PATH'] = '%s;%s' % (i, os.environ['PATH'])

#for i in os.environ['PATH'].split(';'):
#    print(i)

import argparse
import numpy as np
import shapefile
import rasterio

from rasterio.features import rasterize
from shapely.geometry import Polygon
from shapely.ops import cascaded_union

from skimage.io import imread, imsave
import rasterio

from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, cohen_kappa_score, f1_score
import statistics
import sys

# Allow print matrix without truncation
np.set_printoptions(threshold=sys.maxsize)

def calculate_metrics_cm(cm, file = None):
    b_acc = balanced_accuracy_score_from_cm(cm)
    kappa = cohen_kappa_from_cm(cm)
    f1 = f1_score_from_cm(cm)	
    if file is not None:
        #file.write("Accuracy: " + str(acc) + "\n")
        file.write("Accuracy per Class:\n")
        for index, c in enumerate(b_acc[0]):
            file.write('Class {}: {}\n'.format(str(index), str(c)))
        file.write("Balanced_Accuracy: " + str(b_acc[1]) + "\n")
        file.write("Kappa: " + str(kappa) + "\n")
        file.write("F1: " + str(f1) + "\n")
        file.write("Confusion Matrix:\n")
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                file.write("{} ".format(str(cm[row,col])))
            file.write("\n")
        #file.write("Labels\n{}\n".format(np.asarray(labels)))
        #file.write("Predictions\n{}".format(np.asarray(preds)))
        
    
    #print ("\nAccuracy: " + str(acc))
    print ("Accuracy per Class:")
    for index, c in enumerate(b_acc[0]):
        print('Class {}: {}'.format(index, c))
    print("Balanced_Accuracy: " + str(b_acc[1]))
    print("Kappa: " + str(kappa))
    print("F1: " + str(f1))
    print(cm)

def f1_score_from_cm(confusion):

    # Sum columns and rows
    sum_col = np.sum(confusion, axis=0)
    sum_row = np.sum(confusion, axis=1)
    diag = np.diag(confusion)

    # Macro precision/recall
    precision = diag / (sum_col.astype(float) + 1e-15)
    recall = diag / (sum_row.astype(float) + 1e-15)
    
    # Compute Macro F1
    f1_score_per_class = 2.*(precision * recall) / ((precision + recall) + 1e-15)
    macro_f1 = np.mean(f1_score_per_class)
    
    return macro_f1

def cohen_kappa_from_cm(confusion, weights=None):
    r"""Cohen's kappa: a statistic that measures inter-annotator agreement.
    This function computes Cohen's kappa [1]_, a score that expresses the level
    of agreement between two annotators on a classification problem. It is
    defined as
    .. math::
        \kappa = (p_o - p_e) / (1 - p_e)
    where :math:`p_o` is the empirical probability of agreement on the label
    assigned to any sample (the observed agreement ratio), and :math:`p_e` is
    the expected agreement when both annotators assign labels randomly.
    :math:`p_e` is estimated using a per-annotator empirical prior over the
    class labels [2]_.
    Read more in the :ref:`User Guide <cohen_kappa>`.
    Parameters
    ----------
    confusion : confusion matrix
    weights : str, optional
        List of weighting type to calculate the score. None means no weighted;
        "linear" means linear weighted; "quadratic" means quadratic weighted.
    Returns
    -------
    kappa : float
        The kappa statistic, which is a number between -1 and 1. The maximum
        value means complete agreement; zero or lower means chance agreement.

    """
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    if weights is None:
        w_mat = np.ones([n_classes, n_classes], dtype=np.int)
        w_mat.flat[:: n_classes + 1] = 0
    elif weights == "linear" or weights == "quadratic":
        w_mat = np.zeros([n_classes, n_classes], dtype=np.int)
        w_mat += np.arange(n_classes)
        if weights == "linear":
            w_mat = np.abs(w_mat - w_mat.T)
        else:
            w_mat = (w_mat - w_mat.T) ** 2
    else:
        raise ValueError("Unknown kappa weighting type.")

    k = np.sum(w_mat * confusion) / float(np.sum(w_mat * expected))
    return 1 - k
    
def balanced_accuracy_score_from_cm(C, sample_weight=None,adjusted=False):
    """Compute the balanced accuracy
    The balanced accuracy in binary and multiclass classification problems to
    deal with imbalanced datasets. It is defined as the average of recall
    obtained on each class.
    The best value is 1 and the worst value is 0 when ``adjusted=False``.
    Read more in the :ref:`User Guide <balanced_accuracy_score>`.
    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) target values.
    y_pred : 1d array-like
        Estimated targets as returned by a classifier.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    adjusted : bool, default=False
        When true, the result is adjusted for chance, so that random
        performance would score 0, and perfect performance scores 1.
    Returns
    -------
    balanced_accuracy : float
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(C) / (C.sum(axis=1)).astype(float)
    if np.any(np.isnan(per_class)):
        #print('y_pred contains classes not in y_true')
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    if adjusted:
        n_classes = len(per_class)
        chance = 1. / n_classes
        score -= chance
        score /= 1. - chance
    return per_class, score
    
    
def main():
    parser = argparse.ArgumentParser(description='Create Raster Mask from Shapefile')
    parser.add_argument('--inRaster_Map', type=str, required=True,
                        help='Path to Raster Map')
    parser.add_argument('--inRaster_Reference', type=str, required=True,
                        help='Path to Raster Reference')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to folder where output metrics will be saved')
    
    args = parser.parse_args()
    inRaster_Map = args.inRaster_Map
    inRaster_Reference = args.inRaster_Reference
    outRaster = args.output_path
    print(args)

    # Create output path if not exists
    base_output = os.path.join(outRaster, '5_Compute_Metrics')
    if not os.path.exists(base_output):
         os.makedirs(base_output, exist_ok=True)
            
    maps_list = []
    mask_list = []

    maps_list.append(inRaster_Map)
    mask_list.append(inRaster_Reference)

    #output_name = 'myrecords'
    fp = open(os.path.join(base_output,'metrics_map'), 'w+')
    for tmp_map, tmp_mask in zip(maps_list,mask_list):

        data_map = rasterio.open(tmp_map).read()[:,:,:]
        data_mask = rasterio.open(tmp_mask).read()[:,:,:]
    
        '''
        Merging Classes
        '''

        '''
        # Prediction
        for c in [3,6,9,13]:
            try:
                data_map[data_map == c] = 0
                data_mask[data_mask == c] = 0
            except:
                print('Exception in class {}'.format(c))
                continue 
        '''
        
        #Remove 254 values
        bg_filter = data_mask != 254
    
        data_map = data_map[bg_filter]
        data_mask = data_mask[bg_filter]
    
        # Classes in predictions and labels need to be the same set
        cm = confusion_matrix(np.asarray(data_mask),np.asarray(data_map))
        calculate_metrics_cm(cm, fp)
        fp.close()

    
if __name__ == '__main__':
    main()
    
