import os, sys
import arcpy
import imp

import Create_Mask
import Create_Tiles
import Train_Model
import Evaluate_Model
import Compute_Metrics
import Preprocessing_HistogramMatch

imp.reload(Create_Mask)
imp.reload(Create_Tiles)
imp.reload(Train_Model)
imp.reload(Evaluate_Model)
imp.reload(Compute_Metrics)
imp.reload(Preprocessing_HistogramMatch)

from Create_Mask import Create_Mask_Tool as Tool1
from Create_Tiles import Create_Tiles_Tool as Tool2
from Train_Model import Train_Model_Tool as Tool3
from Evaluate_Model import Evaluate_Model_Tool as Tool4
from Compute_Metrics import Compute_Metrics_Tool as Tool5
from Preprocessing_HistogramMatch import Preprocessing_HistogramMatch_Tool as Tool6

class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Compute Classification Map"
        self.alias = "training"

        # List of tool classes associated with this toolbox
        self.tools = [Tool1, Tool2, Tool3, Tool4, Tool5, Tool6]






