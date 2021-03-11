import subprocess
import arcpy

class Train_Model_Tool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Training Deep Learning Model for Classification"
        self.description = "Given a raster and a shapefile containing examples some classes " + \
                           "create a classification model over that region."
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        params = None

        # First parameter
        param0 = arcpy.Parameter(
            displayName="Input Raster Dataset",
            name="in_rasterdataset",
            datatype=["DERasterDataset"],
            parameterType="Required",
            direction="Input")

        # Second parameter
        param1 = arcpy.Parameter(
            displayName="Input Raster Reference",
            name="in_shapefile",
            datatype=["DEShapefile"],
            parameterType="Required",
            direction="Input")

        # Third parameter
        param2 = arcpy.Parameter(
            displayName="Output Model",
            name="out_model",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input")

        param2.filter.list = ['.pt']

        param0.values = r"D:\Arcgis_Plugin\results\1_PREPARE_DATA\split"
        param1.values = r"D:\Arcgis_Plugin\results\out_raster\mask.tif"
        param2.values = r"D:\Arcgis_Plugin\results"

        params = [param0, param1, param2]

        return params


    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""

        dataset_path = parameters[0].valueAsText
        inRasterReference = parameters[1].valueAsText
        output_path = parameters[2].valueAsText

        print(dataset_path)
        print(inRasterReference)
        print(output_path)

        epochs = 200
        learning_rate = 0.01
        batch = 64
        optimizer_type = 'adam'
        early_stop = 20

        fine_tunning_imagenet = 'True'
        network_type = 'deeplabv3_50'
        only_top_layers = 'False'
        ignore_zero = 'True'

        #dataset_path = r'../1_PREPARE_DATA/results/1_PREPARE_DATA/split/'
        #inRasterReference = r'../0_CREATE_MASK/data_example/out_raster/mask.tif'
        #output_path = r'D:\Arcgis_Plugin\results'

        # Call cmd in conda env
        conda_python = r"C:\Users\edemir\anaconda3\envs\arc105\python.exe -W ignore"
        target_script = r"D:\Arcgis_Plugin\code\2_Segmentation_BASE\train.py " \
                        r"--epochs  {} " \
                        r"--learning_rate  {} "\
                        r"--batch  {} " \
                        r"--optimizer_type  {} " \
                        r"--early_stop  {} " \
                        r"--fine_tunning_imagenet  {} " \
                        r"--network_type  {} " \
                        r"--only_top_layers  {} " \
                        r"--ignore_zero  {} " \
                        r"--dataset_path  {} " \
                        r"--inRasterReference {} " \
                        r"--output_path {}".format(epochs, learning_rate, batch, optimizer_type, early_stop,
                                                   fine_tunning_imagenet, network_type, only_top_layers, ignore_zero,
                                                   dataset_path,inRasterReference,output_path)
        cmd = conda_python + " " + target_script

        messages.addMessage(cmd)

        my_subprocess = subprocess.Popen(cmd)

        return
