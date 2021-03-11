#import os, sys
#from subprocess import Popen, PIPE, check_output
import os,subprocess
import arcpy

class Create_Mask_Tool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "(0) Treino: Cria uma imagem raster a partir do shapefile de referencia"
        self.description = "Given a raster and a shapefile containing examples some classes " + \
                           "create a raster mask over that region."
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
            displayName="Input Shapefile",
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

        #param2.filter.list = ['.pt']

        param0.values = r"D:\Arcgis_Plugin_Saida\data\brumadinho\Geoeye_Raster\Geoeye_T3.tif"
        param1.values = r"D:\Arcgis_Plugin_Saida\data\brumadinho\Shapefile\merged_T3\merged_T3.shp"
        param2.values = r"D:\Arcgis_Plugin_Saida\results\New Folder"

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

        inRaster = parameters[0].valueAsText
        inShapefile = parameters[1].valueAsText
        outModel = parameters[2].valueAsText

        messages.addMessage(inRaster)
        messages.addMessage(inShapefile)
        messages.addMessage(outModel)
        
        mydir = os.path.split(os.path.abspath(__file__))[0]
        homepath = os.path.expanduser(os.getenv('USERPROFILE'))

        #Call cmd in conda env
        conda_python = r"{}\anaconda3\envs\arc105\python.exe -W ignore".format(homepath)
        target_script = r'{}\code\0_CREATE_MASK\create_masks.py --inRaster_Reference="{}" --inShapefile_Reference="{}" --outRaster="{}"'.format(mydir, inRaster, inShapefile, outModel)
        cmd = conda_python + " " + target_script
        #my_subprocess = subprocess.Popen(cmd, stderr=subprocess.PIPE)
        my_subprocess = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Redirect error output to arcgis
        output, error = my_subprocess.communicate()
        messages.addMessage(output)
        messages.addMessage(error)
  
        return
