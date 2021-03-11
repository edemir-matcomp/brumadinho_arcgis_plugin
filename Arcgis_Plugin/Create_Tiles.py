import os,subprocess
import arcpy

class Create_Tiles_Tool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "(1) Treino: Cria patches de imagens sem sobreposicao"
        self.description = "Given a raster and a shapefile containing examples some classes " + \
                           "create a raster mask over that region."
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        params = None

        # Raster Image
        param0 = arcpy.Parameter(
            displayName="Input Raster Dataset",
            name="in_raster_image",
            datatype=["DERasterDataset"],
            parameterType="Required",
            direction="Input")

        # Raster MDT
        param1 = arcpy.Parameter(
            displayName="Input Raster MDT",
            name="in_raster_mdt",
            datatype=["DERasterDataset"],
            parameterType="Optional",
            direction="Input")

        # Raster DEC
        param2 = arcpy.Parameter(
            displayName="Input Raster DEC",
            name="in_raster_dec",
            datatype=["DERasterDataset"],
            parameterType="Optional",
            direction="Input")

        # Raster Reference
        param3 = arcpy.Parameter(
            displayName="Input Raster Reference",
            name="in_raster_reference",
            datatype=["DERasterDataset"],
            parameterType="Required",
            direction="Input")

        # Third parameter
        param4 = arcpy.Parameter(
            displayName="Output Model",
            name="out_model",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input")

        #param2.filter.list = ['.pt']


        param0.values = r"D:\Arcgis_Plugin\data\brumadinho\Geoeye_Raster\Geoeye_T3.tif"
        param1.values = r"D:\Arcgis_Plugin\data\brumadinho\MDT\MDT_T3_Resampled.tif"
        param2.values = r"D:\Arcgis_Plugin\data\brumadinho\Declividade\Declividade_T3_Resampled.tif"
        param3.values = r"D:\Arcgis_Plugin\results\out_raster\mask.tif"
        param4.values = r"D:\Arcgis_Plugin\results"

        params = [param0, param1, param2, param3, param4]

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
        inRaster_MDT = parameters[1].valueAsText
        inRaster_DEC = parameters[2].valueAsText
        inRaster_Reference = parameters[3].valueAsText
        outModel = parameters[4].valueAsText

        messages.addMessage(inRaster)
        messages.addMessage(inRaster_MDT)
        messages.addMessage(inRaster_DEC)
        messages.addMessage(inRaster_Reference)
        messages.addMessage(outModel)

        mydir = os.path.split(os.path.abspath(__file__))[0]
        homepath = os.path.expanduser(os.getenv('USERPROFILE'))

        #Call cmd in conda env
        conda_python = r"{}\anaconda3\envs\arc105\python.exe -W ignore".format(homepath)
        target_script = r'{}\code\1_PREPARE_DATA\rasterio_tiles.py --inRaster_Image="{}" --inRaster_MDT="{}" --inRaster_DEC="{}" --inRaster_Reference="{}" --outRaster="{}"'.format(mydir, inRaster, inRaster_MDT, inRaster_DEC, inRaster_Reference, outModel)
        cmd = conda_python + " " + target_script

        messages.addMessage(cmd)

        my_subprocess = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = my_subprocess.communicate()
        messages.addMessage(output)
        messages.addMessage(error)

        return
