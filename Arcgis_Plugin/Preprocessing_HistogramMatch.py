import os,subprocess
import arcpy

class Preprocessing_HistogramMatch_Tool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "(3) Teste: Computa o Histogram Matching entre a nova imagem e uma imagem de referencia"
        self.description = "Given a reference raster and new raster " + \
                           "compute new matching raster image."
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        params = None

        # First parameter
        param0 = arcpy.Parameter(
            displayName="Input Raster Map",
            name="in_raster_dataset",
            datatype=["DERasterDataset"],
            parameterType="Required",
            direction="Input")

        # Second parameter
        param1 = arcpy.Parameter(
            displayName="Input Raster Original",
            name="in_raster_reference",
            datatype=["DERasterDataset"],
            parameterType="Required",
            direction="Input")

        # Output parameter
        param2 = arcpy.Parameter(
            displayName="Output Map Path",
            name="out_model",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input")

        param0.values = r"D:\Arcgis_Plugin_Saida\data\brumadinho\Geoeye_Raster\Pleiades_T1.tif"
        param1.values = r"D:\Arcgis_Plugin_Saida\data\brumadinho\Geoeye_Raster\Geoeye_T3.tif"
        param2.values = r"D:\Arcgis_Plugin_Saida\results"

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

        inRaster_Map = parameters[0].valueAsText
        inRaster_Reference = parameters[1].valueAsText
        output_path = parameters[2].valueAsText

        messages.addMessage(inRaster_Map)
        messages.addMessage(inRaster_Reference)
        messages.addMessage(output_path)

        mydir = os.path.split(os.path.abspath(__file__))[0]
        homepath = os.path.expanduser(os.getenv('USERPROFILE'))

        # Call cmd in conda env
        conda_python = r"{}\anaconda3\envs\arc105\python.exe -W ignore".format(homepath)
        target_script = r"{}\code\4_Histogram_Matching\histogram_matching.py " \
                        r'--inRaster_Map="{}" ' \
                        r'--inRaster_Reference="{}" ' \
                        r'--output_path="{}"'.format(mydir, inRaster_Map, inRaster_Reference, output_path)

        cmd = conda_python + " " + target_script

        messages.addMessage(cmd)

        my_subprocess = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = my_subprocess.communicate()
        messages.addMessage(output)
        messages.addMessage(error)

        return
