import os,subprocess
import arcpy

class Evaluate_Model_Tool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "(4) Teste: Usar modelo treinado em uma nova imagem."
        self.description = "Given a raster and his background " + \
                           "create a prediction mask over the region."
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        params = None

        # First parameter
        param0 = arcpy.Parameter(
            displayName="Input Raster Dataset",
            name="in_raster_dataset",
            datatype=["DERasterDataset"],
            parameterType="Required",
            direction="Input")

        # First parameter
        param1 = arcpy.Parameter(
            displayName="Input Raster_MDT Dataset",
            name="in_raster_mdt_dataset",
            datatype=["DERasterDataset"],
            parameterType="Optional",
            direction="Input")

        # First parameter
        param2 = arcpy.Parameter(
            displayName="Input Raster_DEC Dataset",
            name="in_raster_dec_dataset",
            datatype=["DERasterDataset"],
            parameterType="Optional",
            direction="Input")

        # Model parameter
        param4 = arcpy.Parameter(
            displayName="Input Model Trained",
            name="in_model",
            datatype=["DEFile"],
            parameterType="Required",
            direction="Input")

        # Output parameter
        param5 = arcpy.Parameter(
            displayName="Output Map Path",
            name="out_model",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input")

        param0.values = r"D:\Arcgis_Plugin_Saida\data\brumadinho\Geoeye_Raster\Geoeye_T3.tif"
        param1.values = r"D:\Arcgis_Plugin_Saida\data\brumadinho\MDT\MDT_T3_Resampled.tif"
        param2.values = r"D:\Arcgis_Plugin_Saida\data\brumadinho\Declividade\Declividade_T3_Resampled.tif"
        param4.values = r"D:\Arcgis_Plugin_Saida\results\2_Segmentation_BASE\fcn_50_final_model_ft_fold_0"
        param5.values = r"D:\Arcgis_Plugin_Saida\results"

        params = [param0, param1, param2, param4, param5]

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

        inRaster_Image = parameters[0].valueAsText
        inRaster_MDT = parameters[1].valueAsText
        inRaster_DEC = parameters[2].valueAsText
        model_path = parameters[3].valueAsText
        output_path = parameters[4].valueAsText
        batch = 64

        mydir = os.path.split(os.path.abspath(__file__))[0]
        homepath = os.path.expanduser(os.getenv('USERPROFILE'))

        # Call cmd in conda env
        conda_python = r"{}\anaconda3\envs\arc105\python.exe -W ignore".format(homepath)
        target_script = r"{}\code\2_Segmentation_BASE_2\generate_map.py " \
                        r'--model_path="{}" ' \
                        r"--batch  {} " \
                        r'--inRaster_Image="{}" ' \
                        r'--inRaster_MDT="{}" ' \
                        r'--inRaster_DEC="{}" ' \
                        r'--output_path="{}"'.format(mydir, model_path, batch, inRaster_Image, inRaster_MDT,
                                                   inRaster_DEC, output_path)

        cmd = conda_python + " " + target_script

        messages.addMessage(cmd)

        #my_subprocess = subprocess.Popen(cmd)
        my_subprocess = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = my_subprocess.communicate()
        messages.addMessage(output)
        messages.addMessage(error)

        return
