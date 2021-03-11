inRaster_Reference='../data/brumadinho/Geoeye_Raster/Geoeye_T3.tif'
inShapefile_Reference='../data/brumadinho/Shapefile/shapefile_brumadinho.shp'
outRaster='data_example/out_raster'

rm data_example/out_raster/mask.tif

#dataset_path='/home/edemir/FAPEMIG_BRUMADINHO/data/montesanto/dataset_montesanto'
#output_path='/home/edemir/FAPEMIG_BRUMADINHO/data/montesanto/results/'${dataset_path##*/}

# Create a directory for logging
mkdir -p $output_path'/'$network_type\_$learning_rate\_$optimizer

script -c "python3 -W ignore create_masks.py \
    --inRaster_Reference $inRaster_Reference \
    --inShapefile_Reference $inShapefile_Reference \
    --outRaster $outRaster'/'
    " results/logs_runs

