inRaster_Image='../data/brumadinho/Geoeye_Raster/Geoeye_T3.tif'
inRaster_Reference='../0_CREATE_MASK/data_example/out_raster/mask.tif'
patchSize=256
outRaster='results'

mkdir -p split/train/imgs/
mkdir -p split/train/masks/
mkdir -p split/train/dec/
mkdir -p split/train/mdt/
mkdir -p split/val/imgs/
mkdir -p split/val/masks/
mkdir -p split/val/dec/
mkdir -p split/val/mdt/

rm results/imgs/*
rm results/masks/*
rm results/dec/*
rm results/mdt/*

rm split/train/imgs/*
rm split/train/masks/*
rm split/train/dec/*
rm split/train/mdt/*
rm split/val/imgs/*
rm split/val/masks/*
rm split/val/dec/*
rm split/val/mdt/*

script -c "python3 -W ignore rasterio_tiles.py \
    --inRaster_Image $inRaster_Image \
    --inRaster_Reference $inRaster_Reference \
    --patchSize $patchSize \
    --outRaster $outRaster'/'
    " results/logs_runs
