import os, time

mydir = os.path.split(os.path.abspath(__file__))[0]
homepath = os.path.expanduser(os.getenv('USERPROFILE'))

list_lib = [r'{}\anaconda3\envs\arc105'.format(homepath), \
     r'{}\anaconda3\envs\arc105\Library\mingw-w64\bin'.format(homepath), \
     r'{}\anaconda3\envs\arc105\Library\usr\bin'.format(homepath), \
     r'{}\anaconda3\envs\arc105\Library\bin'.format(homepath), \
     r'{}\anaconda3\envs\arc105\Scripts'.format(homepath), \
     r'{}\anaconda3\envs\arc105\bin'.format(homepath), \
     r'{}\anaconda3\condabin'.format(homepath)]

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
        
def generate_mask_map_biomas(inRaster_Reference, inShapefile_Reference, outRaster):

    # Need read rows, cols from original raster image
    with rasterio.open(inRaster_Reference, "r") as src:
        im_size = (src.meta['height'], src.meta['width'])
        raster_meta = src.meta
    
    # Read shapefile and extract records
    shp = shapefile.Reader(inShapefile_Reference)
    list_records = shp.shapeRecords()
    
    # Get total number of classes from shapefile
    set_classes = set()
    for feature in list_records:
        # Get record
        rec = feature.record.as_dict()
        set_classes.add(str(rec['id'])) 
        #print('Adding class {}'.format(str(rec['id'])))
    list_classes = sorted(list(set_classes))
    
    #Iterate over the number total of classes
    print(list_classes)

    poly_shp_dict = {}
    for label in list_classes:
    
        # Create a list of shapes for each class
        poly_shp_dict[str(label)] = []

    #print(poly_shp_dict)

    for index, feature in enumerate(list_records):

        if feature.shape.shapeType == 0:
            print('NULL shapefile founded.')
            continue
        # Get record
        rec = feature.record.as_dict()

        # Dict Keys for each polygon/feature
        #['bbox', 'parts', 'points', 'shapeType', 'shapeTypeName']
        geo_feat = feature.shape.__geo_interface__
        
        # Trying reajust transform in rasterize function
        if geo_feat['type'] == 'Polygon':
            #print(len(list(geo_feat['coordinates'])))    
            poly = Polygon(list(geo_feat['coordinates'][0]), geo_feat['coordinates'][1:])
            poly_shp_dict[str(rec['id'])].append(poly)
        elif geo_feat['type'] == 'MultiPolygon':
            for p in geo_feat['coordinates']:
                poly = Polygon(list(p[0]),p[1:])
                poly_shp_dict[str(rec['id'])].append(poly)

    # Create a Tuple of geometries and values
    all_shapes = []
    all_geo = []
    for label in list_classes:
        for i in poly_shp_dict[str(label)]:
            all_shapes.append( (i,int(label)) )
            all_geo.append(i)
     
    mask = rasterize(shapes=all_shapes, out_shape=im_size, all_touched=False, fill=254, transform=raster_meta['transform'])
    
    print(np.unique(mask))
    
    final_mask = mask.astype("float32")
    
    # CHANGE NODATA TO OTHER VALUE, THAN ZERO!!!!
    bin_mask_meta = src.meta.copy()
    bin_mask_meta.update({'count': 1})
    bin_mask_meta.update({'dtype':'float32'})
    bin_mask_meta.update({'nodata': -9999.0})
    
    print(bin_mask_meta)

    # Parse output name for reference mask
    output_name = os.path.splitext( os.path.basename(inShapefile_Reference) )[0]
    output_name = 'reference_{}.tif'.format(output_name)
    
    with rasterio.open(os.path.join(outRaster,output_name), 'w', **bin_mask_meta) as dst:
        dst.write(final_mask, 1)
    
def main():
    parser = argparse.ArgumentParser(description='Create Raster Mask from Shapefile')
    parser.add_argument('--inRaster_Reference', type=str, required=True,
                        help='Path to Raster Reference')
    parser.add_argument('--inShapefile_Reference', type=str, required=True,
                        help='Path to Shapefile Reference')
    parser.add_argument('--outRaster', type=str, required=True,
                        help='Path to folder where output Raster will be saved')
    
    args = parser.parse_args()
    inRaster_Reference = args.inRaster_Reference
    inShapefile_Reference = args.inShapefile_Reference
    outRaster = args.outRaster
    print(args)

    # Create output path if not exists
    base_output = os.path.join(os.getcwd(), outRaster, '0_CREATE_MASK')
    if not os.path.exists(base_output):
         os.makedirs(base_output, exist_ok=True)
            
    # Run process
    generate_mask_map_biomas(inRaster_Reference, inShapefile_Reference, base_output)
    
if __name__ == '__main__':
    main()
    
