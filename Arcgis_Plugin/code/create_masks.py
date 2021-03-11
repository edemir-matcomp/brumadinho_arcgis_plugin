import argparse
import numpy as np
import shapefile

import rasterio
from rasterio.features import rasterize

from shapely.geometry import Polygon
from shapely.ops import cascaded_union

import time



def poly_from_utm(polygon, transform):
    poly_pts = []

    poly = cascaded_union(polygon)
    for i in np.array(poly.exterior.coords):

        poly_pts.append(~transform * tuple(i))

    new_poly = Polygon(poly_pts)
    return new_poly
        
def generate_mask_map_biomas(inRaster_Reference, inShapefile_Reference, outRaster):

    # Need read rows, cols from original raster image
    with rasterio.open(inRaster_Reference, "r") as src:
        im_size = (src.meta['height'], src.meta['width'])
        raster_meta = src.meta
    
    # Read shapefile and extract records
    shp = shapefile.Reader(inShapefile_Reference)
    list_records = shp.shapeRecords()
    
    # Get total number of classes from shapefile
    # TODO: this naive method need to loop over all shapes
    set_classes = set()
    for feature in list_records:
        # Get record
        rec = feature.record.as_dict()
        set_classes.add(str(rec['id'])) 
        #print('Adding class {}'.format(str(rec['id'])))
    list_classes = sorted(list(set_classes))
    
    #Iterate over the number total of classes
    #list_classes = [1,2]
    
    poly_shp_dict = {}
    for label in list_classes:
    
        # Create a list of shapes for each class
        poly_shp_dict[str(label)] = []

    #print(poly_shp_dict)

    for feature in list_records:

        # Get record
        rec = feature.record.as_dict()
        
        # Dict Keys for each polygon/feature
        #['bbox', 'parts', 'points', 'shapeType', 'shapeTypeName']
        geo_feat = feature.shape.__geo_interface__
        
        #print(geo_feat['type'])
        
        if geo_feat['type'] == 'Polygon':
            poly = poly_from_utm(Polygon(list(geo_feat['coordinates'][0])), raster_meta['transform'])
            poly_shp_dict[str(rec['id'])].append(poly)
        elif geo_feat['type'] == 'MultiPolygon':
            for p in geo_feat['coordinates']:
                poly = poly_from_utm(Polygon(list(p[0])), raster_meta['transform'])
                poly_shp_dict[str(rec['id'])].append(poly)

    # Create a raster mask for each class
    #final_mask = np.zeros(im_size)
    
    final_mask = np.full(im_size,254)
    #final_mask_bg = np.full(im_size,254)
    
    for label in list_classes:
        print('Filling class {}'.format(int(label)))
        # TODO MASK COM OVERLAP
        mask = rasterize(shapes=poly_shp_dict[str(label)], out_shape=im_size)
        #final_mask += mask*int(label)
        
        valid_slide = np.nonzero(mask)
        final_mask[valid_slide] = int(label)
        #final_mask_bg[valid_slide] = int(0)
    
        # Save Individual Masks
        '''
        #mask = mask.astype("uint16")
        mask = mask.astype("float32")
        save_path = 'mask_label_{}.tif'.format(label)
        bin_mask_meta = src.meta.copy()
        bin_mask_meta.update({'count': 1})
        with rasterio.open(save_path, 'w', **bin_mask_meta) as dst:
            dst.write(mask * label, 1)
        '''
            
    #Save Final Mask
    final_mask = final_mask.astype("float32")
    bin_mask_meta = src.meta.copy()
    bin_mask_meta.update({'count': 1})
    with rasterio.open(outRaster+'/mask.tif', 'w', **bin_mask_meta) as dst:
        dst.write(final_mask, 1)
        
    #final_mask_bg = final_mask_bg.astype("int64")
    #with rasterio.open(outRaster+'/mask_bg.tif', 'w', **bin_mask_meta) as dst_bg:
    #    dst_bg.write(final_mask_bg, 1)
    
            
      
def main():
    parser = argparse.ArgumentParser(description='Create Raster Mask from Shapefile')
    parser.add_argument('--inRaster_Reference', type=str, required=True,
                        help='Path to Raster Reference')
    parser.add_argument('--inShapefile_Reference', type=str, required=True,
                        help='Path to Shapefile Reference')
    parser.add_argument('--outRaster', type=str, required=True,
                        help='Path to folder where output Raster will be saved')

    time.sleep(10)
    
    args = parser.parse_args()
    inRaster_Reference = args.inRaster_Reference
    inShapefile_Reference = args.inShapefile_Reference
    outRaster = args.outRaster

    
    
    
    # Run process
    generate_mask_map_biomas(inRaster_Reference, inShapefile_Reference, outRaster)
    
if __name__ == '__main__':
    time.sleep(10)
    main()
    
