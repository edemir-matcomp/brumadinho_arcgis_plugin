import os, time

mydir = os.path.split(os.path.abspath(__file__))[0]
homepath = os.path.expanduser(os.getenv('USERPROFILE'))
#print('Meu diretorio: {}'.format(mydir))
#print('Minha Home: {}'.format(homepath))

list_lib = [r'{}\anaconda3\envs\arc105'.format(homepath), \
     r'{}\anaconda3\envs\arc105\Library\mingw-w64\bin'.format(homepath), \
     r'{}\anaconda3\envs\arc105\Library\usr\bin'.format(homepath), \
     r'{}\anaconda3\envs\arc105\Library\bin'.format(homepath), \
     r'{}\anaconda3\envs\arc105\Scripts'.format(homepath), \
     r'{}\anaconda3\envs\arc105\bin'.format(homepath), \
     r'{}\anaconda3\condabin'.format(homepath)]

for i in list_lib:
    os.environ['PATH'] = '%s;%s' % (i, os.environ['PATH'])

import rasterio
import numpy as np
import argparse

def _match_cumulative_cdf(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """

    src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_unique_indices].reshape(source.shape)
    
def match_histograms(image, reference, *, multichannel=False):
    """Adjust an image so that its cumulative histogram matches that of another.
    The adjustment is applied separately for each channel.
    Parameters
    ----------
    image : ndarray
        Input image. Can be gray-scale or in color.
    reference : ndarray
        Image to match histogram of. Must have the same number of channels as
        image.
    multichannel : bool, optional
        Apply the matching separately for each channel.
    Returns
    -------
    matched : ndarray
        Transformed input image.
    Raises
    ------
    ValueError
        Thrown when the number of channels in the input image and the reference
        differ.
    References
    ----------
    .. [1] http://paulbourke.net/miscellaneous/equalisation/
    """
    if image.ndim != reference.ndim:
        raise ValueError('Image and reference must have the same number '
                         'of channels.')

    if multichannel:
        if image.shape[-1] != reference.shape[-1]:
            raise ValueError('Number of channels in the input image and '
                             'reference image must match!')

        matched = np.empty(image.shape, dtype=image.dtype)
        for channel in range(image.shape[-1]):
        
            # Filter only valid pixels
            tmp_image = image[:,:,channel]
            tmp_reference = reference[:,:,channel]
            
            
            matched_channel = _match_cumulative_cdf(tmp_image[tmp_image != -9999.0],
                                                 tmp_reference[tmp_reference != -9999.0])
            
            tmp_image[tmp_image != -9999.0] = matched_channel
            matched[..., channel] = tmp_image
               
    else:
        matched = _match_cumulative_cdf(image, reference)

    return matched

def main():

    parser = argparse.ArgumentParser(description='Create Raster Mask from Shapefile')
    parser.add_argument('--inRaster_Map', type=str, required=True,
                        help='Path to Raster Map')
    parser.add_argument('--inRaster_Reference', type=str, required=True,
                        help='Path to Raster Reference')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to folder where output metrics will be saved')


    args = parser.parse_args()
    inRaster_Map = args.inRaster_Map
    inRaster_Reference = args.inRaster_Reference
    outRaster = args.output_path
    print(args)

    # Create output path if not exists
    base_output = os.path.join(outRaster, '3_Histogram_Matching')
    if not os.path.exists(base_output):
         os.makedirs(base_output, exist_ok=True)

    reference = rasterio.open(inRaster_Reference).read()[:,:,:].transpose(1,2,0).copy()

    with rasterio.open(inRaster_Map) as src:
            
            print(src.dtypes)
            print(src.nodatavals)
        
            # Get Info from original raster
            image = src.read()[:,:,:].transpose(1,2,0).copy()
            
            print('Image to Matching: ',image.shape, image.dtype)
            print('Reference Image: ',reference.shape, reference.dtype)

            # Histogram Matching
            matched = match_histograms(image, reference, multichannel=True).copy()
            print('Matched Image: ',matched.shape)
            matched = matched.transpose(2,0,1).copy()
            print('Matched Image Transpose: ',matched.shape)
            print(matched.dtype)
            
            # Save new image georeferenced tif
            bin_mask_meta = src.meta.copy()
            output_name = os.path.splitext(os.path.basename(inRaster_Map))[0]
            with rasterio.open(os.path.join(outRaster,'3_Histogram_Matching',output_name+'_HistogramMatch.tif'), 'w', **bin_mask_meta) as dst:
                dst.write(matched)

            print("Histogram Match Image in  {}".format('Pleiades_T1_HistogramMatch.tif'))

if __name__ == '__main__':
    main()
