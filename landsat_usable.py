import geopandas as gpd
import fiona
import gdal
from osgeo import ogr,osr
import rasterio as rio
from rasterio.mask import mask
import rasterio.features
import rasterio.warp
from shapely.geometry import box

import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

import yaml
import json

from pathlib import Path
import os
from os import walk

#Setting path structure
CWD = Path.cwd()
#print(CWD)
DATA = Path(CWD/'data')
#print (DATA)
INTER  = DATA/'inter'
INPUT  = DATA/'input'
OUTPUT = DATA/'output'

# Import User Config file to set globals
config_file = DATA/'config.yaml'
user_config = yaml.load(open(config_file), Loader=yaml.FullLoader)

# Globals
brightness_bands = user_config['UHI']['bands'].get('brightness', [])
reflectance_bands = user_config['UHI']['bands'].get('reflectance', [])
radiance_bands=user_config['UHI']['bands'].get('radiance', [])
allbands = brightness_bands+reflectance_bands

# REGION = user_config['Global'].get('REGION', [0])
# UTM =  user_config['Global'].get('UTM', [0])
# self.__WGS84= user_config['Global'].get('self.__WGS84', [0])
#crs = UTM

class calc_landsat_indices(object):
    def __init__(self, REGION, UTM, WGS84, extent):
        self.__REGION = REGION
        self.__UTM = UTM
        self.__WGS84 = WGS84
        self.__extent = extent
        #print("Done Init!")
        
    def __tuple_to_geojson(self, extent:tuple):
        
        '''This function saves extent input from the user to the geojson og two different
        projections 1) WGS84 and 2) UTM'''
        bbox= tuple(self.__extent)
        b = box(self.__extent[0],self.__extent[1],self.__extent[2],self.__extent[3])
        df=pd.DataFrame()
        df['geometry']=[b]
        shape_of_box=gpd.GeoDataFrame(df, geometry=df.geometry, crs=self.__WGS84)
        ##print (shape_of_box)
        shape_of_box.to_file(f'{INTER/self.__REGION}_box.geojson', driver='GeoJSON')
        box_to_utm=shape_of_box.to_crs(epsg=self.__UTM)
        box_to_utm.to_file(f'{INTER/self.__REGION}_box_{self.__UTM}.geojson', driver='GeoJSON')
        box_to_utm.to_file(f'{INTER/self.__REGION}_box_{self.__UTM}.shp')
        region_box= gpd.read_file(f'{INTER/self.__REGION}_box_{self.__UTM}.geojson')
        return region_box , bbox
    
    def tuple_to_geojson(self, extent:tuple):
        
        '''This function saves extent input from the user to the geojson og two different
        projections 1) WGS84 and 2) UTM
        '''
        
        bbox= tuple(self.__extent)
        b = box(self.__extent[0],self.__extent[1],self.__extent[2],self.__extent[3])
        df=pd.DataFrame()
        df['geometry']=[b]
        shape_of_box=gpd.GeoDataFrame(df, geometry=df.geometry, crs=self.__WGS84)
        ##print (shape_of_box)
        shape_of_box.to_file(f'{INTER/self.__REGION}_box.geojson', driver='GeoJSON')
        box_to_utm=shape_of_box.to_crs(epsg=self.__UTM)
        box_to_utm.to_file(f'{INTER/self.__REGION}_box_{self.__UTM}.geojson', driver='GeoJSON')
        box_to_utm.to_file(f'{INTER/self.__REGION}_box_{self.__UTM}.shp')
        region_box= gpd.read_file(f'{INTER/self.__REGION}_box_{self.__UTM}.geojson')
        return region_box , bbox
    
    def __list_files(self,dir):
        
        '''This function checks files present in the input dictionary and returns a list 
        '''
        raw = []
        for root, dirs, files in os.walk(dir):
            for name in files:
                raw.append(os.path.join(root, name))
        return raw

#     def __clip_tile(self, name, bands=[2,3,4,5,6,7,10,11]):
#         for band in bands:
#             tile = f'{INPUT}/{name}{band}.TIF'
#             !gdalwarp -cutline {INTER/REGION}_box_{UTM}.shp -crop_to_cutline -dstalpha {INPUT}/LC08_L1TP_147040_20190521_20190604_01_T1_B10.TIF {INTER}/cropped_{band}.tif
#         return None
    
    def __create_info_lists(self, raw:list):
        
        '''This function loops through the raw list of elements created and 
        retuns following information:
        year_month1 : first year month information present in the landsat 8 file name
        year_month2_list: second year month information present in the landsat 8 file name
        product_list: Name of the landsat product including WRS row and path
        metayear_month_list: Metadata for the specific product
        '''
        for idx, path in enumerate(raw):
            ##print(path)
            if path.split('\\')[-1].split('_')[7].split('.')[1] =='TIF':
                path1=path.split('\\')[-1]
                a=str(path1)
                ##print(a)
                path_list=[]
                path_list.append(a)
                year_month1=a.split('_')[3]#.split('.')[0].split('_')[3]
                year_month1_list=[]
                year_month1_list.append(year_month1)
                ##print(year_month1_list)
                year_month2=a.split('_')[4]#.split('.')[0].split('_')[3]
                year_month2_list=[]
                year_month2_list.append(year_month2) 
                product =a.split('_')[2]#.split('.')[0].split('_')[3]
                product_list=[]
                product_list.append(product) 
                ##print(product_list)

            elif path.split('\\')[-1].split('_')[7].split('.')[1] =='txt':
                path2=path.split('\\')[-1]
                ##print(path2)
                b=str(path2)
                metadata_list=b
                metayear_month_list=[]
                metayear_month_list.append(metadata_list)
                ##print(metayear_month_list)

        return year_month1_list, year_month2_list, product_list, metayear_month_list

    def __iterative_clipping(self, year_month1_list:list, year_month2_list:list, product_list:list):
        
        '''
        This function helps to clip each and ever landsat band required
        and saves it into the intermediate folder for the future use.
        '''
        date_list = [] 
        [date_list.append(x) for x in year_month1_list if x not in date_list]
        date_list2 = [] 
        [date_list2.append(x) for x in year_month2_list if x not in date_list2]
        prod_list = [] 
        [prod_list.append(x) for x in product_list if x not in prod_list]

        #Path(f'{INTER}/{REGION}/{year_month1}/').mkdir(exist_ok=True)
        bands=[2,3,4,5,6,7,10,11]
        for band in bands:
            for i, j in enumerate (date_list):
                for k,l in enumerate(date_list2):
                    for m,n in enumerate (prod_list):
                
                        if not os.path.exists(f'{INTER}/{self.__REGION}/{j}'):
                            os.makedirs(f'{INTER}/{self.__REGION}/{j}')

                        with fiona.open(f'{INTER}/{self.__REGION}_box_{self.__UTM}.geojson', "r") as shapefile:
                            shapes = [feature["geometry"] for feature in shapefile]

                        with rasterio.open(f"{INPUT}/LC08_L1TP_{n}_{j}_{l}_01_T1_B{band}.TIF") as src:
                            out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
                            out_meta = src.meta

                        out_meta.update({"driver": "GTiff",
                                         "height": out_image.shape[1],
                                         "width": out_image.shape[2],
                                         "transform": out_transform})

                        with rasterio.open(f'{INTER}/cropped_{band}.tif', "w", **out_meta) as dest:
                            dest.write(out_image)
        return None
    
    def __get_metaData_dict(self, text_file_path):
        text_file = open(text_file_path, "r").read() #change

        meta_data_dict={}
        for num, line in enumerate(text_file.strip().split("\n")):
            if num==(len(text_file.strip().split("\n"))-1):
                continue
            split_line =  line.strip().split(' = ')
            meta_data_dict[split_line[0]] = split_line[1]
        return meta_data_dict

##################################################
    def read_data_asarray(self, input_tiff_path):
        gtif = gdal.Open(input_tiff_path)
        band = gtif.GetRasterBand(1)
        bandArray = band.ReadAsArray()
        return band, bandArray



    def __extract_brightness(self, band:int, tiff_data, metadata:dict):
        res = {}
#         radiometric_meta = metadata['RADIOMETRIC_RESCALING']
#         thermal_constans_meta = metadata['TIRS_THERMAL_CONSTANTS']

        TOA_spectral_radiance = (float(
            metadata[f'RADIANCE_MULT_BAND_{band}']
            ) * tiff_data[1]) + float(metadata[f'RADIANCE_ADD_BAND_{band}'])
        
#        top_atmos_brightness = float(metadata[f'K2_CONSTANT_BAND_{band}']) / (np.log(float(metadata[f'K1_CONSTANT_BAND_{band}']) / TOA_spectral_radiance) + 1)

        ta_brightness1=float(metadata[f'K2_CONSTANT_BAND_{band}'])
        ta_brightness2= np.log(float(metadata[f'K1_CONSTANT_BAND_{band}']) / TOA_spectral_radiance) 
        top_atmos_brightness=(ta_brightness1/ta_brightness2+1)
        
        res[band] = top_atmos_brightness
        return res


    def __extract_reflectance(self, band:int, tiff_data, metadata:dict):
        res = {}
#         radiometric_meta = metadata['L1_METADATA_FILE']['RADIOMETRIC_RESCALING']
        sun_elevation_meta = metadata['SUN_ELEVATION']

        TOA_planetary_reflectance_no = (float(
            metadata[f'REFLECTANCE_MULT_BAND_{band}']
        ) * tiff_data[1]) + float(metadata[f'REFLECTANCE_ADD_BAND_{band}'])

        theta_se = math.pi / 180 * float(sun_elevation_meta)
        # theta_sz = 90 - theta_se
    
        TOA_planetary_reflectance = TOA_planetary_reflectance_no / (math.sin(theta_se))
        # TOA_planetary_reflectance = TOA_planetary_reflectance_no / (np.cos(theta_sz))

        res[band] = TOA_planetary_reflectance
        return res

    brightness = {}
#         for band in brightness_bands:
#             tdata = read_data_asarray(f'{DATA}/intermediate/{j}/calgary_band_{band}.tif')
#             brightness.update(extract_brightness(band, tdata))
    reflectance = {}
#         for band in reflectance_bands:
#             tdata = read_data_asarray(f'{DATA}/intermediate/{j}/calgary_band_{band}.tif')
#             reflectance.update(extract_reflectance(band, tdata))


        ###

    def __calculate_brightness(self, metadata:dict, brightness:dict):
        #brightness = {}
        for band in brightness_bands:
            tdata = self.read_data_asarray(f"{INTER}/cropped_{band}.tif")
            brightness.update(self.__extract_brightness(band, tdata, metadata))
            #print (brightness)

    def __calculate_reflectance(self, metadata:dict, reflectance:dict):
        #reflectance = {}
        for band in reflectance_bands:
            tdata = self.read_data_asarray(f"{INTER}/cropped_{band}.tif")
            reflectance.update(self.__extract_reflectance(band, tdata, metadata))
        

    # Calculate LST
    def __calculate_NDVI(self, reflectance):
        check_4 = np.logical_and(reflectance[5] > 0, reflectance[4] > 0 ) ### Checking for nan/filler values in mir and nir
        NDVI = np.where(check_4, ((reflectance[5] - reflectance[4] ) / ( reflectance[5] + reflectance[4] )), -1)
        return NDVI
    
    def __calculate_NDMI(self, reflectance):
        check_4 = np.logical_and(reflectance[5] > 0, reflectance[6] > 0 ) ### Checking for nan/filler values in mir and nir
        NDMI = np.where(check_4, ((reflectance[5] - reflectance[6] ) / ( reflectance[5] + reflectance[6] )), -1)
        return NDMI


    def __calculate_NDBI(self, reflectance):
        check_6 = np.logical_and(reflectance[5] > 0, reflectance[6] > 0)
        NDBI = np.where(check_6, ((reflectance[5] - reflectance[6] ) / ( reflectance[5] + reflectance[6] )), -1)
        return NDBI


    def __calculate_vegetation(self, reflectance):
        ndvi_array = self.__calculate_NDVI(reflectance)
        #NDBI = calculate_NDBI(reflectance)
        vegetation_proportion = ((ndvi_array - ndvi_array.min())/(ndvi_array.max()-ndvi_array.min())**2)
        return vegetation_proportion

    def __calculate_MNDWI(self, reflectance):
        check_6 = np.logical_and(reflectance[3] > 0, reflectance[6] > 0)
        MNDWI = np.where(check_6, ((reflectance[3] - reflectance[6] ) / ( reflectance[3] + reflectance[6] )), -1)
        return MNDWI

    def __calculate_ALBEDO(self, reflectance):
        check_6 = np.logical_and(reflectance[3] > 0, reflectance[6] > 0)
        ALBEDO = np.where(check_6, (((0.356 * reflectance[2]) + (0.15 * reflectance[4]) + (0.373 * reflectance[5])
                                     + (0.085 * reflectance[6]) + (0.072 * reflectance[7]) - 0.0018)/ 1.016) , -1)
        return ALBEDO

    def __calculate_emissivity(self, reflectance):
        vegetation_proportion = self.__calculate_vegetation(reflectance)
        emissivity = 0.004 * vegetation_proportion + 0.986
        return emissivity


    def __calculate_LST(self, brightness, reflectance):
        ndvi_array=self.__calculate_NDVI(reflectance)
        check_7 = np.logical_and(brightness[10] > 0, brightness[11] > 0)
        
        pv= ((ndvi_array - ndvi_array.min())/(ndvi_array.max()-ndvi_array.min())**2)
        ##print(pv)
        #land surface emissivity
        em = 0.004 * pv + 0.986

        lst_10 = np.where(check_7, ((brightness[10] / (1 + (0.00115 * brightness[10] / 1.4388) * np.log(em))) -273.15), -1)
        ##print(lst_10)      
        lst_11 = np.where(check_7, ((brightness[11] / (1 + (0.00115 * brightness[11] / 1.4388) * np.log(em))) -273.15), -1)
        lst_avg = (np.array(lst_10) + np.array(lst_11)) / 2
      #  #print(lst_avg)
        return lst_avg
    
    
    def mainfunction_landsat_calculation(self):
        region_box , bbox = self.__tuple_to_geojson(self.__extent)
        raw = self.__list_files(INPUT)
        year_month1_list, year_month2_list, product_list, metayear_month_list = self.__create_info_lists(raw)
        self.__iterative_clipping(year_month1_list, year_month2_list, product_list)
        
        date_list = [] 
        [date_list.append(x) for x in year_month1_list if x not in date_list]
        date_list2 = [] 
        [date_list2.append(x) for x in year_month2_list if x not in date_list2]
        prod_list = [] 
        [prod_list.append(x) for x in product_list if x not in prod_list]
        for i, k in enumerate (metayear_month_list):
            ##print(k)
            #ls_txt = open(f"{INPUT}/{k}", "r")
            ls_txt=f"{INPUT}/{k}"
            ##print(ls_txt)
            metadata= self.__get_metaData_dict(ls_txt)
        brightness = {}
        reflectance = {}
        self.__calculate_brightness(metadata, brightness)
        self.__calculate_reflectance(metadata, reflectance)           
        ##print(reflectance)
        for i, j in enumerate (date_list):
            ##print(j)
            ndvi = self.__calculate_NDVI(reflectance)
            ndmi = self.__calculate_NDMI(reflectance)
            ndbi = self.__calculate_NDBI(reflectance)
            vegetation_proportion = self.__calculate_vegetation(reflectance)
            mndwi = self.__calculate_MNDWI(reflectance)
            albedo = self.__calculate_ALBEDO(reflectance)
            lst_avg= self.__calculate_LST(brightness, reflectance)

           
        if not os.path.exists(f'{OUTPUT}/{self.__REGION}'):
            os.makedirs(f'{OUTPUT}/{self.__REGION}/')
        with rio.Env():
            region = rasterio.open(f"{INTER}/cropped_4.tif")
            profile = region.profile.copy()
            profile.update({'dtype': 'float64', 'count': 1})

        Path(f'{DATA}/output').mkdir(exist_ok=True)
        with rio.open(f'{DATA}/output/{self.__REGION}/{j}_{self.__REGION}_lst.tif', 'w', **profile) as f:
                f.write(lst_avg, 1)
                
        Path(f'{DATA}/output').mkdir(exist_ok=True)
        with rio.open(f'{DATA}/output/{self.__REGION}/{j}_{self.__REGION}_ndvi.tif', 'w', **profile) as f:
                f.write(ndvi, 1)
                
        Path(f'{DATA}/output').mkdir(exist_ok=True)
        with rio.open(f'{DATA}/output/{self.__REGION}/{j}_{self.__REGION}_ndmi.tif', 'w', **profile) as f:
                f.write(ndmi, 1)
                
        Path(f'{DATA}/output').mkdir(exist_ok=True)
        with rio.open(f'{DATA}/output/{self.__REGION}/{j}_{self.__REGION}_ndbi.tif', 'w', **profile) as f:
                f.write(ndbi, 1)
                
        Path(f'{DATA}/output').mkdir(exist_ok=True)
        with rio.open(f'{DATA}/output/{self.__REGION}/{j}_{self.__REGION}_albedo.tif', 'w', **profile) as f:
                f.write(albedo, 1)     
                
        Path(f'{DATA}/output').mkdir(exist_ok=True)
        with rio.open(f'{DATA}/output/{self.__REGION}/{j}_{self.__REGION}_mndwi.tif', 'w', **profile) as f:
                f.write(mndwi, 1)   
                
        return print('Check the output foder')