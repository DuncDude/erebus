import ee
import geemap
import requests
import time
import rasterio
import gdown
import os
import cv2
import numpy as np
#import torch


PROJECT_ID = 'erebus-469901'
DATASET = 'COPERNICUS/S2_SR_HARMONIZED'
LOCALIMAGEPATH = './my_aoi_image.tif'
#defined rectangle area
SW = {'lat': float('69.47064973844004'), 'long': float('-98.39302540719589')}
NE = {'lat': float('69.67831460276568'), 'long': float('-97.94473564995305')}

def auth():
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)



def define_area():
    # Define AOI using (west, south, east, north)
    aoi = ee.Geometry.BBox(
        SW['long'],  # west
        SW['lat'],   # south
        NE['long'],  # east
        NE['lat']    # north
    )
    return aoi



def load_sentinel_imagery(aoi):
# Assuming ee is initialized and aoi is defined
# Load Sentinel-2 Surface Reflectance data
    sentinel_collection = ee.ImageCollection(DATASET) \
        .filterBounds(aoi) \
        .filterDate('2023-06-01', '2023-08-01') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))

    # Take median to reduce clouds/noise
    sentinel_image = sentinel_collection.median().clip(aoi)

    #return image(s)
    return sentinel_image



def visualize(sentinel_image):
    Map = geemap.Map(center=[(SW['lat'] + NE['lat'])/2, (SW['long'] + NE['long'])/2], zoom=10)
    Map.addLayer(sentinel_image, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}, 'Sentinel-2 RGB')
    Map



def export_to_google(sentinel_image):
    task = ee.batch.Export.image.toDrive(
    image=sentinel_image.select(['B4', 'B3', 'B2']),
    description='sentinel_export',
    folder='earth_engine_exports',
    fileNamePrefix='my_aoi_image',
    region=aoi.bounds().getInfo()['coordinates'],
    scale=10,
    crs='EPSG:4326',
    maxPixels=1e13
    )
    task.start()
    print('Export task started. Check your Google Drive for the image.')
    while task.status()['state'] in ['READY', 'RUNNING']:
        print('Task is', task.status()['state'])    
        time.sleep(2)  # wait 10 seconds before checking again

    print('Task finished with state:', task.status()['state'])
    if task.status()['state'] == 'COMPLETED':
        print('Export succeeded! Check your Google Drive folder.')
    else:
        print('Export failed or was cancelled:', task.status())



def download_local(aoi, sentinel_image):
    # Prepare the download URL parameters
    url = sentinel_image.select(['B4', 'B3', 'B2']).getDownloadURL({
    'scale': 10,
    'crs': 'EPSG:4326',
    'region': aoi.bounds().getInfo()['coordinates'],
    'fileFormat': 'GeoTIFF'
    })
    print('Download URL:', url)

    # Optional: Download the file with requests
    response = requests.get(url)

    # Save to file in your current directory
    with open('sentinel_image.tif', 'wb') as f:
        f.write(response.content)

    print('Image downloaded to sentinel_image.tif')



def download_proccessed_image_from_drive():

    file_id = 'my_aoi_image.tif'  # get this from the Drive file's shareable link
    output = 'my_aoi_image.tif'

    gdown.download(f'https://drive.google.com/uc?id={file_id}', output, quiet=False)



def load_local_image():
    print(f"Loading local image '{LOCALIMAGEPATH}'...")
    with rasterio.open(LOCALIMAGEPATH) as src:
        img = src.read()
        profile = src.profile

    img = np.moveaxis(img, 0, -1)  # (H, W, C)
    img = img.astype(np.float32) / 10000.0
    img = np.clip(img, 0, 1)

    print(f"Image shape (H x W x C): {img.shape}")
    return img, profile


# Step 2: Tile and save to YOLOv5 image folder
def tile_and_save(img, tile_size=640, overlap=0, save_dir='./yolov5/data/images/test'):
    print("Tiling image...")
    os.makedirs(save_dir, exist_ok=True)
    h, w, c = img.shape
    step = tile_size - overlap
    tile_paths = []

    for y in range(0, h - tile_size + 1, step):
        for x in range(0, w - tile_size + 1, step):
            tile = img[y:y + tile_size, x:x + tile_size, :]
            tile_bgr = cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            tile_filename = f'tile_{y}_{x}.jpg'
            tile_path = os.path.join(save_dir, tile_filename)
            cv2.imwrite(tile_path, tile_bgr)
            tile_paths.append(tile_path)

    print(f"Saved {len(tile_paths)} tiles to {save_dir}")
    return tile_paths


# # Step 3: Run inference using YOLOv5
# def run_yolo_inference(source_dir=YOLO_TILE_DIR, model_name=YOLO_MODEL, save_dir=YOLO_OUTPUT_DIR, run_name=YOLO_OUTPUT_NAME):
#     print("Loading YOLOv5 model...")
#     model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)

#     print(f"Running inference on tiles in: {source_dir}")
#     results = model(source_dir)
#     results.save(save_dir=os.path.join(save_dir, run_name))

#     print("Saving detection results...")
#     df = results.pandas().xyxy[0]
#     csv_path = os.path.join(save_dir, run_name, 'detections.csv')
#     df.to_csv(csv_path, index=False)
#     print(f"Detections saved to: {csv_path}")

#     return df



def run_inference():
    # Load model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Inference on an image or folder of images
    results = model('./yolov5/data/images/test')

    # Print results
    results.print()

    # Save results (images with bounding boxes)
    results.save()

    # Get pandas dataframe of results
    df = results.pandas().xyxy[0]
    print(df)

def main():
    print("authenticating and init....")
    auth()
    #print("defining area")
    #aoi = define_area()
    #print("loading iamge")
    #sentinel_image = load_sentinel_imagery(aoi)
    # visualize(sentinel_image)
    # print("exporting to google drive")
    # export_to_google(sentinel_image)
    # print("downloading locally")
    # download_local(aoi, sentinel_image) #only works with small files sub 300mb

    print("loading local image 'my_aoi_image.tif' from working dir")
    img, profile = load_local_image()
    print("tiling image")
    tile_and_save(img)

    # print("detecting objects")
    # run_inference()

main()

