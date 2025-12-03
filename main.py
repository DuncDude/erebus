import ee
import geemap
import requests
import time
import rasterio
import gdown
import os
import cv2
import numpy as np
from glob import glob
import torch
import cv2
#import torch


PROJECT_ID = 'erebus-469901'
DATASET = 'COPERNICUS/S2_SR_HARMONIZED'
#DATASET ='DLR/WSF/WSF2015/v1'
#DATASET = 'JRC/GHSL/P2023A/GHS_BUILT_C'
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
    dataset_id = DATASET
    print(f"Loading dataset: {dataset_id}")

    collection = ee.ImageCollection(dataset_id).filterBounds(aoi)

    # Handle Sentinel-2 vs other datasets differently
    if "COPERNICUS/S2" in dataset_id:
        collection = (collection
                      .filterDate('2023-06-01', '2023-08-01')
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)))
        image = collection.median().clip(aoi)

    elif "COPERNICUS/S1" in dataset_id:
        collection = (collection
                      .filterDate('2023-06-01', '2023-08-01')
                      .filter(ee.Filter.eq('instrumentMode', 'IW')))
        image = collection.mean().clip(aoi)

    else:
        # Non-Sentinel datasets (GHSL, MODIS, etc.)
        image = ee.ImageCollection(dataset_id).filterBounds(aoi).mean().clip(aoi)

    # Print available bands for debugging
    try:
        print("Available bands:", image.bandNames().getInfo())
    except Exception as e:
        print("⚠️ Could not retrieve band names:", e)

    return image



def visualize(sentinel_image):
    Map = geemap.Map(center=[(SW['lat'] + NE['lat']) / 2,
                             (SW['long'] + NE['long']) / 2],
                     zoom=8)

    bands = sentinel_image.bandNames().getInfo()
    if set(['B4', 'B3', 'B2']).issubset(bands):
        vis_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}
    else:
        vis_params = {'min': 0, 'max': 1}

    Map.addLayer(sentinel_image, vis_params, 'Dataset Preview')
    Map





def export_to_google(sentinel_image, aoi, description='sentinel_export', folder='earth_engine_exports'):
    # Check if the image exists and has bands
    if sentinel_image is None:
        raise ValueError("The provided image is None. Check your filters or image collection.")
    
    band_names = sentinel_image.bandNames().getInfo()
    if not band_names:
        raise ValueError("The image has no bands. Verify the dataset and filters.")
    
    print("Available bands:", band_names)

    # Automatically select suitable bands
    if set(['B4', 'B3', 'B2']).issubset(band_names):
        image_to_export = sentinel_image.select(['B4', 'B3', 'B2'])  # Sentinel-2 RGB
        scale = 10
    elif set(['VV', 'VH']).intersection(band_names):
        image_to_export = sentinel_image.select(['VV'])  # Sentinel-1 backscatter
        scale = 10
    elif len(band_names) == 1:
        # Single-band datasets like GHSL or settlement maps
        image_to_export = sentinel_image.select(band_names[0])
        scale = 30
    else:
        # Fall back to the first three available bands
        image_to_export = sentinel_image.select(band_names[:3])
        scale = 30

    # Define region and start export
    region_coords = aoi.bounds().getInfo()['coordinates']

    task = ee.batch.Export.image.toDrive(
        image=image_to_export,
        description=description,
        folder=folder,
        fileNamePrefix='my_aoi_image.tif',
        region=region_coords,
        scale=scale,
        crs='EPSG:4326',
        maxPixels=1e13
    )
    task.start()
    print(f"✅ Export task '{description}' started...")

    # Monitor progress
    while task.status()['state'] in ['READY', 'RUNNING']:
        print('Task is', task.status()['state'])
        time.sleep(5)

    state = task.status()['state']
    print(f"Task finished with state: {state}")
    if state == 'COMPLETED':
        print("🎉 Export succeeded! Check your Google Drive folder.")
    else:
        print("❌ Export failed or was cancelled:", task.status())




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
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.to('cuda')  # use GPU
    results = model('yolov5/data/images/images.jpeg')
    #image_paths = glob(os.path.join('yolov5/data/images/test', '*.jpg'))

    # Run inference on all images
    # results = model(image_paths)

    results.print()

    # Save results (images with bounding boxes)
    results.save()

    # Get pandas dataframe of results
    df = results.pandas().xyxy[0]
    print(df)



def run_inference_cpu():
    # Load YOLOv5s model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.to('cpu')  # or 'cuda' if you have a GPU

    # Get all image files in directory (jpg, png, jpeg)
    image_dir = 'yolov5/data/images/test'
    image_paths = glob(os.path.join(image_dir, '*.*'))  # matches any file type

    # Filter only supported image types (optional)
    image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Found {len(image_paths)} images")
    
    # Run inference on all images
    results = model(image_paths)

    # Print summary
    results.print()

    # Save annotated results (images with boxes)
    results.save(save_dir='runs/detect/cpu_inference')

    # Access detections as pandas dataframe
    for i, path in enumerate(image_paths):
        print(f"\nResults for: {path}")
        print(results.pandas().xyxy[i])  # bounding boxes per image



def detect_rectangles(image_dir):
    # Get all image files (jpg, png, jpeg)
    image_paths = glob(os.path.join(image_dir, '*.*'))
    image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Found {len(image_paths)} images in '{image_dir}'")

    # Output folder
    output_dir = os.path.join(image_dir, 'rectangles_output')
    os.makedirs(output_dir, exist_ok=True)

    for path in image_paths:
        print(f"Processing: {os.path.basename(path)}")

        # Read and process image
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through contours
        for contour in contours:
            # Approximate the contour to a polygon
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

            # A rectangle has 4 sides
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the image with detected rectangles
        output_path = os.path.join(output_dir, os.path.basename(path))
        cv2.imwrite(output_path, image)

        print(f"Saved result to: {output_path}")

    print("✅ Rectangle detection complete!")


def main():
    choice = input("download image? y/n")
    if choice == 'y':
            
        print("authenticating and init....")
        auth()
        print("defining area")
        aoi = define_area()
        print("loading image")
        sentinel_image = load_sentinel_imagery(aoi)
        visualize(sentinel_image)
        print("exporting to google drive")
        export_to_google(sentinel_image, aoi)
        # print("downloading locally")
        # download_local(aoi, sentinel_image) #only works with small files sub 300mb
    else:
        pass

    choice2 = input("procces image? y/n")
    if choice == 'y':
        print("loading local image 'my_aoi_image.tif' from working dir")
        img, profile = load_local_image()
        print("tiling image")
        tile_and_save(img)
        print("open cv dectecing rectangles")
        detect_rectangles("yolov5/data/images/test")
        # print("detecting objects")
        # run_inference()

        # print("detecting objects")
        # run_inference_cpu()
    else:
        return

main()
