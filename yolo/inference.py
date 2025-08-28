import cv2
from ultralytics import YOLO
from pathlib import Path
import argparse
import rasterio
import numpy as np
import json

#transform pixel to coordinates
def pixelToCoords(transform, x, y):
    lon, lat = transform * (x, y)
    return lon, lat

#save json with detected values to file
def saveDetectedValuesToJSON(results, outputFile):
    with open(outputFile, 'w') as f:
        json.dump(results, f, indent=2)

#detect a tree with specified YOLO model
def treeDetector(imagePath,
                 modelPath,
                 conf=0.05,
                 fromGeoTIFF=False,
                 saveCVOutput=False,
                 outputFolder='.'):

    model = YOLO(modelPath)
    results = []

    imgWidth = 0
    imgHeight = 0
    coordSystem = None

    img = cv2.imread(imagePath)

    if fromGeoTIFF:
        # Load the geotiff image with rasterio to get geospatial information
        with rasterio.open(imagePath) as src:
            transform = src.transform
            coordSystem = src.crs
            profile = src.profile
            
            imageArray = src.read([1, 2, 3])
            imageArray = np.transpose(imageArray, (1, 2, 0))
            imageArrayForCV = imageArray.copy()
            
            results = model(imageArray, conf=conf)
            coordSystem = src.crs
            imgWidth = src.width
            imgHeight = src.height
            
    else:
        results = model(img, conf=conf)
        imgWidth = img.shape[1]
        imgHeight = img.shape[0]
    
    treeCounter = 0
    detectedTrees = []

    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            for box, confidence in zip(boxes, confidences):
                if confidence >= conf:
                    lon = 0.0
                    lat = 0.0
                    boundingCoords = None

                    x1, y1, x2, y2 = box.astype(int)
                    treeCounter += 1

                    if fromGeoTIFF:
                        xCenter = (x1 + x2) / 2
                        yCenter = (y1 + y2) / 2

                        lon, lat = pixelToCoords(transform, xCenter, yCenter)

                        boundingCoords = {
                            "top_left": pixelToCoords(transform, x1, y1),
                            "top_right": pixelToCoords(transform, x2, y1),
                            "bottom_right": pixelToCoords(transform, x2, y2),
                            "bottom_left": pixelToCoords(transform, x1, y2)
                        }

                    if saveCVOutput:
                        cv2.rectangle(imageArrayForCV, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Add detection to list
                    detectedTrees.append({
                        "id": treeCounter,
                        "confidence": float(confidence),
                        "bbox": {
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2)
                        },
                        "center": {
                            "pixel": {
                                "x": float(xCenter),
                                "y": float(yCenter)
                            },
                            "geo": {
                                "lon": float(lon),
                                "lat": float(lat)
                            }
                        },
                        "bbox_geo": boundingCoords,
                    })

    print(f"Processing completed for {Path(imagePath).name}")

    if saveCVOutput:
        print(f"Saving annotated image to: {outputFolder}")
        try:
            if fromGeoTIFF:
                outputPath = Path(outputFolder) / (Path(imagePath).stem + "_result.tif")
                # Remove PIXELTYPE=SIGNEDBYTE from profile to avoid warning in GDAL 3.7+
                if 'PIXELTYPE' in profile and profile['PIXELTYPE'] == 'SIGNEDBYTE':
                    profile = profile.copy()
                    del profile['PIXELTYPE']
                with rasterio.open(str(outputPath), 'w', **profile) as dst:
                     #transpose back image bands to be compatbile with rasterio
                    imageArrayForCV = np.transpose(imageArrayForCV, (2, 0, 1))
                    dst.write(imageArrayForCV)
                print(f"GeoTIFF saved in path: {outputPath}")
            else:
                outputPath = Path(outputFolder) / (Path(imagePath).stem + "_result.jpg")
                cv2.imwrite(str(outputPath), img)
                print(f"Img saved in path: {outputPath}")
        except Exception as e:
            print(f"Error saving annotated image: {e}")

    print(f"Num of trees detected: {treeCounter}")
    

    results_dict = {
        "image_path": str(imagePath),
        "crs": str(coordSystem) if coordSystem else None,
        "image_size": {
            "width": imgWidth,
            "height": imgHeight
        },
        "trees_detected": treeCounter,
        "detections": detectedTrees
    }
    
    return results_dict

# yes i know, there are no help messages, my bad
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', required=True)
    parser.add_argument('--model', '-m', required=True)
    parser.add_argument('--conf', '-c', type=float, default=0.25)
    parser.add_argument('--geotiff', '-g', action='store_true')
    parser.add_argument('--saveoutput', '-s', action='store_true')
    parser.add_argument('--output-folder', '-o', type=str, default='.')
    parser.add_argument('--save-json', '-j', action='store_true')
    args = parser.parse_args()
    
    results = treeDetector(args.image, args.model, args.conf, args.geotiff, args.saveoutput, args.output_folder)
    
    if args.save_json:
        jsonOutputPath = Path(args.output_folder) / (Path(args.image).stem + "_results.json")
        saveDetectedValuesToJSON(results, jsonOutputPath)

if __name__ == "__main__":
    main()