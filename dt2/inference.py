import argparse
import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio
from detectron2.engine import DefaultPredictor
from detectree2.models.train import setup_cfg

#predict tree presence with detectree2, and returns a dictionary with detected data
#
#ok, lets recap all the outputs parameters from detecree2 https://detectron2.readthedocs.io/en/latest/tutorials/models.html
#
#pred_masks:        A tensor of shape (N, H, W) giving a binary mask for each detected 
#                   instance (tree crown). Each mask shows the pixels belonging to that instance.
#pred_boxes:        Bounding boxes for each detected instance, usually as coordinates of 
#                   the rectangle surrounding the mask.
#scores:            Confidence scores for each detected instance, indicating prediction certainty.
#pred_classes:      Class labels for each instance (usually one class "tree" in Detectree2).
#pred_keypoints:    Keypoint detections as a tensor (N, number_keypoints, 3), with x, y coordinates and confidence scores.
def predict(
        predictor: DefaultPredictor, 
        image: np.ndarray, 
        conf: float = 0.5) -> Dict[str, Any]:
    
    outputs = predictor(image)
    outputDictionary = {
            "polygons": [],
            "scores": [],
            "num_detections": 0
        }
    
    #todo: support gpu inference?
    instances = outputs["instances"].to("cpu")
    
    if len(instances) == 0:
        return outputDictionary
    
    # filtering by conf treshold
    scores = instances.scores.numpy()
    validIndices = scores >= conf
    
    if not validIndices.any():
        return outputDictionary
    
    filteredScore = scores[validIndices].tolist()
    
    # mask extraction and polygons conversion with geopandas validation
    polygons = []
    if hasattr(instances, 'pred_masks'):
        masks = instances.pred_masks.numpy()[validIndices]
        crownGeometries = []
        
        for mask in masks:
            # extract contours from binary mask using opencv
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largestContour = max(contours, key=cv2.contourArea)
                polygonCoords = largestContour.reshape(-1, 2).tolist()
                
                # create shapely polygon for validation
                if len(polygonCoords) >= 3:  # need at least 3 points for polygon
                    try:
                        crownGeometry = Polygon(polygonCoords)
                        crownGeometries.append(crownGeometry)
                    except:
                        continue
        
        # validate and simplify crowns using geopandas (following official detectree2 workflow)
        if crownGeometries:
            crowns = gpd.GeoSeries(crownGeometries)
            crowns = crowns[crowns.is_valid]
            crowns = crowns.simplify(0.3)
            crownAreas = crowns.area.tolist()
            
            # extract coordinates from simplified geometries
            for geom in crowns:
                if geom and geom.is_valid and hasattr(geom, 'exterior'):
                    # convert coordinates to plain python lists for JSON serialization
                    coords = [[float(x), float(y)] for x, y in geom.exterior.coords[:-1]]
                    polygons.append(coords)
                else:
                    polygons.append([])
    
    results = {
        "polygons": polygons,
        "scores": filteredScore,
        "areas": crownAreas if 'crown_areas' in locals() else [],
        "num_detections": len(filteredScore)
    }
    
    return results

#poly display on map tile
def displayPolygons(image: np.ndarray,
                    results: Dict[str, Any],
                    outputPath: str) -> None:
    copiedImage = image.copy()
    
    # Only process polygons if there are any
    if results["polygons"]:
        for i, polygon in enumerate(results["polygons"]):
            if polygon:
                score = results["scores"][i] if i < len(results["scores"]) else 0
                
                # convert poly to numpy array for opencv
                polyNumpyArray = np.array(polygon, dtype=np.int32)
                overlay = copiedImage.copy()
                cv2.fillPoly(overlay, [polyNumpyArray], (0, 255, 0))
                copiedImage = cv2.addWeighted(copiedImage, 0.7, overlay, 0.3, 0)
                
                # draw outline
                cv2.polylines(copiedImage, [polyNumpyArray], True, (0, 255, 0), 2)
                
                # write conf score
                if len(polygon) > 0:
                    xCenter = int(np.mean([p[0] for p in polygon]))
                    yCenter = int(np.mean([p[1] for p in polygon]))
                    label = f"Tree: {score:.2f}"
                    cv2.putText(copiedImage, label, (xCenter-20, yCenter),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imwrite(outputPath, copiedImage)

#transform pixel to coordinates for geotiff
def pixelToCoords(transform, x, y):
    lon, lat = transform * (x, y)
    return lon, lat

#save json with detected values to file
def saveDetectedValuesToJSON(results, outputFile):
    with open(outputFile, 'w') as f:
        json.dump(results, f, indent=2)

#load and predict trees from specified image
def processImage(predictor: DefaultPredictor,
                 imagePath: str,
                 conf: float,
                 fromGeoTIFF: bool = False,
                 outputFolder: str = '.',
                 saveAnnotatedImage: bool = False) -> Dict[str, Any]:
    try:
        image = None
        transform = None
        profile = None
        coordSystem = None
        
        if fromGeoTIFF:
            # Load the geotiff image with rasterio to get geospatial information
            with rasterio.open(imagePath) as src:
                transform = src.transform
                coordSystem = src.crs
                profile = src.profile
                
                # read rgb bands
                imageArray = src.read([1, 2, 3])
                imageArray = np.transpose(imageArray, (1, 2, 0))
                image = cv2.cvtColor(imageArray, cv2.COLOR_RGB2BGR)  # convert rgb to bgr for opencv
        else:
            image = cv2.imread(imagePath)
            
        if image is None:
            raise ValueError(f"Could not load image from {imagePath}")
            
        results = predict(predictor, image, conf)
        
        # add geospatial info to results if processing geotiff
        if fromGeoTIFF:
            results["crs"] = str(coordSystem) if coordSystem else None
            results["transform"] = list(transform.to_gdal()) if transform else None
            
            # add geographic coordinates to polygons if available
            if transform is not None and "polygons" in results:
                geoPolygons = []
                for i, polygon in enumerate(results["polygons"]):
                    if polygon and len(polygon) > 0:
                        # calculate centroid for geographic coordinates
                        xCoords = [point[0] for point in polygon]
                        yCoords = [point[1] for point in polygon]
                        xCenter = np.mean(xCoords)
                        yCenter = np.mean(yCoords)
                        
                        # convert to geographic coordinates
                        lon, lat = pixelToCoords(transform, xCenter, yCenter)
                        geoPolygons.append({
                            "polygon": polygon,
                            "geo_center": {
                                "lon": float(lon),
                                "lat": float(lat)
                            }
                        })
                    else:
                        geoPolygons.append({"polygon": polygon})
                results["geo_polygons"] = geoPolygons
        
        # save visualization output if requested
        print(f"Saving annotated image to: {outputFolder}")
        if saveAnnotatedImage:
            try:
                # save geotiff output if input was geotiff
                if fromGeoTIFF:
                    # create geotiff with annotations overlaid (or original image if no detections)
                    annotatedGeoPath = Path(outputFolder) / (Path(imagePath).stem + "_annotated.tif")
                    imageForGeo = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert back to rgb for geotiff
                    
                    # overlay polygons on image (or save original if no detections)
                    copiedImage = imageForGeo.copy()
                    if results["polygons"]:
                        # First draw all filled polygons on the copied image
                        for i, polygon in enumerate(results["polygons"]):
                            if polygon:
                                numpyPoly = np.array(polygon, dtype=np.int32)
                                cv2.fillPoly(copiedImage, [numpyPoly], (0, 255, 0))
                                
                                # add confidence score label
                                if len(polygon) > 0:
                                    score = results["scores"][i] if i < len(results["scores"]) else 0
                                    xCenter = int(np.mean([p[0] for p in polygon]))
                                    yCenter = int(np.mean([p[1] for p in polygon]))
                                    label = f"Tree: {score:.2f}"
                                    cv2.putText(copiedImage, label, (xCenter-20, yCenter),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        
                        # Then apply transparency effect once after all polygons are drawn
                        if len(results["polygons"]) > 0:
                            overlay = copiedImage.copy()
                            copiedImage = cv2.addWeighted(copiedImage, 0.7, imageForGeo, 0.3, 0)
                        
                        # Finally draw all polygon outlines on top
                        for i, polygon in enumerate(results["polygons"]):
                            if polygon:
                                numpyPoly = np.array(polygon, dtype=np.int32)
                                cv2.polylines(copiedImage, [numpyPoly], True, (0, 255, 0), 2)
                    
                    # save as geotiff
                    # Remove PIXELTYPE=SIGNEDBYTE from profile to avoid warning in GDAL 3.7+
                    if 'PIXELTYPE' in profile and profile['PIXELTYPE'] == 'SIGNEDBYTE':
                        profile = profile.copy()
                        del profile['PIXELTYPE']
                    with rasterio.open(str(annotatedGeoPath), 'w', **profile) as dst:
                        # convert rgb back to bgr and then to rasterio format
                        imageArrayForCV = cv2.cvtColor(copiedImage, cv2.COLOR_RGB2BGR)
                        imageArrayForCV = np.transpose(imageArrayForCV, (2, 0, 1))
                        dst.write(imageArrayForCV)
                    print(f"GeoTIFF annotated image saved to: {annotatedGeoPath}")
                else:
                    annotatedPath = Path(outputFolder) / (Path(imagePath).stem + "_annotated.jpg")
                    displayPolygons(image, results, str(annotatedPath))
                    print(f"Annotated image saved to: {annotatedPath}")

            except Exception as e:
                print(f"Error saving annotated image: {e}")

        return results
        
    except Exception as e:
        print(f"Error processing {imagePath}: {str(e)}")
        return {"error": str(e), "num_detections": 0}

# yes, it'll detect trees, need any other comment? :)
def detectTrees(
        modelPath: str,
        imagePath: str,
        conf: float = 0.5,
        fromGeoTIFF: bool = False,
        outputFolder: str = '.',
        saveAnnotated: bool = True) -> Dict[str, Any]:

    os.makedirs(outputFolder, exist_ok=True)
    
    cfg = setup_cfg(update_model=modelPath)
    cfg.OUTPUT_DIR = outputFolder
    #add support to cuda? maybe mps if available?
    cfg.MODEL.DEVICE = "cpu"
 
    print(f"Using device: {cfg.MODEL.DEVICE}")
    
    predictor = DefaultPredictor(cfg)
    results = processImage(predictor, imagePath, conf, fromGeoTIFF, outputFolder, saveAnnotated)
    
    # add image info to results
    results["image_path"] = str(imagePath)
    if fromGeoTIFF:
        with rasterio.open(imagePath) as src:
            results["image_size"] = {
                "width": src.width,
                "height": src.height
            }
    else:
        image = cv2.imread(imagePath)
        if image is not None:
            results["image_size"] = {
                "width": image.shape[1],
                "height": image.shape[0]
            }
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", required=True)
    parser.add_argument("--image_path", "-i", required=True)
    parser.add_argument("--confidence_threshold",  "-c", type=float, default=0.5)
    parser.add_argument("--geotiff", "-g", action='store_true')
    parser.add_argument("--saveoutput", "-s", action='store_true')
    parser.add_argument("--output-folder", "-o", type=str, default='.')
    parser.add_argument("--save-json", "-j", action='store_true')
    parser.add_argument("--no_annotation", action="store_true")
    
    args = parser.parse_args()
    
    # determine if we should save annotated images based on flags
    saveAnnotated = not args.no_annotation and args.saveoutput
    
    results = detectTrees(
        args.model_path,
        args.image_path,
        args.confidence_threshold,
        fromGeoTIFF=args.geotiff,
        outputFolder=args.output_folder,
        saveAnnotated=saveAnnotated
    )
    
    if "error" not in results:
        print(f"Detection complete: {results['num_detections']} trees detected")
        
        # save json results if requested
        if args.save_json:
            jsonOutputPath = Path(args.output_folder) / (Path(args.image_path).stem + "_results.json")
            saveDetectedValuesToJSON(results, str(jsonOutputPath))
            print(f"JSON results saved to: {jsonOutputPath}")
    else:
        print(f"Error: {results['error']}")


if __name__ == "__main__":
    main()