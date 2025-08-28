import os
import json
import argparse
import glob

#convert yolo bounding box to polygon coordinates
def boundingBoxToPoligon(bbox, transform=None):
    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    
    polygonCoords = [
        [x1, y1],
        [x2, y1], 
        [x2, y2],
        [x1, y2],
        [x1, y1] 
    ]
    
    if transform:
        geoCoords = []
        for x, y in polygonCoords:
            lon, lat = transform * (x, y)
            geoCoords.append([lon, lat])
        return geoCoords
    
    return polygonCoords

#convert yolo detection to geojson feature
def yoloToGeoJsonFeature(detection, imageInfo=None):
    geoCoords = None
    if detection.get('bbox_geo'):
        bboxGeo = detection['bbox_geo']
        geoCoords = [
            [bboxGeo['top_left'][0], bboxGeo['top_left'][1]],
            [bboxGeo['top_right'][0], bboxGeo['top_right'][1]], 
            [bboxGeo['bottom_right'][0], bboxGeo['bottom_right'][1]],
            [bboxGeo['bottom_left'][0], bboxGeo['bottom_left'][1]],
            [bboxGeo['top_left'][0], bboxGeo['top_left'][1]]  # close polygon
        ]
    else:
        geoCoords = boundingBoxToPoligon(detection['bbox']) #fallback
    
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [geoCoords]
        },
        "properties": {
            "id": detection['id'],
            "confidence": detection['confidence'],
            "detection_type": "yolo_bbox",
            "center_lon": detection.get('center', {}).get('geo', {}).get('lon'),
            "center_lat": detection.get('center', {}).get('geo', {}).get('lat'),
            "bbox_pixel": detection['bbox']
        }
    }
    
    if imageInfo:
        feature['properties']['source_image'] = imageInfo.get('image_path')
        feature['properties']['crs'] = imageInfo.get('crs')
    
    return feature

#convert detectree2 detection to geojson feature
def detectree2ToGeoJsonFeature(detection, detectionIdx, imageInfo=None):
    #use geo_polygons if available, otherwise pixel polygons
    if imageInfo and imageInfo.get('geo_polygons') and detectionIdx < len(imageInfo['geo_polygons']):
        geoPolygon = imageInfo['geo_polygons'][detectionIdx]
        coords = geoPolygon.get('polygon', [])
        centerLon = geoPolygon.get('geo_center', {}).get('lon')
        centerLat = geoPolygon.get('geo_center', {}).get('lat')
        
        if coords and imageInfo.get('transform'):
            transform = imageInfo['transform']
            geoCoords = []
            for x, y in coords:
                #apply transformation
                lon = transform[0] + x * transform[1] + y * transform[2]
                lat = transform[3] + x * transform[4] + y * transform[5]
                geoCoords.append([lon, lat])
            coords = geoCoords
    else: #fallback
        coords = detection
        if coords and imageInfo and imageInfo.get('transform'):
            transform = imageInfo['transform']
            geoCoords = []
            for x, y in coords:
                lon = transform[0] + x * transform[1] + y * transform[2]
                lat = transform[3] + x * transform[4] + y * transform[5]
                geoCoords.append([lon, lat])
            coords = geoCoords
        centerLon = None
        centerLat = None
    
    #echeck if poly are closed
    if coords and len(coords) > 0 and coords[0] != coords[-1]:
        coords.append(coords[0])
    
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords] if coords else []
        },
        "properties": {
            "id": detectionIdx + 1,
            "detection_type": "detectree2_polygon",
            "center_lon": centerLon,
            "center_lat": centerLat
        }
    }
    
    if imageInfo:
        feature['properties']['source_image'] = imageInfo.get('image_path')
        feature['properties']['crs'] = imageInfo.get('crs')
    
    return feature

#parse json detection file and convert to geojson features
def parseJsonFile(jsonPath):
    try:
        with open(jsonPath, 'r') as f:
            data = json.load(f)
        
        features = []
        
        #we area assuming that if contains detection is format yolo, 
        #if contains polygons contains detectree2: is it weak?
        #todo: make stronger assumptions
        if 'detections' in data: #yolo format
            for detection in data['detections']:
                feature = yoloToGeoJsonFeature(detection, data)
                features.append(feature)
                
        elif 'polygons' in data: #detectree2 format
            polygons = data['polygons']
            scores = data.get('scores', [])
            
            for i, polygon in enumerate(polygons):
                if polygon:  # skip empty polygons
                    feature = detectree2ToGeoJsonFeature(polygon, i, data)
                    if i < len(scores):
                        feature['properties']['confidence'] = scores[i]
                    features.append(feature)
        
        print(f"Processed {jsonPath}: {len(features)} detections")
        return features
        
    except Exception as e:
        print(f"Error processing {jsonPath}: {str(e)}")
        return []

#find all json files in a folder
def findJsonFiles(folderPath):
    jsonPatterns = ['*_results.json', '*.json']
    jsonFiles = []
    
    for pattern in jsonPatterns:
        jsonFiles.extend(glob.glob(os.path.join(folderPath, pattern)))
    
    return sorted(list(set(jsonFiles)))

#merge multiple json files into single geojson
def mergeJsonToGeoJson(jsonFiles, outputPath):
    allFeatures = []
    
    for jsonFile in jsonFiles:
        features = parseJsonFile(jsonFile)
        allFeatures.extend(features)
    
    #create geojson structure with proper CRS specification
    geoJson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": "EPSG:4326"
            }
        },
        "features": allFeatures,
        "properties": {
            "total_detections": len(allFeatures),
            "source_files": len(jsonFiles),
            "generated_by": "satellite_tree_cover_detection_pipeline"
        }
    }
    
    with open(outputPath, 'w') as f:
        json.dump(geoJson, f, indent=2)
    
    print(f"Created GeoJSON with {len(allFeatures)} features from {len(jsonFiles)} files")
    print(f"Output saved to: {outputPath}")
    
    return geoJson

#process folder and convert all json to geojson
def processFolder(inputFolder, outputPath):
    if not os.path.exists(inputFolder):
        print(f"Error: Input folder not found: {inputFolder}")
        return False

    jsonFiles = findJsonFiles(inputFolder)
    
    if not jsonFiles:
        print(f"No JSON files found in {inputFolder}")
        return False
    
    print(f"Found {len(jsonFiles)} JSON files to process")

    os.makedirs(os.path.dirname(outputPath), exist_ok=True)
    geoJson = mergeJsonToGeoJson(jsonFiles, outputPath)
    
    return len(geoJson['features']) > 0

def main():
    parser = argparse.ArgumentParser()
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', '-i', help='Input folder containing JSON files')
    group.add_argument('--files', nargs='+', help='List of specific JSON files to merge')
    
    parser.add_argument('--output', '-o', required=True, help='Output GeoJSON file path')
    
    args = parser.parse_args()
    
    try:
        if args.input:
            success = processFolder(args.input, args.output)
        else:
            success = len(mergeJsonToGeoJson(args.files, args.output)['features']) > 0
            
        if success:
            print("JSON to GeoJSON conversion completed successfully")
        else:
            print("Failed to convert JSON files")
            
    except Exception as e:
        print(f"Conversion failed: {str(e)}")

if __name__ == "__main__":
    main()