import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import glob

# import existing inference modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolo'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'detectree2'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'tiles'))

from yolo.inference import treeDetector as yoloDetector
from dt2.inference import detectTrees as detectree2Detector
from tiles.split_geotiff_tiles import splitGeotiffInTiles
from tiles.merge_geotiff_tiles import mergeTilesFromFolder

# import GeoJSON conversion functions
sys.path.append(os.path.dirname(__file__))
from json_to_geojson import findJsonFiles, mergeJsonToGeoJson

# logging, maybe useless? should be removed to make the script smaller?
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TreeDetectionPipeline:
    
    # init modeltype and other shenaningans. we must keep compatible with yolo and detectree,
    # so the user can choose his preferred one
    def __init__(self, modelType: str, modelPath: str, confidence: float = 0.5):
        self.modelType = modelType.lower()
        self.modelPath = modelPath
        self.confidence = confidence
        
        if self.modelType not in ['yolo', 'detectree2']:
            raise ValueError("modelType must be either 'yolo' or 'detectree2'")
        
        if not os.path.exists(self.modelPath):
            raise FileNotFoundError(f"Model file not found: {self.modelPath}")
            
        logger.info(f"Initialized {self.modelType.upper()} pipeline with confidence {confidence}")
    
    # called to predict trees presence on a single file.
    # this calls the yolo detector or the detectree2 detector declared in other folders
    def detectSingleImage(self, imagePath: str, outputFolder: str,
                          saveImages: bool = True,
                          saveJson: bool = True) -> Dict[str, Any]:
        logger.info(f"Processing single image: {imagePath}")
        
        os.makedirs(outputFolder, exist_ok=True)
        
        # geotiff is true by default, other formats are not supported in this pipeline
        try:
            if self.modelType == 'yolo':
                results = yoloDetector(
                    imagePath=imagePath,
                    modelPath=self.modelPath,
                    conf=self.confidence,
                    fromGeoTIFF=True,
                    saveCVOutput=saveImages,
                    outputFolder=os.path.join(outputFolder, "annotated")
                )
            else:  # detectree2
                results = detectree2Detector(
                    modelPath=self.modelPath,
                    imagePath=imagePath,
                    conf=self.confidence,
                    fromGeoTIFF=True,
                    outputFolder=os.path.join(outputFolder, "annotated"),
                    saveAnnotated=saveImages
                )
            
            # user can chooose if he want to save json or not, and also if he want to save annotated
            # images. if he choose not to save json, it would be impossibile to generate GeoJson at the end
            if saveJson:
                json_path = Path(outputFolder) / "json" / f"{Path(imagePath).stem}_results.json"
                with open(json_path, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"JSON results saved to: {json_path}")
            
            if saveImages:
                logger.info(f"Annotated images should be saved in: {os.path.join(outputFolder, 'annotated')}")
            else:
                logger.info("Saving annotated images was disabled")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing {imagePath}: {str(e)}")
            return {"error": str(e), "image_path": imagePath}
    
    # process an entire folter of tiles
    def processTilesFolder(self, 
                           inputFolder: str, 
                           outputFolder: str,
                           saveImages: bool = True, 
                           saveJson: bool = True) -> List[Dict[str, Any]]:
        logger.info(f"Processing tiles folder: {inputFolder}")
        
        #we don't want any non geotiff file
        geotiff_extensions = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
        
        tiles = []
        for extension in geotiff_extensions:
            pattern_path = os.path.join(inputFolder, extension)
            found_files = glob.glob(pattern_path)
            tiles.extend(found_files)
        
        results = []
        for i, tile_path in enumerate(tiles, 1):
            logger.info(f"Processing tile {i}/{len(tiles)}: {os.path.basename(tile_path)}")
            
            result = self.detectSingleImage(
                imagePath=tile_path,
                outputFolder=outputFolder,
                saveImages=saveImages,
                saveJson=saveJson
            )
            
            results.append(result)
            
            if i % 10 == 0 or i == len(tiles):
                logger.info(f"Completed {i}/{len(tiles)} tiles")
        
        return results
    
    # we can also start from a big geotiff file instead of a folder of geotiff
    # is it dangerous? what if the file is 1G?
    # todo: should be removed?
    def processLargeGeotiff(self, 
                            geotiffPath: str, 
                            outputFolder: str,
                            tileSize: int = 640, 
                            saveImages: bool = True,
                            saveJson: bool = True) -> Tuple[List[Dict[str, Any]], str]:
        logger.info(f"Processing large GeoTIFF: {geotiffPath}")
        
        base_name = Path(geotiffPath).stem
        tilesFolder = os.path.join(outputFolder, f"{base_name}_tiles")

        logger.info(f"Splitting GeoTIFF into {tileSize}x{tileSize} tiles...")
        tileFiles = splitGeotiffInTiles(
            inputPath=geotiffPath,
            outputFolder=tilesFolder,
            tileSize=tileSize
        )
        
        if not tileFiles:
            logger.error("Failed to create tiles from GeoTIFF")
            return [], tilesFolder

        # process tiles
        results = self.processTilesFolder(
            inputFolder=tilesFolder,
            outputFolder=outputFolder,
            saveImages=saveImages,
            saveJson=saveJson
        )
        
        return results, tilesFolder
    
    # this function merge all the annotated tiles, is useless if you consider the original aim of this
    # project that should only geo-locate trees, but is useful for debugging
    # and is soo cool to have a big image with all annotations
    def mergeAnnotatedTiles(self, 
                            pipelineSummary: Dict[str, Any], 
                            outputFolder: str) -> bool:
        logger.info("Starting tile merging process...")
        
        try:
            annotated_pattern = f"*_result.tif" if self.modelType == 'yolo' else f"*_annotated.tif"
            annotated_files = glob.glob(os.path.join(outputFolder, "annotated", annotated_pattern))
            
            if not annotated_files:
                logger.warning(f"No annotated tiles found with pattern: {annotated_pattern}")
                return False
            
            logger.info(f"Found {len(annotated_files)} annotated tiles to merge")
            
            input_basename = Path(pipelineSummary.get("input_path", "merged")).stem
            merged_filename = f"{input_basename}_merged_annotated.tif"
            merged_path = os.path.join(outputFolder, "annotated", merged_filename)
            
            success = mergeTilesFromFolder(
                folderPath=os.path.join(outputFolder, "annotated"),
                outputPath=merged_path,
                basenameFilter=None,  # will find tiles by pattern matching
                nodata=None
            )
            
            if success:
                logger.info(f"Successfully merged annotated tiles to: {merged_path}")
                pipelineSummary["merged_annotated_geotiff"] = merged_path
                return True
            else:
                logger.error("Failed to merge annotated tiles")
                return False
                
        except Exception as e:
            logger.error(f"Error during tile merging: {str(e)}")
            return False
    
    # this can convert the json woth detected trees into a geojson
    def convertToGeojson(self, pipelineSummary: Dict[str, Any], outputFolder: str) -> bool:
        logger.info("Starting JSON to GeoJSON conversion...")
        
        try:
            jsonFiles = findJsonFiles(os.path.join(outputFolder, "json"))
            
            if not jsonFiles:
                logger.warning(f"No JSON files found in {outputFolder}")
                return False
            
            logger.info(f"Found {len(jsonFiles)} JSON files to convert")
            
            inputBasename = Path(pipelineSummary.get("input_path", "detections")).stem
            geojsonFilename = f"{inputBasename}_detections.geojson"
            geojsonPath = os.path.join(outputFolder, "json", geojsonFilename)
            
            geoJson = mergeJsonToGeoJson(jsonFiles, geojsonPath)
            
            # geojson is composed by a number of features, that are polygons detected by 
            # a ML algorythm
            if geoJson and geoJson.get('features'):
                logger.info(f"Successfully created GeoJSON with {len(geoJson['features'])} features")
                pipelineSummary["geojson_output"] = geojsonPath
                pipelineSummary["total_geojson_features"] = len(geoJson['features'])
                return True
            else:
                logger.error("Failed to create GeoJSON or no features found")
                return False
                
        except Exception as e:
            logger.error(f"Error during GeoJSON conversion: {str(e)}")
            return False
    
    # that's the real deal
    def runPipeline(self, 
                    inputPath: str, 
                    outputFolder: str = "output",
                    tileSize: int = 640, 
                    saveImages: bool = True,
                    saveJson: bool = True, 
                    mergeTiles: bool = True,
                    createGeojson: bool = True) -> Dict[str, Any]:
        logger.info("-"*50)
        logger.info("STARTING TREE DETECTION PIPELINE")
        logger.info("-"*50)
        
        # Create output folder and subfolders
        os.makedirs(outputFolder, exist_ok=True)
        os.makedirs(os.path.join(outputFolder, "json"), exist_ok=True)
        os.makedirs(os.path.join(outputFolder, "annotated"), exist_ok=True)
        
        pipelineSummary = {
            "input_path": inputPath,
            "model_type": self.modelType,
            "model_path": self.modelPath,
            "confidence": self.confidence,
            "output_folder": outputFolder,
            "results": []
        }
        
        try:
            if os.path.isfile(inputPath):
                # process single file
                if inputPath.lower().endswith(('.tif', '.tiff')):
                    results, tilesFolder = self.processLargeGeotiff(
                        geotiffPath=inputPath,
                        outputFolder=outputFolder,
                        tileSize=tileSize,
                        saveImages=saveImages,
                        saveJson=saveJson
                    )
                    pipelineSummary["processing_mode"] = "large_geotiff"
                    pipelineSummary["tiles_folder"] = tilesFolder
                else:
                    result = self.detectSingleImage(
                        imagePath=inputPath,
                        outputFolder=outputFolder,
                        saveImages=saveImages,
                        saveJson=saveJson
                    )
                    results = [result]
                    pipelineSummary["processing_mode"] = "single_image"
                    
            elif os.path.isdir(inputPath):
                # process folder
                results = self.processTilesFolder(
                    inputFolder=inputPath,
                    outputFolder=outputFolder,
                    saveImages=saveImages,
                    saveJson=saveJson
                )
                pipelineSummary["processing_mode"] = "tiles_folder"
                
            else:
                raise FileNotFoundError(f"Input path not found: {inputPath}")
            
            pipelineSummary["results"] = results
            
            # Calculate summary statistics
            successful_results = [r for r in results if "error" not in r]
            pipelineSummary["total_processed"] = len(results)
            pipelineSummary["successful_processed"] = len(successful_results)
            pipelineSummary["failed_processed"] = len(results) - len(successful_results)
            
            if successful_results:
                total_detections = sum(
                    r.get("trees_detected", r.get("num_detections", 0))
                    for r in successful_results
                )
                pipelineSummary["total_detections"] = total_detections
            
            # merge annotated tiles if requested and multiple tiles were processed
            if mergeTiles and len(successful_results) > 1 and saveImages:
                merge_success = self.mergeAnnotatedTiles(pipelineSummary, outputFolder)
                pipelineSummary["tile_merge_success"] = merge_success
            elif mergeTiles and not saveImages:
                logger.info("Tile merging skipped because image saving is disabled")
            elif mergeTiles and len(successful_results) <= 1:
                logger.info("Tile merging skipped because only one tile was processed")
            
            # convert JSON results to GeoJSON if requested and JSON files were saved
            if createGeojson and saveJson and successful_results:
                geojson_success = self.convertToGeojson(pipelineSummary, outputFolder)
                pipelineSummary["geojson_conversion_success"] = geojson_success
            elif createGeojson and not saveJson:
                logger.info("GeoJSON conversion skipped because JSON saving is disabled")
            elif createGeojson and not successful_results:
                logger.info("GeoJSON conversion skipped because no successful detections were made")
            
            # save pipeline summary
            summary_path = os.path.join(outputFolder, "json", "pipeline_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(pipelineSummary, f, indent=2)
            
            logger.info("="*50)
            logger.info("PIPELINE EXECUTION COMPLETED")
            logger.info(f"Total processed: {pipelineSummary['total_processed']}")
            logger.info(f"Successful: {pipelineSummary['successful_processed']}")
            logger.info(f"Failed: {pipelineSummary['failed_processed']}")
            if "total_detections" in pipelineSummary:
                logger.info(f"Total detections: {pipelineSummary['total_detections']}")
            if "total_geojson_features" in pipelineSummary:
                logger.info(f"GeoJSON features: {pipelineSummary['total_geojson_features']}")
            if "geojson_output" in pipelineSummary:
                logger.info(f"GeoJSON saved to: {pipelineSummary['geojson_output']}")
            if "merged_annotated_geotiff" in pipelineSummary:
                logger.info(f"Merged GeoTIFF saved to: {pipelineSummary['merged_annotated_geotiff']}")
            logger.info(f"Results saved to: {outputFolder}")
            logger.info("="*50)
            
            return pipelineSummary
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            pipelineSummary["error"] = str(e)
            return pipelineSummary

# if you pass this script in a prompt it will say that this is ai generated because
# all of those help messages written seriously good... :)
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input', '-i', required=True,
                       help='Input path (single GeoTIFF file or folder of tiles)')
    parser.add_argument('--model', '-m', required=True, choices=['yolo', 'detectree2'],
                       help='Model type to use for detection')
    parser.add_argument('--model-path', '-p', required=True,
                       help='Path to model file')
    parser.add_argument('--output', '-o', default='output',
                       help='Output folder (default: output)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='Confidence threshold for detections (default: 0.5)')
    parser.add_argument('--tile-size', '-t', type=int, default=640,
                       help='Tile size for splitting large images (default: 640)')
    parser.add_argument('--no-images', action='store_true',
                       help='Skip saving annotated images')
    parser.add_argument('--no-json', action='store_true',
                       help='Skip saving JSON detection data')
    parser.add_argument('--no-merge', action='store_true',
                       help='Skip merging tiles back to GeoTIFF')
    parser.add_argument('--no-geojson', action='store_true',
                       help='Skip converting JSON results to GeoJSON')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        pipeline = TreeDetectionPipeline(
            modelType=args.model,
            modelPath=args.model_path,
            confidence=args.confidence
        )
        
        summary = pipeline.runPipeline(
            inputPath=args.input,
            outputFolder=args.output,
            tileSize=args.tile_size,
            saveImages=not args.no_images,
            saveJson=not args.no_json,
            mergeTiles=not args.no_merge,
            createGeojson=not args.no_geojson
        )
        
        if "error" in summary:
            sys.exit(1)
        else:
            sys.exit(0)  # success
            
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()