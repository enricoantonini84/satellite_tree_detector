import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import glob
import rasterio
from rasterio.merge import merge
import numpy as np

#parse tile filename to extract row, col, and tile size information
def parseTileFilename(filename):
    #pattern for files created by split_geotiff_tiles.py and annotated versions
    patterns = [
        r'(.+)_tile_(\d{3})_(\d{3})_(\d+)x(\d+)\.tif',  # original pattern
        r'(.+)_tile_(\d{3})_(\d{3})_(\d+)x(\d+)_annotated\.tif',  # detectree2 annotated
        r'(.+)_tile_(\d{3})_(\d{3})_(\d+)x(\d+)_result\.tif'  # yolo result
    ]
    
    filename_base = os.path.basename(filename)
    
    for pattern in patterns:
        match = re.match(pattern, filename_base)
        if match:
            basename, row, col, width, height = match.groups()
            return {
                'basename': basename,
                'row': int(row),
                'col': int(col),
                'tileSize': int(width),
                'width': int(width),
                'height': int(height)
            }
    return None

#find all geotiff tiles in a folder that match the expected naming pattern
def findTilesInFolder(folderPath, basenameFilter=None):
    tilePatterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    allFiles = []
    
    for pattern in tilePatterns:
        allFiles.extend(glob.glob(os.path.join(folderPath, pattern)))
    
    #filter files that match tile naming pattern
    tiles = []
    for filePath in allFiles:
        tileInfo = parseTileFilename(filePath)
        if tileInfo:
            if basenameFilter is None or tileInfo['basename'] == basenameFilter:
                tiles.append(filePath)
    
    return sorted(tiles)

#analyze tile files to determine grid structure and properties
def getTileGridInfo(tileFiles):
    if not tileFiles:
        raise ValueError("No tile files provided")
    
    #parse all tile filenames
    tileInfos = []
    for tileFile in tileFiles:
        info = parseTileFilename(tileFile)
        if info:
            info['path'] = tileFile
            tileInfos.append(info)
    
    if not tileInfos:
        raise ValueError("No valid tile files found")
    
    #determine grid bounds
    rows = [info['row'] for info in tileInfos]
    cols = [info['col'] for info in tileInfos]
    
    minRow, maxRow = min(rows), max(rows)
    minCol, maxCol = min(cols), max(cols)
    
    #check tile sizes are consistent
    tileSizes = list(set(info['tileSize'] for info in tileInfos))
    if len(tileSizes) > 1:
        print(f"Warning: Multiple tile sizes found: {tileSizes}. Using first one: {tileSizes[0]}")
    
    tileSize = tileSizes[0]
    
    #get basename (should be consistent)
    basenames = list(set(info['basename'] for info in tileInfos))
    if len(basenames) > 1:
        print(f"Warning: Multiple basenames found: {basenames}. Using first one: {basenames[0]}")
    
    basename = basenames[0]
    
    gridInfo = {
        'basename': basename,
        'tileSize': tileSize,
        'minRow': minRow,
        'maxRow': maxRow,
        'minCol': minCol,
        'maxCol': maxCol,
        'gridRows': maxRow - minRow + 1,
        'gridCols': maxCol - minCol + 1,
        'totalTilesExpected': (maxRow - minRow + 1) * (maxCol - minCol + 1),
        'tilesFound': len(tileInfos),
        'tileInfos': {(info['row'], info['col']): info for info in tileInfos}
    }
    
    print(f"Grid analysis: {gridInfo['gridRows']}x{gridInfo['gridCols']} grid, "
          f"{gridInfo['tilesFound']}/{gridInfo['totalTilesExpected']} tiles found")
    
    return gridInfo

#merge geotiff tiles back into a single geotiff file
def mergeGeotiffTiles(tileFiles, outputPath, nodata=None):
    try:
        print(f"Merging {len(tileFiles)} tiles into {outputPath}")
        
        #get grid information
        gridInfo = getTileGridInfo(tileFiles)
        
        #open all tile datasets
        tileDatasets = []
        for tileFile in tileFiles:
            try:
                dataset = rasterio.open(tileFile)
                tileDatasets.append(dataset)
            except Exception as e:
                print(f"Warning: Could not open tile {tileFile}: {e}")
                continue
        
        if not tileDatasets:
            raise ValueError("No valid tile datasets could be opened")
        
        print(f"Successfully opened {len(tileDatasets)} tile datasets")
        
        #use rasterio.merge to combine tiles
        mergedData, mergedTransform = merge(tileDatasets, nodata=nodata)
        
        #get metadata from first tile (they should all be consistent)
        firstTile = tileDatasets[0]
        mergedProfile = firstTile.profile.copy()
        
        #update profile for merged data
        mergedProfile.update({
            'height': mergedData.shape[1],
            'width': mergedData.shape[2],
            'transform': mergedTransform,
            'compress': 'lzw'
        })
        
        if nodata is not None:
            mergedProfile['nodata'] = nodata
        
        #write merged geotiff
        # Remove PIXELTYPE=SIGNEDBYTE from profile to avoid warning in GDAL 3.7+
        if 'PIXELTYPE' in mergedProfile and mergedProfile['PIXELTYPE'] == 'SIGNEDBYTE':
            mergedProfile = mergedProfile.copy()
            del mergedProfile['PIXELTYPE']
        with rasterio.open(outputPath, 'w', **mergedProfile) as dst:
            dst.write(mergedData)
        
        #close all tile datasets
        for dataset in tileDatasets:
            dataset.close()
        
        #verify output file
        with rasterio.open(outputPath) as mergedFile:
            print(f"Merged GeoTIFF created successfully:")
            print(f"  Size: {mergedFile.width}x{mergedFile.height}")
            print(f"  Bounds: {mergedFile.bounds}")
            print(f"  CRS: {mergedFile.crs}")
            print(f"  Bands: {mergedFile.count}")
        
        return True
        
    except Exception as e:
        print(f"Error merging tiles: {str(e)}")
        return False

#find and merge all compatible tiles from a folder
def mergeTilesFromFolder(folderPath, outputPath, basenameFilter=None, nodata=None):
    if not os.path.exists(folderPath):
        print(f"Error: Folder not found: {folderPath}")
        return False
    
    #find tiles
    tileFiles = findTilesInFolder(folderPath, basenameFilter)
    
    if not tileFiles:
        print(f"Error: No compatible tiles found in {folderPath}")
        return False
    
    print(f"Found {len(tileFiles)} tiles to merge")
    
    #create output directory if needed
    os.makedirs(os.path.dirname(outputPath), exist_ok=True)
    
    return mergeGeotiffTiles(tileFiles, outputPath, nodata)

def main():
    parser = argparse.ArgumentParser()
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', '-i', help='Input folder containing tiles')
    group.add_argument('--tiles', nargs='+', help='List of specific tile files to merge')
    
    parser.add_argument('--output', '-o', required=True, help='Output path for merged GeoTIFF')
    parser.add_argument('--basename', '-b', help='Filter tiles by basename')
    parser.add_argument('--nodata', type=float, help='NoData value for output')
    
    args = parser.parse_args()
    
    try:
        success = False
        
        if args.input:
            success = mergeTilesFromFolder(
                folderPath=args.input,
                outputPath=args.output,
                basenameFilter=args.basename,
                nodata=args.nodata
            )
        else:
            success = mergeGeotiffTiles(
                tileFiles=args.tiles,
                outputPath=args.output,
                nodata=args.nodata
            )
        
        if success:
            print(f"Successfully merged tiles to: {args.output}")
        else:
            print("Failed to merge tiles")
            
    except Exception as e:
        print(f"Merge operation failed: {str(e)}")

if __name__ == "__main__":
    main()