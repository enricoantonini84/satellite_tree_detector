import os
import re
import math
import argparse
from PIL import Image
from typing import Tuple, Dict
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import numpy as np

#todo: tile size should be passed as param?
TILE_SIZE = 512

#filename parser, for file that has lat long and zoom
def tileFilenameParser(filename: str) -> Tuple[float, float, int]:
    pattern = r'tile_(-?\d+\.?\d*)_(-?\d+\.?\d*)_z(\d+)\.(jpg|png)'
    match = re.match(pattern, filename)
    if match:
        lat = float(match.group(1))
        lng = float(match.group(2))
        zoom = int(match.group(3))
        return lat, lng, zoom
    else:
        raise ValueError(f"Cannot parse filename: {filename}")

#lat/lon to slippy format (web mercator) conversion
def deg2num(latDeg: float, lonDeg: float, zoom: int) -> Tuple[int, int]:
    lat_rad = math.radians(latDeg)
    n = 2.0 ** zoom
    x = int((lonDeg + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (x, y)

#bounds calculation for geotiff creation with rasterio
def boundsCalculator(tileCoords: Dict[Tuple[int, int], str], 
                     zoom: int) -> Tuple[float, float, float, float]:
    
    xCoords = [x for x, y in tileCoords.keys()]
    yCoords = [y for x, y in tileCoords.keys()]
    xMin, xMax = min(xCoords), max(xCoords)
    yMin, yMax = min(yCoords), max(yCoords)
    
    n = 2.0 ** zoom

    radNorth = math.atan(math.sinh(math.pi * (1 - 2 * yMin / n)))
    radSouth = math.atan(math.sinh(math.pi * (1 - 2 * (yMax + 1) / n)))
    
    west = xMin / n * 360.0 - 180.0 #west bound
    east = (xMax + 1) / n * 360.0 - 180.0 #east bound
    north = math.degrees(radNorth) #north bound
    south = math.degrees(radSouth) #south bound
    
    return west, south, east, north

#save pil image as GeoTIFF, finally!!
def save_as_geotiff(image: Image.Image, 
                    bounds: Tuple[float, float, float, float],
                    outputPath: str) -> bool:
    try:
        west, south, east, north = bounds
        height, width = image.size[1], image.size[0]
        
        transform = from_bounds(west, south, east, north, width, height)
        
        # numpy transformation
        imageArray = np.array(image)
        
        # RGB image, bands must be changed to be compatbile with rasterio (bands, height, width)
        count = imageArray.shape[2]
        imageArray = np.transpose(imageArray, (2, 0, 1))
        
        # call rasterio, write geotiff
        with rasterio.open(
            outputPath,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=count,
            dtype=imageArray.dtype,
            crs=CRS.from_epsg(4326),  # WGS84
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(imageArray)
            
        print(f"GeoTIFF saved: {outputPath}")
        return True
        
    except Exception as e:
        print(f"Error saving GeoTIFF: {e}")
        return False
    
# merge tiles into a single GeoTIFF
def merge_tiles(tilesFolder: str, outputPath: str = "merged_map.jpg"):
    
    tiles = [f for f in os.listdir(tilesFolder) 
                  if f.endswith(('.jpg', '.png')) and f.startswith('tile_')]
    
    if not tiles:
        print(f"No tile files found in {tilesFolder}")
        return
    
    print(f"Found {len(tiles)} tiles to merge")
    
    # parse all tile coordinates
    tilesCoords = {}
    zoom_level = None
    
    for filename in tiles:
        try:
            lat, lng, zoom = tileFilenameParser(filename)
            if zoom_level is None:
                zoom_level = zoom
            elif zoom_level != zoom:
                print(f"Warning: Mixed zoom levels found.")
                return
            
            # convert to slippy tile coordinates
            x, y = deg2num(lat, lng, zoom)
            tilesCoords[(x, y)] = os.path.join(tilesFolder, filename)
            
        except ValueError as e:
            print(f"Skipping file {filename}: {e}")
    
    if not tilesCoords:
        print("No valid tiles found")
        return
    
    # find all tiles x and y
    xCoords = [x for x, y in tilesCoords.keys()]
    yCoords = [y for x, y in tilesCoords.keys()]
    
    # calculate min and max of all tiles
    min_x, max_x = min(xCoords), max(xCoords)
    min_y, max_y = min(yCoords), max(yCoords)
    
    grid_width = max_x - min_x + 1
    grid_height = max_y - min_y + 1
    
    print(f"Grid: {grid_width} x {grid_height} tiles")
    print(f"Output size: {grid_width * TILE_SIZE} x {grid_height * TILE_SIZE} pixels")
    
    outputImage = Image.new('RGB', (grid_width * TILE_SIZE, grid_height * TILE_SIZE))
    
    # Place tiles in correct positions
    tiles_placed = 0
    for (x, y), filepath in tilesCoords.items():
        try:
            # Calculate position in output image
            pixel_x = (x - min_x) * TILE_SIZE
            pixel_y = (y - min_y) * TILE_SIZE
            
            # Open and paste tile
            tile_img = Image.open(filepath)
            outputImage.paste(tile_img, (pixel_x, pixel_y))
            tiles_placed += 1
            
            if tiles_placed % 50 == 0:
                print(f"Placed {tiles_placed}/{len(tilesCoords)} tiles")
                
        except Exception as e:
            print(f"Error processing tile {filepath}: {e}")

    # Calculate bounds from slippy coordinates
    bounds = boundsCalculator(tilesCoords, zoom_level)
    
    # Handle None output_path
    if outputPath is None:
        outputPath = "merged_map.jpg"
        
    baseName = os.path.splitext(outputPath)[0]
    geotiffPath = f"{baseName}_georeferenced.tif"
        
    success = save_as_geotiff(outputImage, bounds, geotiffPath)
    if success:
        print(f"GeoTIFF saved to: {geotiffPath}")
    else:
        print("Failed to generate GeoTIFF")
    
    return outputImage, tilesCoords

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tiles_directory",
        nargs="?",
        help="Directory containing tile images"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file name (default: merged_map.jpg)"
    )
    
    args = parser.parse_args()
    tilesFolder = args.tiles_directory
    outputFile = args.output
    
    if not tilesFolder or not os.path.exists(tilesFolder):
        print(f"Directory {tilesFolder} not found!")
        print("Available directories:")
        for item in os.listdir('.'):
            if os.path.isdir(item):
                print(f"  - {item}")
        return
    
    merge_tiles(tilesFolder, outputFile)

if __name__ == "__main__":
    main()