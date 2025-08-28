import os
import math
from typing import Tuple, List
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import numpy as np
import argparse
import glob

#convert tile pixel coordinates to map coordinates using the dataset's affine transform.
#the resulting coordinates are in the units of the dataset's CRS (e.g., degrees for EPSG:4326).
def tileBoundsCalculator(srcTransf, window: Window) -> Tuple[float, float, float, float]:
    left, top = srcTransf * (window.col_off, window.row_off)
    right, bottom = srcTransf * (window.col_off + window.width, window.row_off + window.height)
    
    return left, bottom, right, top

#geotiff splitter
def splitGeotiffInTiles(inputPath: str, 
                        outputFolder: str = "tiles_640", 
                        tileSize: int = 640) -> List[str]:
    
    os.makedirs(outputFolder, exist_ok=True)
    
    outputFiles = []
    baseName = os.path.splitext(os.path.basename(inputPath))[0]
    
    try:
        with rasterio.open(inputPath) as src:
            
            # number of tiles needed
            cols = math.ceil(src.width / tileSize)
            rows = math.ceil(src.height / tileSize)
            
            print(f"Generationg {cols}x{rows} = {cols * rows} tiles of {tileSize}x{tileSize} pixels")
            
            tileCounter = 0
            
            for row in range(rows):
                for col in range(cols):
                    cOffset = col * tileSize
                    rOffset = row * tileSize
                    
                    # check bounds
                    realWidth = min(tileSize, src.width - cOffset)
                    realHeight = min(tileSize, src.height - rOffset)
                    
                    if realWidth <= 0 or realHeight <= 0:
                        continue
                    
                    window = Window(cOffset, rOffset, realWidth, realHeight)
                    
                    # geo bounds calculation
                    tBounds = tileBoundsCalculator(src.transform, window)
                    
                    tTransform = from_bounds(
                        tBounds[0], tBounds[1], 
                        tBounds[2], tBounds[3], 
                        realWidth, realHeight
                    )
                    
                    # read data from source
                    tData = src.read(window=window)
                    
                    oFilename = f"{baseName}_tile_{row:03d}_{col:03d}_{tileSize}x{tileSize}.tif"
                    oPath = os.path.join(outputFolder, oFilename)
                    
                    # write GeoTIFF
                    with rasterio.open(
                        oPath,
                        'w',
                        driver='GTiff',
                        height=realHeight,
                        width=realWidth,
                        count=src.count,
                        dtype=src.dtypes[0],
                        crs=src.crs,
                        transform=tTransform,
                        compress='lzw',
                        nodata=src.nodata
                    ) as dst:
                        dst.write(tData)
                    
                    outputFiles.append(oPath)
                    tileCounter += 1
                    
            print(f"Created {tileCounter} tiles in {outputFolder}")
            
    except Exception as e:
        print(f"Error processing {inputPath}: {e}")
        return []
    
    return outputFiles

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input GeoTIFF file')
    parser.add_argument('-o', '--output', default='tiles_640', 
                       help='Output directory (default: tiles_640)')
    parser.add_argument('-s', '--size', type=int, default=640, 
                       help='Tile size in pixels (default: 640)')
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        splitGeotiffInTiles(args.input, args.output, args.size)
    else:
        print(f"Error: {args.input} is not a valid file")

if __name__ == "__main__":
    main()