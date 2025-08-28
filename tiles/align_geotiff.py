import os
import argparse
from typing import Tuple
import rasterio
from rasterio.transform import Affine
import numpy as np


#offset calculation
def offsetCalc(current: Tuple[float, float], 
               real: Tuple[float, float]) -> Tuple[float, float]:
    cLat, cLong = current #current lat long
    rLat, rLong = real #real lat long
    diffLat = rLat - cLat
    diffLong = rLong - cLong    
    return diffLat, diffLong #delta lat long

#alignment function
def align(inputPath: str, 
          outputPath: str, 
          cCoords: Tuple[float, float], 
          rCoords: Tuple[float, float]) -> bool:
    
    try:
        diffLat, diffLong = offsetCalc(cCoords, rCoords)
        print(f"Calculated offset - Lat: {diffLat:.8f}, Lng: {diffLong:.8f}")
        
        with rasterio.open(inputPath) as src:
            imageData = src.read()
            orTransform = src.transform

            # new transform with the offset applied
            # | a  b  c |
            # | d  e  f |
            # | 0  0  1 |
            # where:
            # a, e = pixel size (scale)
            # b, d = rotation (0 for north-up images)
            # c, f = translation (top-left corner coordinates)
            newTransform = Affine(
                orTransform.a,  # pixel width (unchanged)
                orTransform.b,  # rotation (unchanged)
                orTransform.c + diffLong,  # left coordinate + lng offset
                orTransform.d,  # rotation (unchanged)
                orTransform.e,  # pixel height (unchanged)
                orTransform.f + diffLat   # top coordinate + lat offset
            )
            
            with rasterio.open(
                outputPath,
                'w',
                driver='GTiff',
                height=src.height,
                width=src.width,
                count=src.count,
                dtype=src.dtypes[0],
                crs=src.crs,
                transform=newTransform, #offset
                compress='lzw'
            ) as dst:
                dst.write(imageData)

            with rasterio.open(outputPath) as alignedSource:
                print(f"New bounds: {alignedSource.bounds}")
            return True
            
    except Exception as e:
        print(f"Align error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input GeoTIFF file")
    parser.add_argument("-o", "--output", help="Output file name")
    parser.add_argument(
        "--c-lat", 
        type=float, 
        required=True, 
        help="Current latitude"
    )
    parser.add_argument(
        "--c-lon", 
        type=float, 
        required=True, 
        help="Current longitude"
    )
    parser.add_argument(
        "--r-lat", 
        type=float, 
        required=True, 
        help="Real latitude"
    )
    parser.add_argument(
        "--r-lon", 
        type=float, 
        required=True, 
        help="Real longitude"
    )
    
    args = parser.parse_args()
    
    if not args.output:
        baseName = os.path.splitext(args.input)[0]
        args.output = f"{baseName}_aligned.tif"
    
    currentCoords = (args.c_lat, args.c_lon)
    realCoords = (args.r_lat, args.r_lon)
    
    success = align(
        args.input, 
        args.output, 
        currentCoords, 
        realCoords
    )
    
    if success:
        print(f"GeoTIFF saved to: {args.output}")
    else:
        print("Failed to align GeoTIFF")


if __name__ == "__main__":
    main()