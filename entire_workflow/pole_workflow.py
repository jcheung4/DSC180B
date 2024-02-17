import sys
import os
import shutil

import coordinate_traverse
import pole_detection

#coordinate_traverse.traverse_collect_images()

# Example of how to run script:
# python3 pole_workflow.py '32.8209644,-117.1861909' '32.8195283,-117.1861259'

loc1 = sys.argv[1]
loc2 = sys.argv[2]

temp_dir = 'temp_images'
temp_bb_dir = 'temp_bb_images'
temp_gif_dif = 'temp_gif_images'


if __name__== "__main__":
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
        
    if not os.path.exists(temp_bb_dir):
        os.mkdir(temp_bb_dir)
    
    if not os.path.exists(temp_gif_dif):
        os.mkdir(temp_gif_dif)
    coors = coordinate_traverse.traverse_collect_images(loc1, loc2, temp_dir)
    coordinate_traverse.traverse_straight(loc1 = loc1, loc2 = loc2, coors=coors, dir=temp_gif_dif)
    coordinate_traverse.gif_gen(dir=temp_gif_dif,filename='sample_traverse',duration = 0.1)

    print("FINISHED GETTING IMAGES")
    pole_detection.run_detection(loc1, loc2)
    
    
    shutil.rmtree('__pycache__')