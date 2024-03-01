import sys
import os
import shutil
import logging

# Clears out results.txt for demo purposes
with open('static/results.txt', 'w') as file:
    file.write('\nWaiting for results...')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('entire_workflow/logfile.txt', mode='a'),
        logging.FileHandler('static/sample-log.txt', mode = 'w')
    ]
)
    
logging.info("Start of pole_workflow.py")

import coordinate_traverse
import pole_detection

# Example of how to run script:
# python3 entire_workflow/pole_workflow.py '32.8209644,-117.1861909' '32.8195283,-117.1861259'

loc1 = sys.argv[1]
loc2 = sys.argv[2]

logging.info(f"First Coordinate Pair: {loc1}")
logging.info(f"Second Coordinte Pair: {loc2}")

temp_dir = 'entire_workflow/temp_images'
temp_bb_dir = 'entire_workflow/temp_bb_images'
temp_gif_dif = 'entire_workflow/temp_gif_images'


if __name__== "__main__":
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
        
    if not os.path.exists(temp_bb_dir):
        os.mkdir(temp_bb_dir)
    
    if not os.path.exists(temp_gif_dif):
        os.mkdir(temp_gif_dif)
    coors = coordinate_traverse.traverse_collect_images(loc1, loc2, temp_dir)
    
    logging.info("Collecting Images")
    coordinate_traverse.traverse_straight(loc1 = loc1, loc2 = loc2, coors=coors, dir=temp_gif_dif)
    
    coordinate_traverse.gif_gen(dir=temp_gif_dif, output_dir = 'entire_workflow/', filename='sample_traverse',duration = 0.1)
    logging.info("Finishec Creating Gif")
    
    shutil.copy(
        'entire_workflow/sample_traverse.gif',
        'static/sample_traverse.gif'
    )

    logging.info("Finished Collecting Images")
    print("FINISHED GETTING IMAGES")
    pole_detection.run_detection(loc1, loc2)
    
    
    shutil.rmtree('entire_workflow/__pycache__')
    shutil.rmtree('entire_workflow/temp_images')
    shutil.rmtree('entire_workflow/temp_bb_images')
    shutil.rmtree('entire_workflow/temp_gif_images')
    
    logging.info("Finished pole_workflow.py")