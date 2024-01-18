import images_individual as ii

OUTPATH = '../../data/images/'

if __name__== "__main__":
    ii.images_kevin('../../data/kevin_structures.json', OUTPATH)
    ii.images_sunny('../../data/sunny_structure_coordinates.json', OUTPATH)
    ii.images_jonathan('../../data/jonathan_structures.json', OUTPATH)
    ii.images_derek()