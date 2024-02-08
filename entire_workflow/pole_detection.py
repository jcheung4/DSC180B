def run_detection(loc1, loc2):
    import torch, torchvision
    print(torch.__version__, torch.cuda.is_available())
    torch.set_grad_enabled(False)

    import torchvision.transforms as T
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    import psycopg2
    
    import os
    import json

    import pycocotools.coco as coco
    from pycocotools.coco import COCO
    import numpy as np
    import skimage.io as io
    import matplotlib.pyplot as plt
    import pylab
    pylab.rcParams['figure.figsize'] = (10.0, 8.0)

    from PIL import Image
    import glob

    import re

    # To sort images because it was doing left0, left1, left10, instead of left0, left1, left2
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(out_bbox, size):
        img_w, img_h = size
        b = box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def filter_bboxes_from_outputs(outputs,
                                threshold=0.7):

        # keep only predictions with confidence above threshold
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > threshold

        probas_to_keep = probas[keep]

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

        return probas_to_keep, bboxes_scaled

    def plot_finetuned_results(pil_img, prob=None, boxes=None, img_name=None):
        plt.figure(figsize=(16,10))
        plt.imshow(pil_img)
        ax = plt.gca()
        colors = COLORS * 100
        if prob is not None and boxes is not None:
            for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                            fill=False, color=c, linewidth=3))
                cl = p.argmax()
                text = f'{finetuned_classes[cl]}: {p[cl]:0.2f}'
                ax.text(xmin, ymin, text, fontsize=15,
                        bbox=dict(facecolor='yellow', alpha=0.5))
            plt.axis('off')
            
            save_path = os.path.join('temp_bb_images', img_name.split('/')[1])
            # Save the plot as an image
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            
            #plt.show()

    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
            [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


    # Parameters for Model
    num_classes = 2
    finetuned_classes = ['Metal', 'Wooden']

    # Loading Fine Tuned Model
    model = torch.hub.load('facebookresearch/detr',
                        'detr_resnet50',
                        pretrained=False,
                        num_classes=num_classes)

    checkpoint = torch.load('models/checkpoint_Q1.pth',
                            map_location='cpu')

    model.load_state_dict(checkpoint['model'],
                        strict=False)

    # Current parameters we used to determine if bounding box is valid (Currently only using area)
    min_area = 10000
    min_height = 250

    print("Finised Setup Before DB")
    # Setting up Database
    conn = psycopg2.connect(
        host = 'localhost',
        database='mydatabase',
        user = 'dsc180b',
        password = 'dsc180b'
    )

    cur = conn.cursor()

    cur.execute(
        '''
        CREATE TABLE mypoles (
            id SERIAL PRIMARY KEY,
            latitude DOUBLE PRECISION,
            longitude DOUBLE PRECISION,
            type VARCHAR(6)
        )
        '''
    )
    
    print("FINISHED CREATING TEMP DB")

    def run_worflow(my_image, my_model, img_name):
        # mean-std normalize the input image (batch-size: 1)
        img = transform(my_image).unsqueeze(0)

        # propagate through the model
        outputs = my_model(img)

        for threshold in [0.75]:

            probas_to_keep, bboxes_scaled = filter_bboxes_from_outputs(outputs,
                                                                    threshold=threshold)
            
            if probas_to_keep is not None and bboxes_scaled is not None:
                for p, (xmin, ymin, xmax, ymax)in zip(probas_to_keep, bboxes_scaled.tolist()):
                    cl = p.argmax()
                    
                    img_split = img_name.split('_')
                    latitude = img_split[2]
                    longitude = img_split[3][:-4]
                    pole_type = finetuned_classes[cl]
                    
                    cur_area = (xmax - xmin) * (ymax - ymin)
                    cur_height = ymax - ymin
                    
                    if cur_area > min_area:
                    #if height > min_height:
                        insert_query = f" \
                        INSERT INTO mypoles (latitude, longitude, type) VALUES \
                        ({latitude}, {longitude}, '{pole_type}') \
                        "
                        
                        cur.execute(insert_query)
                        
                        print(True)

            plot_finetuned_results(my_image,
                                probas_to_keep,
                                bboxes_scaled,
                                img_name)
        
    image_paths = glob.glob('temp_images/*.jpg')
    left_images = []
    right_images = []

    for image in image_paths:
        if 'left' in image:
            left_images.append(image)
        else:
            right_images.append(image)
            
    left_images = sorted(left_images, key=natural_sort_key)
    right_images = sorted(right_images, key=natural_sort_key)

    for image in left_images:
        print(image)    
        im = Image.open(image)

        run_worflow(im, model, image)
        
    for image in right_images:
        print(image)    
        im = Image.open(image)

        run_worflow(im, model, image)

    # Extract longitudes and latitudes from input strings
    latitude1, longitude1 = loc1.split(',')
    latitude2, longitude2 = loc2.split(',')
    
    cur.execute(
        '''
        SELECT * FROM mypoles;
        '''
    )

    # Fetch all rows from the result set
    rows = cur.fetchall()

    # Print the result
    for row in rows:
        print(row)
        
    my_poles_count = {'Wooden': 0, 'Metal': 0}

    for row in rows:
        if row[-1] == 'Wooden':
            my_poles_count['Wooden'] += 1
        else:
            my_poles_count['Metal'] += 1
        
    print("Our Count: ")
    print(my_poles_count)

    cur.execute(
        f'''
        SELECT *
        FROM poles
        WHERE
            -- Haversine formula for distance calculation
            6371000 * 2 * ASIN(
                SQRT(
                    POWER(SIN(RADIANS((latitude - {latitude1}) / 2)), 2) +
                    COS(RADIANS({latitude1})) * COS(RADIANS(latitude)) *
                    POWER(SIN(RADIANS((longitude - {longitude1}) / 2)), 2)
                )
            ) <= 100 -- distance in meters
            OR
            6371000 * 2 * ASIN(
                SQRT(
                    POWER(SIN(RADIANS((latitude - {latitude2}) / 2)), 2) +
                    COS(RADIANS({latitude2})) * COS(RADIANS(latitude)) *
                    POWER(SIN(RADIANS((longitude - {longitude2}) / 2)), 2)
                )
            ) <= 100; -- distance in meters

        '''
    )
    
    # Fetch all rows from the result set
    rows = cur.fetchall()

    # Print the result
    for row in rows:
        print(row)
        
    db_poles_count = {'Wooden': 0, 'Metal': 0}

    for row in rows:
        if row[-1] == 'Wooden':
            db_poles_count['Wooden'] += 1
        else:
            db_poles_count['Metal'] += 1
        
    print("Dummy DB Count:")
    print(db_poles_count)
    
    cur.execute(
    '''
    DROP TABLE mypoles;
    '''
    )
    
    with open('results.txt', 'w') as file:
        file.write("Our Pole Count:\n")
        json.dump(my_poles_count, file, indent=4)
        
        # Add a newline for separation
        file.write("\n\n")
        
        # Write "Their Count:" and the second dictionary
        file.write("Dummy DB Pole Count:\n")
        json.dump(db_poles_count, file, indent=4)