#!/usr/bin/env python3

import datetime
import json
import os
import re
import fnmatch
from shutil import copyfile
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
import multiprocessing as mp
import random
import math

SHOULD_COPY = True

IMAGE_DIR_NAME = "images"
ANNOTATION_DIR_NAME = "annotations"

SOURCE_DIR = "E:\\Data\\ROCO_9_Inch_Only_Features_256_Whole\\"
SOURCE_IMAGE_DIR = os.path.join(SOURCE_DIR, "images")
SOURCE_LABEL_DIR = os.path.join(SOURCE_DIR, "labels")

DATASET_NAME = "ROCOFootprints_9_Inch_Incl_Only_Feature_256_Whole_Single"

TRAIN_FOLDER_NAME = "train_stage"
VALIDATION_FOLDER_NAME = "val_stage"

IMAGE_FILE_EXTENSIONS = ['*.png']
ANNOTATION_FILE_EXTENSIONS = ['*.tif']

# TRAIN_SET_SIZE = 25000
# VALIDATION_SET_SIZE = 250

MAX_TRAIN_VALIDATION_SETS = 1

STARTING_TRAINING_SET_NUMBER = 1

NUM_THREADS = mp.cpu_count()

STATS = []

THREAD_LOCK = mp.Lock()

INFO = {
    "description": "ROCO Footprints Dataset - 9 inch - only fetures - 256x256 - whole dataset",
    # "url": "https://github.com/waspinator/pycococreator",
    "version": "0.0.1",
    "year": 2019,
    "contributor": "Calvin Echols",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

ROOT_DIR = "{}_{}".format(DATASET_NAME, INFO["year"])

LICENSES = [
    # {
    #     "id": 1,
    #     "name": "Attribution-NonCommercial-ShareAlike License",
    #     "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    # }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'footprint',
        'supercategory': 'building',
    },
]

def filter_for_jpeg(root, files, extensions):
    file_types = extensions#IMAGE_FILE_EXTENSIONS
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    
    return files

def filter_for_annotations(root, files, image_filename, extensions):
    file_types = extensions#ANNOTATION_FILE_EXTENSIONS
    file_types = '|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files

#copy to new locations with split training and validation sets.
def split_data_for_training():
    training_set_number = STARTING_TRAINING_SET_NUMBER - 1
    if not os.path.exists(ROOT_DIR):
        os.mkdir(ROOT_DIR)
    print("Walking {} to get the image chips for analysis.".format(SOURCE_IMAGE_DIR))
    image_file_walk = os.walk(SOURCE_IMAGE_DIR)
    filtered_image_files = []
    filtered_annotation_image_files_per_category = []
    print("Loading target image files")
    for image_root, _, image_files in image_file_walk:
        print("Filtering files from {} to find relevant files.".format(SOURCE_IMAGE_DIR))
        filtered_filenames = filter_for_jpeg(image_root, image_files, IMAGE_FILE_EXTENSIONS)        
        filtered_image_files.extend(filtered_filenames)
    number_of_images = len(filtered_image_files)
    # possible_number_of_train_val_sets = number_of_images % (TRAIN_SET_SIZE + VALIDATION_SET_SIZE)
    # number_of_train_val_sets = MAX_TRAIN_VALIDATION_SETS # if MAX_TRAIN_VALIDATION_SETS <= possible_number_of_train_val_sets else possible_number_of_train_val_sets
    categories = get_category_data()
    for category in categories:
        print("populating file data for annotations of category where id = {}".format(category["id"]))
        filtered_annotation_image_files_per_category.append({
            "id": category["id"],
            "name": category["name"],
            "files": get_annotation_files_by_categories(category)
        })
    number_of_samples = len(filtered_image_files)
    random.shuffle(filtered_image_files)
    per_subset_images = np.array_split(filtered_image_files, MAX_TRAIN_VALIDATION_SETS)
    print("Found {} number of images. Will split into {} number of training and validation sets.".format(number_of_images, MAX_TRAIN_VALIDATION_SETS))
    subsets = []
    subset_number = 1
    for subset in per_subset_images:
        subsets.append( {
            "annotations": filtered_annotation_image_files_per_category,
            "images": subset,
            "subset_number": subset_number
        })
        subset_number += 1
    # process_files(subsets[0])
    with mp.Pool(processes = NUM_THREADS) as p:
        p.map(process_files, subsets)
    

def process_files(subset):
    images = subset["images"]
    filtered_annotation_image_files_per_category = subset["annotations"]
    subset_number = subset["subset_number"]
    num_images = len(images)
    train_set_size = math.floor(num_images * 0.95)
    loop_index = 0
    training_set_number = subset_number
    training_set_dir_name = os.path.join(ROOT_DIR, "{}{}".format(TRAIN_FOLDER_NAME, training_set_number))
    validation_set_dir_name = os.path.join(ROOT_DIR, "{}{}".format(VALIDATION_FOLDER_NAME, training_set_number))    
    make_dirs(training_set_number, training_set_dir_name, validation_set_dir_name)
    while loop_index < num_images:
        random_target = images[loop_index]
        # filtered_image_files.remove(random_target)
        if loop_index < train_set_size:
            print("{} - Put the data in the current train set # {}.".format(loop_index, training_set_number))
            copy_image_and_annotation_files_to_directory(random_target, filtered_annotation_image_files_per_category, training_set_dir_name)
        else:
            print("{} - Put the data in the current validation set # {}.".format(loop_index, training_set_number))
            copy_image_and_annotation_files_to_directory(random_target, filtered_annotation_image_files_per_category, validation_set_dir_name)
        loop_index += 1

def make_dirs(training_set_number, training_set_dir_name, validation_set_dir_name):
    if not os.path.exists(training_set_dir_name):
        os.mkdir(training_set_dir_name)        
        os.mkdir(os.path.join(training_set_dir_name, IMAGE_DIR_NAME))
        os.mkdir(os.path.join(training_set_dir_name, ANNOTATION_DIR_NAME))    
    if not os.path.exists(validation_set_dir_name):
        os.mkdir(validation_set_dir_name)
        os.mkdir(os.path.join(validation_set_dir_name, IMAGE_DIR_NAME))
        os.mkdir(os.path.join(validation_set_dir_name, ANNOTATION_DIR_NAME))

def copy_image_and_annotation_files_to_directory(image_file, filtered_annotation_image_files_per_category, target_directory):
    file_name = os.path.split(image_file)[1]
    print("Finding annotations for {}".format(file_name))
    file_name_without_extension = os.path.splitext(file_name)[0]
    copyfile(image_file, os.path.join(target_directory, IMAGE_DIR_NAME, file_name))
    for category in filtered_annotation_image_files_per_category:
        id = category["id"]
        name = category["name"]
        target_annotation_file = next((x for x in category["files"] if file_name_without_extension in x), False)
        if target_annotation_file != False:
            target_annotation_file_name = os.path.split(target_annotation_file)[1]
            (target_annotation_file_name_without_extension, target_annotation_file_name_extension) =  os.path.splitext(target_annotation_file_name)
            # target_annotation_file_name_without_extension = os.path.splitext(target_annotation_file_name)[0]
            copyfile(target_annotation_file, os.path.join(target_directory, ANNOTATION_DIR_NAME, "{}_{}{}".format(target_annotation_file_name_without_extension, name, target_annotation_file_name_extension)))
            print ("Found a file that matches the target image chip. {}".format(target_annotation_file))
        else:
            print("No annotation found for {} where category id = {}".format(image_file, id))


def get_category_data():
    categories = []
    for x in CATEGORIES:
        file_types = ANNOTATION_FILE_EXTENSIONS
        id = x['id']
        categories.append({            
            "id": id,
            "name": x['name'],
            "supercategory": x['supercategory'],            
            "category_path": os.path.join(SOURCE_LABEL_DIR, str(id)),
            "file_types": ANNOTATION_FILE_EXTENSIONS,
            "regex": re.compile(r'|'.join([fnmatch.translate(x) for x in file_types]))
        })
    print("Found the following categories: {}".format(categories))
    return categories

def get_annotation_files_by_categories(category):
    filtered_annoatation_image_files = []
    print("Walking {} to get the annotation chips for analysis.".format(category["category_path"]))
    annotation_image_file_walk = os.walk(category["category_path"])
    print("Loading target annotation image files for category id = {}".format(category["id"]))
    for image_root, _, image_files in annotation_image_file_walk:
        print("Filtering files from {} to find relevant files.".format(category["category_path"]))
        filtered_filenames = filter_for_jpeg(image_root, image_files, ANNOTATION_FILE_EXTENSIONS)        
        filtered_annoatation_image_files.extend(filtered_filenames)
    print("Found {} annotation image files for category id = {}".format(len(filtered_annoatation_image_files), category["id"]))
    return filtered_annoatation_image_files

def shapes_to_coco_processing(directory):
    images_added = 0
    annotations_added = 0
    images_failed = 0
    annoations_failed = 0
    images_with_no_annotations = 0
    images_with_annotations = 0

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    image_id = 1
    segmentation_id = 1
    image_dir = os.path.join(ROOT_DIR, directory, "images")
    annotation_dir = os.path.join(ROOT_DIR, directory, "annotations")
    print("Annotating data in {} and {}".format(image_dir, annotation_dir))
    for root, _, files in os.walk(image_dir):
        image_files = filter_for_jpeg(root, files, IMAGE_FILE_EXTENSIONS)

        # go through each image
        for image_filename in image_files:
            number_of_annotations = 0
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            #Try to open the image, and if it does not work log it and do not add it to the output.
            did_open = False
            try:
                test_open = Image.open(image_filename)
                if test_open is not None:
                    did_open = True
                    coco_output["images"].append(image_info)
                    images_added += 1
                    print("Image tested and appears valid; {}".format(image_filename))
            except:
                print("Failed to open image: {} Therefore we are skipping it in the subset of data.".format(image_filename))
                images_failed += 1
            
            if did_open:
                # filter for associated png annotations
                for root, _, files in os.walk(annotation_dir):
                    annotation_files = filter_for_annotations(root, files, image_filename, ANNOTATION_FILE_EXTENSIONS)

                    # go through each associated annotation
                    for annotation_filename in annotation_files:
                        print(annotation_filename)
                        class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]

                        category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}                        
                        try:
                            binary_mask = np.asarray(Image.open(annotation_filename)).astype(np.uint8)
                            
                            annotation_info = pycococreatortools.create_annotation_info(
                                segmentation_id, image_id, category_info, binary_mask,
                                image.size, tolerance=2)

                            if annotation_info is not None:
                                coco_output["annotations"].append(annotation_info)
                                annotations_added += 1
                                number_of_annotations += 1
                                print("Annotations found and appears valid: class_id: {}, annotation_filename: {}".format(class_id, annotation_filename))
                        except:
                            print("Failed to open: {}".format(annotation_filename))                             
                            coco_output["images"].remove(image_info)
                            images_added -= 1                                
                        else: #If no errors occurred                        
                            segmentation_id = segmentation_id + 1
                            image_id = image_id + 1
            if number_of_annotations == 0:
                images_with_no_annotations += 1
            else:
                images_with_annotations += 1


    with open('{}/instances_{}.json'.format(ROOT_DIR, directory), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)

    THREAD_LOCK.acquire()
    subset_stats = "Summary Statistics: Subset = directory was {}\nImages added = {}\nImages failed = {}\nannotations added = {}\nannotations failed = {}\nImages with annotations = {}\nImages with no annotations = {}\n\n\n".format(directory, images_added, images_failed, annotations_added, annoations_failed, images_with_annotations, images_with_no_annotations)
    print(subset_stats)
    STATS.append(subset_stats)
    THREAD_LOCK.release()

def main():
    print('Starting annotation generation')
    # filter for jpeg images
    dirs = os.listdir(ROOT_DIR)
    with mp.Pool(processes = NUM_THREADS) as p:
        p.map(shapes_to_coco_processing, dirs)
    # shapes_to_coco_processing(dirs[0])
    print("Finished annotating data.")
    for stats in STATS:
        print(stats)

if __name__ == "__main__":
    if SHOULD_COPY:
        split_data_for_training()
    main()
