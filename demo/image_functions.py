import imageio
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import random

def normalize_image(img):
    # Normalize from 0 255 to -1 1
    return (img / 255) * 2 - 1
    

def crop_image(img):
    # Create a circular mask
    height, width = img.shape
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height / 2, width / 2
    radius = min(center_x, center_y)
    circle_mask = ((y - center_y) ** 2 + (x - center_x) ** 2) <= radius ** 2
    # Apply mask
    img[~circle_mask] = 127.5
    # Image needs to be vertically flipped for some reason
    #return np.flipud(img)
    return img
    

def create_stim_sequence(dom_stim, nondom_stim, N_dom, N_nondom):
    # Inputs:
        # Set of (10) dominant (city) stimuli; list of np.ndarrays
        # Set of (10) non dominant (mountain) stimuli; list of np.arrays
        # Number of dominant stimuli needed in the sequence; int
        # Number of nondominant stimuli needed in the sequence; int
    # Returns:
        # A list of images (np.ndarrays) with: 
            # as many city images as N_dom
            # as many mountain images as N_nondom
            # in pseudo random order such that no two images follow each other
            
    # First, create dom / nondom sequence
    conditions = ['dom'] * N_dom + ['nondom'] * N_nondom
    random.shuffle(conditions)
    stim_set = [None] * len(conditions)
    
    # Second, assign first image 
    stim_set[0] = random.choice(dom_stim) if conditions[0] == 'dom' else random.choice(nondom_stim)
    
    # Finally, iterate and assign remainder of images
    for i in range(1, len(stim_set)):
        new_image = stim_set[i-1]
        # Randomly choose new image as long as the chosen image is equal to the previous image
        while np.array_equal(new_image, stim_set[i-1]):
            if conditions[i] == 'dom':
                new_image = random.choice(dom_stim)
            else:
                new_image = random.choice(nondom_stim)
        
        # Assign the image only when it's not the previous image
        stim_set[i] = new_image

    return (stim_set, conditions)
        
    
if __name__ == '__main__':

    city_files = glob('../scenes5/city/*.jpg')
    mountain_files = glob('../scenes5/mountain/*.jpg')

    cities = []
    mountains = []

    for city_file, mountain_file in zip(city_files, mountain_files):
        city = cv2.imread(city_file, cv2.IMREAD_GRAYSCALE)
        mountain = np.array(imageio.imread(mountain_file))

        city = crop_image(city)
        mountain = crop_image(mountain)

        cities.append(city)
        mountains.append(mountain)

    
    stim_set, conditions = create_stim_sequence(cities, mountains, 90, 10)
    shape = city.shape

    fourcc = cv2.VideoWriter_fourcc(*'H264')
	output_video = cv2.VideoWriter('demo_video.mp4', fourcc, 60, (256, 256),0)
    steps = round(60 * .8)

    for i in range(len(stim_set) - 1):
        transitions = np.linspace(stim_set[i], stim_set[i+1], steps)
        for transition in transitions:
            output_video.write(transition.astype(np.uint8))

    cv2.destroyAllWindows()
	output_video.release()


