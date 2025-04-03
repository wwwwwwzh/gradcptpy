import numpy as np
import random
import scipy.io as sio
import os

def load_scene_vectors():
    """
    Load and concatenate scene vectors from the three specific .mat files
    
    Returns:
    --------
    concatenated_scenes : list
        Concatenated scene vectors from all runs
    """
    # The exact filenames as provided
    filenames = [
        "id_008_run_1_01-Aug-2023_10_19_SceneVector_city_mnt_v1B_beh_fMRI.mat",
        "id_008_run_2_01-Aug-2023_10_33_SceneVector_city_mnt_v1B_beh_fMRI.mat",
        "id_008_run_3_01-Aug-2023_10_44_SceneVector_city_mnt_v1B_beh_fMRI.mat"
    ]
    
    concatenated_scenes = []
    
    # Load all three runs
    for filename in filenames:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Could not find file: {filename}")
        
        mat_data = sio.loadmat(filename)
        scene_vec = mat_data['scene_vec'].flatten()
        concatenated_scenes.extend(scene_vec.tolist())
    
    # Verify the length is 1800
    assert len(concatenated_scenes) == 1800, f"Concatenated scene vector length is {len(concatenated_scenes)}, expected 1800"
    
    return concatenated_scenes

def create_custom_stim_sequence(dom_stim, nondom_stim, N_dom, N_nondom): 
    """
    Creates the stimulus sequence for GradCPT
    
    Inputs:
    -------
    dom_stim : list
        Set of (10) dominant (city) stimuli; list of np.ndarrays
    nondom_stim : list
        Set of (10) non dominant (mountain) stimuli; list of np.arrays
    N_dom : int
        Number of dominant stimuli needed in the sequence
    N_nondom : int
        Number of nondominant stimuli needed in the sequence
        
    Returns:
    --------
    stim_set : list
        A list of images (np.ndarrays) with:
        - as many city images as N_dom
        - as many mountain images as N_nondom
        - in pseudo random order such that no two identical images follow each other
    conditions : list
        List indicating whether each stimulus is dominant ('dom') or non-dominant ('nondom')
    """
    # Verify inputs first
    assert N_dom + N_nondom == 1800, f"Sum of N_dom ({N_dom}) and N_nondom ({N_nondom}) must equal 1800"
    
    # First, create dom / nondom sequence
    conditions = ['dom'] * N_dom + ['nondom'] * N_nondom
    random.shuffle(conditions)
    stim_set = [None] * len(conditions)
    
    # Second, assign first image
    stim_set[0] = random.choice(dom_stim) if conditions[0] == 'dom' else random.choice(nondom_stim)
    
    # Finally, iterate and assign remainder of images
    for i in range(1, len(stim_set)):
        new_image = stim_set[i-1]  # Initialize with previous image to enter the while loop
        
        # Randomly choose new image as long as the chosen image is equal to the previous image
        while np.array_equal(new_image, stim_set[i-1]):
            if conditions[i] == 'dom':
                new_image = random.choice(dom_stim)
            else:
                new_image = random.choice(nondom_stim)
                
        # Assign the image only when it's not the previous image
        stim_set[i] = new_image
            
    return stim_set, conditions