#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.1post4),
    on September 13, 2024, at 10:54
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from global_vars
import random
from datetime import datetime
import pickle
import imageio
from glob import glob
import math
import pandas as pd


# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.1post4'
expName = 'gradcptpy'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\dbrau\\Dropbox\\post_doc\\professional\\projects\\independent\\gradcptpy\\gradcptpy.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('exp')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('welcome_key_resp') is None:
        # initialise welcome_key_resp
        welcome_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='welcome_key_resp',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('block_start_key_resp') is None:
        # initialise block_start_key_resp
        block_start_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='block_start_key_resp',
        )
    if deviceManager.getDevice('es_gradcpt_key_resp') is None:
        # initialise es_gradcpt_key_resp
        es_gradcpt_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='es_gradcpt_key_resp',
        )
    if deviceManager.getDevice('block_rest_key_resp') is None:
        # initialise block_rest_key_resp
        block_rest_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='block_rest_key_resp',
        )
    if deviceManager.getDevice('debrief_resp') is None:
        # initialise debrief_resp
        debrief_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='debrief_resp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "estimate_frame_rate" ---
    # Run 'Begin Experiment' code from function_defs
    def write_data(d, block, task='gradcpt'):
        '''
        A function to write neatly formatted data to a neat_data directory
        # --- PARAMETERS --- #
        d (list of dict): The trial data stored as a list of dictionaries
        block (int): The block number
        task (str): The name of the task
        '''
        
        if not os.path.exists('neat_data'):
            os.mkdir('neat_data')
        subject = expInfo['participant']
        filename = 'neat_data/sub-{}_task-{}_block-{}.csv'.format(subject, task, block)
        pd.DataFrame(d).to_csv(filename, index=False)
        
    
        
    def create_stim_sequence(dom_stim, nondom_stim, N_dom, N_nondom):
        # Creates the stimulus sequence for GradCPT
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
            
    
        
    def save_gradcpt_data(rt, key):
        # Calling vars beyond the local scope of the function
        out = {
            'subject': expInfo['participant'],
            'total_runtime_mins': (datetime.now() - experiment_start_time).total_seconds() / 60,
            'dom_key': dom_key,
            'nondom_key': nondom_key,
            # Start trial counter at one
            'gradcpt_trial': gradcpt_trial+1,
            'frame_count': frame_count,
            'transition_step': transition_step,
            'coherence': transition_step / transition_steps,
            'condition': conditions[gradcpt_trial],
            'resp_key': key,
            'rt': rt
        }
        return out
        
    # --- FUNCTIONS FOR FORMATTING GRADCPT STIMULI --- #
    
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
        img[~circle_mask] = 0
        # Image needs to be vertically flipped for some reason
        return np.flipud(img)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    # Run 'Begin Experiment' code from global_vars
    # ~~ GRADCPT VARIABLES ~~ #
    # Number of GradCPT trials to perform
    N_trials = 80
    # Number of GradCPT blocks to perform
    N_blocks = 2
    # GradCPT stimuli transition timing
    transition_time = .8 # in seconds
    # Min and max trials before ES probe
    next_es_min = 30
    next_es_max = 45
    # What proportion of stimuli are city scenes (dominant)?
    prop_dom = .9
    
    
    # ~~ EXPERIENCE SAMPLING VARIABLES ~~ #
    # Time (s) between each experience sampling item
    es_isi_time = .5
    # Experience sampling items
    # Order will be shuffled except last item (confidence) will always be presented last
    es_items = [
        {'item_name': 'distractedness',
        'text': "Where was your attention during the previous trial?",
        'low_anchor': 'on-task',
        'high_anchor': 'off-task'
        },
        {'item_name': 'affect',
        'text': 'How positive or negative were you feeling?',
        'low_anchor': 'completely\nnegative',
        'high_anchor': 'completely\npositive'
        },
        {'item_name': 'disengage_difficulty',
        'text': 'How difficult was it to disengage from your thoughts?',
        'low_anchor': 'extremely\neasy',
        'high_anchor': 'extremely\ndifficult'
        },
        {'item_name': 'movement',
        'text': 'Were your thoughts freely moving?',
        'low_anchor': 'unmoving',
        'high_anchor': 'moving freely'
        },
        {'item_name': 'deliberate',
        'text': 'How intentional were your thoughts?',
        'low_anchor': 'completely\nunintentional',
        'high_anchor': 'completely\nintentional'
        },
        {'item_name': 'confidence',
        'text': 'How confident are you about your ratings for this trial?',
        'low_anchor': 'completely\nunconfident',
        'high_anchor': 'completely\nconfident'
        }
    ]
    
    # ~~ VARIABLES THAT SHOULDNT NEED TO BE CHANGED ~~ #
    # Random response mapping assignment
    dom_key = 'j' if random.choice([0, 1]) else 'f'
    nondom_key = 'j' if dom_key == 'f' else 'f'
    
    # Stimulus (image) location
    dom_files = glob('scenes5/city/*.jpg')
    nondom_files = glob('scenes5/mountain/*.jpg')
    
    # Total number of each stimulus type
    N_dom = round(N_trials * prop_dom)
    N_nondom = N_trials - N_dom
    
    # Inits
    gradcpt_trial = 0
    key_display = ''
    es_data = []
    gradcpt_data = []
    es_gradcpt_data = []
    refresh_rate = ''
    transition_steps = ''
    experiment_start_time = datetime.now()
    
    
    # ~~ GENERATE STIMULI ~~ # 
    # The actual sequence is generated before each GradCPT run so stimuli are randomized
    # see create_stim_sequence() in function_defs
    dom_stim = []
    nondom_stim = []
    
    # Import and process images (see function_defs code block)
    for dom_file, nondom_file in zip(dom_files, nondom_files):
        # Import
        dom_image = np.array(imageio.imread(dom_file))
        nondom_image = np.array(imageio.imread(nondom_file))
        # Normalize
        dom_image = normalize_image(dom_image)
        nondom_image = normalize_image(nondom_image)
        # Crop
        dom_image = crop_image(dom_image)
        nondom_image = crop_image(nondom_image)
        # Append
        dom_stim.append(dom_image)
        nondom_stim.append(nondom_image)
    
    
    # ~~ EXPERIENCE SAMPLING VARIABLES ~~ #
    es_trial = 0
    do_es = False
    es_text = ''
    es_low_anchor = ''
    es_high_anchor = ''
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    frame_rate_prompt = visual.TextStim(win=win, name='frame_rate_prompt',
        text='Trying to estimate frame rate....',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "welcome" ---
    welcome_key_resp = keyboard.Keyboard(deviceName='welcome_key_resp')
    welcome_text = visual.TextStim(win=win, name='welcome_text',
        text='This is the main part of the experiment. \n\nRemember, when answering questions about your thoughts, respond as accurately as possible and feel free to use the entire scale to indicate your response. And when doing the attention task, respond as quickly and accurately as possible.\n\nPress the space bar to continue reading.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "gradcpt_prep" ---
    gradcpt_prep_text = visual.TextStim(win=win, name='gradcpt_prep_text',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "block_start" ---
    black_start_text = visual.TextStim(win=win, name='black_start_text',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    block_start_key_resp = keyboard.Keyboard(deviceName='block_start_key_resp')
    
    # --- Initialize components for Routine "gradcpt_stim" ---
    gradcpt_image = visual.ImageStim(
        win=win,
        name='gradcpt_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "ES_gradcpt" ---
    es_text_component_gradcpt = visual.TextStim(win=win, name='es_text_component_gradcpt',
        text='',
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    es_gradcpt_slider = visual.Slider(win=win, name='es_gradcpt_slider',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=None, ticks=(0, 1), granularity=0.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    es_gradcpt_key_resp = keyboard.Keyboard(deviceName='es_gradcpt_key_resp')
    low_anchor_gradcpt = visual.TextStim(win=win, name='low_anchor_gradcpt',
        text='',
        font='Open Sans',
        pos=(-.5, -.15), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    es_gradcpt_mouse = event.Mouse(win=win)
    x, y = [None, None]
    es_gradcpt_mouse.mouseClock = core.Clock()
    high_anchor_gradcpt = visual.TextStim(win=win, name='high_anchor_gradcpt',
        text='',
        font='Open Sans',
        pos=(.5, -.15), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    enter_advance_gradcpt = visual.TextStim(win=win, name='enter_advance_gradcpt',
        text='Press "Enter" to advance.',
        font='Open Sans',
        pos=(0, -.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    
    # --- Initialize components for Routine "ES_isi_gradcpt" ---
    es_gradcpt_isi_text = visual.TextStim(win=win, name='es_gradcpt_isi_text',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "prepare" ---
    prepare_text = visual.TextStim(win=win, name='prepare_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "block_rest" ---
    block_rest_text = visual.TextStim(win=win, name='block_rest_text',
        text="That's the end of this phase of the experiment. Press the spacebar to continue.",
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    block_rest_key_resp = keyboard.Keyboard(deviceName='block_rest_key_resp')
    
    # --- Initialize components for Routine "debrief" ---
    debrief_text = visual.TextStim(win=win, name='debrief_text',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    debrief_resp = keyboard.Keyboard(deviceName='debrief_resp')
    
    # --- Initialize components for Routine "Thanks" ---
    thanks_text = visual.TextStim(win=win, name='thanks_text',
        text='Thank you for your participation!',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "estimate_frame_rate" ---
    # create an object to store info about Routine estimate_frame_rate
    estimate_frame_rate = data.Routine(
        name='estimate_frame_rate',
        components=[frame_rate_prompt],
    )
    estimate_frame_rate.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from global_vars
    win.mouseVisible = False
    # Run 'Begin Routine' code from estimate_frame_rate_code
    # Code for manually estimating the monitor refresh rate
    frame_rate_data = []
    start_routine = datetime.now()
    # store start times for estimate_frame_rate
    estimate_frame_rate.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    estimate_frame_rate.tStart = globalClock.getTime(format='float')
    estimate_frame_rate.status = STARTED
    thisExp.addData('estimate_frame_rate.started', estimate_frame_rate.tStart)
    estimate_frame_rate.maxDuration = None
    # keep track of which components have finished
    estimate_frame_rateComponents = estimate_frame_rate.components
    for thisComponent in estimate_frame_rate.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "estimate_frame_rate" ---
    estimate_frame_rate.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *frame_rate_prompt* updates
        
        # if frame_rate_prompt is starting this frame...
        if frame_rate_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            frame_rate_prompt.frameNStart = frameN  # exact frame index
            frame_rate_prompt.tStart = t  # local t and not account for scr refresh
            frame_rate_prompt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(frame_rate_prompt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'frame_rate_prompt.started')
            # update status
            frame_rate_prompt.status = STARTED
            frame_rate_prompt.setAutoDraw(True)
        
        # if frame_rate_prompt is active this frame...
        if frame_rate_prompt.status == STARTED:
            # update params
            pass
        
        # if frame_rate_prompt is stopping this frame...
        if frame_rate_prompt.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > frame_rate_prompt.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                frame_rate_prompt.tStop = t  # not accounting for scr refresh
                frame_rate_prompt.tStopRefresh = tThisFlipGlobal  # on global time
                frame_rate_prompt.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'frame_rate_prompt.stopped')
                # update status
                frame_rate_prompt.status = FINISHED
                frame_rate_prompt.setAutoDraw(False)
        # Run 'Each Frame' code from estimate_frame_rate_code
        frame_rate_data.append(math.floor((datetime.now() - start_routine).total_seconds()))
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            estimate_frame_rate.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in estimate_frame_rate.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "estimate_frame_rate" ---
    for thisComponent in estimate_frame_rate.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for estimate_frame_rate
    estimate_frame_rate.tStop = globalClock.getTime(format='float')
    estimate_frame_rate.tStopRefresh = tThisFlipGlobal
    thisExp.addData('estimate_frame_rate.stopped', estimate_frame_rate.tStop)
    # Run 'End Routine' code from estimate_frame_rate_code
    _, counts = np.unique(np.array(frame_rate_data), return_counts=True)
    refresh_rate = np.mean(counts)
    transition_steps = round(refresh_rate * transition_time)
    
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if estimate_frame_rate.maxDurationReached:
        routineTimer.addTime(-estimate_frame_rate.maxDuration)
    elif estimate_frame_rate.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "welcome" ---
    # create an object to store info about Routine welcome
    welcome = data.Routine(
        name='welcome',
        components=[welcome_key_resp, welcome_text],
    )
    welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for welcome_key_resp
    welcome_key_resp.keys = []
    welcome_key_resp.rt = []
    _welcome_key_resp_allKeys = []
    # Run 'Begin Routine' code from welcome_hide_mouse
    win.mouseVisible = False
    # store start times for welcome
    welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    welcome.tStart = globalClock.getTime(format='float')
    welcome.status = STARTED
    thisExp.addData('welcome.started', welcome.tStart)
    welcome.maxDuration = None
    # keep track of which components have finished
    welcomeComponents = welcome.components
    for thisComponent in welcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "welcome" ---
    welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welcome_key_resp* updates
        waitOnFlip = False
        
        # if welcome_key_resp is starting this frame...
        if welcome_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcome_key_resp.frameNStart = frameN  # exact frame index
            welcome_key_resp.tStart = t  # local t and not account for scr refresh
            welcome_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcome_key_resp.started')
            # update status
            welcome_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(welcome_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(welcome_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if welcome_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = welcome_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _welcome_key_resp_allKeys.extend(theseKeys)
            if len(_welcome_key_resp_allKeys):
                welcome_key_resp.keys = _welcome_key_resp_allKeys[-1].name  # just the last key pressed
                welcome_key_resp.rt = _welcome_key_resp_allKeys[-1].rt
                welcome_key_resp.duration = _welcome_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *welcome_text* updates
        
        # if welcome_text is starting this frame...
        if welcome_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcome_text.frameNStart = frameN  # exact frame index
            welcome_text.tStart = t  # local t and not account for scr refresh
            welcome_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcome_text.started')
            # update status
            welcome_text.status = STARTED
            welcome_text.setAutoDraw(True)
        
        # if welcome_text is active this frame...
        if welcome_text.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            welcome.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in welcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcome" ---
    for thisComponent in welcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for welcome
    welcome.tStop = globalClock.getTime(format='float')
    welcome.tStopRefresh = tThisFlipGlobal
    thisExp.addData('welcome.stopped', welcome.tStop)
    # check responses
    if welcome_key_resp.keys in ['', [], None]:  # No response was made
        welcome_key_resp.keys = None
    thisExp.addData('welcome_key_resp.keys',welcome_key_resp.keys)
    if welcome_key_resp.keys != None:  # we had a response
        thisExp.addData('welcome_key_resp.rt', welcome_key_resp.rt)
        thisExp.addData('welcome_key_resp.duration', welcome_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "gradcpt_prep" ---
    # create an object to store info about Routine gradcpt_prep
    gradcpt_prep = data.Routine(
        name='gradcpt_prep',
        components=[gradcpt_prep_text, key_resp],
    )
    gradcpt_prep.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from instruction_fill_code
    # Instruction fill 
    
    the_dom_key = 'the {} key'.format(dom_key)
    nondom = 'nothing'
    
    
    
    instruction_fill = ('Now you will perform the attention task. A series of '
    'pictures will gradually be presented on the screen one after another. In this '
    'phase of the experiment, you will press {} when you see a picture of a city '
    'scene, and you will press {} when you see a picture of a mountain scene. '
    'It is important to respond as quickly and accurately as possible and to keep '
    'both hands on the keyboard at all times.\n\nWhen you answer thought probes '
    'during this phase, remember to again be as honest as possible. Whether you '
    'indicate that your attention was or was not on task will not affect your participation. '
    '\n\nPress the space bar to see the response '
    'mappings one more time before beginning.'.format(the_dom_key, nondom))
    gradcpt_prep_text.setText(instruction_fill)
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for gradcpt_prep
    gradcpt_prep.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    gradcpt_prep.tStart = globalClock.getTime(format='float')
    gradcpt_prep.status = STARTED
    thisExp.addData('gradcpt_prep.started', gradcpt_prep.tStart)
    gradcpt_prep.maxDuration = None
    # keep track of which components have finished
    gradcpt_prepComponents = gradcpt_prep.components
    for thisComponent in gradcpt_prep.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "gradcpt_prep" ---
    gradcpt_prep.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *gradcpt_prep_text* updates
        
        # if gradcpt_prep_text is starting this frame...
        if gradcpt_prep_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            gradcpt_prep_text.frameNStart = frameN  # exact frame index
            gradcpt_prep_text.tStart = t  # local t and not account for scr refresh
            gradcpt_prep_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(gradcpt_prep_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'gradcpt_prep_text.started')
            # update status
            gradcpt_prep_text.status = STARTED
            gradcpt_prep_text.setAutoDraw(True)
        
        # if gradcpt_prep_text is active this frame...
        if gradcpt_prep_text.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            gradcpt_prep.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in gradcpt_prep.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "gradcpt_prep" ---
    for thisComponent in gradcpt_prep.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for gradcpt_prep
    gradcpt_prep.tStop = globalClock.getTime(format='float')
    gradcpt_prep.tStopRefresh = tThisFlipGlobal
    thisExp.addData('gradcpt_prep.stopped', gradcpt_prep.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "gradcpt_prep" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    blocks = data.TrialHandler2(
        name='blocks',
        nReps=N_blocks, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(blocks)  # add the loop to the experiment
    thisBlock = blocks.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock:
            globals()[paramName] = thisBlock[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisBlock in blocks:
        currentLoop = blocks
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
        if thisBlock != None:
            for paramName in thisBlock:
                globals()[paramName] = thisBlock[paramName]
        
        # --- Prepare to start Routine "block_start" ---
        # create an object to store info about Routine block_start
        block_start = data.Routine(
            name='block_start',
            components=[black_start_text, block_start_key_resp],
        )
        block_start.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from determine_block_type
        
        key_display = 'Key for city scenes: {}\n When you see a mountain scene, do not make a response.'.format(dom_key)
        
        
        
        black_start_text.setText(key_display + '\n\nPress the space bar to begin.')
        # create starting attributes for block_start_key_resp
        block_start_key_resp.keys = []
        block_start_key_resp.rt = []
        _block_start_key_resp_allKeys = []
        # store start times for block_start
        block_start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        block_start.tStart = globalClock.getTime(format='float')
        block_start.status = STARTED
        thisExp.addData('block_start.started', block_start.tStart)
        block_start.maxDuration = None
        # keep track of which components have finished
        block_startComponents = block_start.components
        for thisComponent in block_start.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "block_start" ---
        # if trial has changed, end Routine now
        if isinstance(blocks, data.TrialHandler2) and thisBlock.thisN != blocks.thisTrial.thisN:
            continueRoutine = False
        block_start.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *black_start_text* updates
            
            # if black_start_text is starting this frame...
            if black_start_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                black_start_text.frameNStart = frameN  # exact frame index
                black_start_text.tStart = t  # local t and not account for scr refresh
                black_start_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(black_start_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'black_start_text.started')
                # update status
                black_start_text.status = STARTED
                black_start_text.setAutoDraw(True)
            
            # if black_start_text is active this frame...
            if black_start_text.status == STARTED:
                # update params
                pass
            
            # *block_start_key_resp* updates
            waitOnFlip = False
            
            # if block_start_key_resp is starting this frame...
            if block_start_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                block_start_key_resp.frameNStart = frameN  # exact frame index
                block_start_key_resp.tStart = t  # local t and not account for scr refresh
                block_start_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(block_start_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'block_start_key_resp.started')
                # update status
                block_start_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(block_start_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(block_start_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if block_start_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = block_start_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _block_start_key_resp_allKeys.extend(theseKeys)
                if len(_block_start_key_resp_allKeys):
                    block_start_key_resp.keys = _block_start_key_resp_allKeys[-1].name  # just the last key pressed
                    block_start_key_resp.rt = _block_start_key_resp_allKeys[-1].rt
                    block_start_key_resp.duration = _block_start_key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                block_start.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in block_start.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "block_start" ---
        for thisComponent in block_start.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for block_start
        block_start.tStop = globalClock.getTime(format='float')
        block_start.tStopRefresh = tThisFlipGlobal
        thisExp.addData('block_start.stopped', block_start.tStop)
        # check responses
        if block_start_key_resp.keys in ['', [], None]:  # No response was made
            block_start_key_resp.keys = None
        blocks.addData('block_start_key_resp.keys',block_start_key_resp.keys)
        if block_start_key_resp.keys != None:  # we had a response
            blocks.addData('block_start_key_resp.rt', block_start_key_resp.rt)
            blocks.addData('block_start_key_resp.duration', block_start_key_resp.duration)
        # Run 'End Routine' code from inits
        # Init counters
        probe_count = 0
        frame_count = 0
        transition_step = 0
        next_es_trial = round(random.uniform(next_es_min, next_es_max))
        gradcpt_trial = 0
        es_trial = 0
        gradcpt_data = []
        es_gradcpt_data = []
        do_es = False
        
        # Init clock
        routine_start = datetime.now()
        
        # Init stim
        stim_set, conditions = create_stim_sequence(dom_stim, nondom_stim, N_dom, N_nondom)
        
        transition_current = np.linspace(np.zeros((256, 256)), # Grey
                                  stim_set[gradcpt_trial], # Image 1
                                  transition_steps)
        transition_next = np.linspace(stim_set[gradcpt_trial], # Image 1
                                  stim_set[gradcpt_trial+1], # Image 2
                                  transition_steps)
        
        
        trial_start_time = datetime.now()
        
        feedback = 'GradCPT trial: {}\nNext probe: {}'.format(gradcpt_trial, next_es_trial)
        
        event.clearEvents()
        
        # the Routine "block_start" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials = data.TrialHandler2(
            name='trials',
            nReps=N_trials * 2, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(trials)  # add the loop to the experiment
        thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrial in trials:
            currentLoop = trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            
            # --- Prepare to start Routine "gradcpt_stim" ---
            # create an object to store info about Routine gradcpt_stim
            gradcpt_stim = data.Routine(
                name='gradcpt_stim',
                components=[gradcpt_image],
            )
            gradcpt_stim.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from stim_code
            es_trial = 0
            pressed_key = False
            # Need this and *not* a keyboard component
            kb = keyboard.Keyboard()
            # store start times for gradcpt_stim
            gradcpt_stim.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            gradcpt_stim.tStart = globalClock.getTime(format='float')
            gradcpt_stim.status = STARTED
            thisExp.addData('gradcpt_stim.started', gradcpt_stim.tStart)
            gradcpt_stim.maxDuration = None
            # keep track of which components have finished
            gradcpt_stimComponents = gradcpt_stim.components
            for thisComponent in gradcpt_stim.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "gradcpt_stim" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            gradcpt_stim.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from stim_code
                # if end of trial
                if transition_step == transition_steps:
                    # If no key was pressed in that trial, log omission
                    if not pressed_key:
                        gradcpt_data.append(save_gradcpt_data(0, 0))
                    # Reset pressed key check
                    pressed_key = False
                    # Update trial count
                    gradcpt_trial += 1
                    transition_step = 0
                    trial_start_time = datetime.now()
                
                    # If it's after the last trial (trial is zero indexed)
                    # End routine and loop
                    if gradcpt_trial == N_trials:
                        trials.finished = True
                        continueRoutine = False
                        transition_current = transition_next
                        # Do one more experience sample
                        do_es = True
                        probe_count += 1
                    # If it's probe time
                    elif gradcpt_trial == next_es_trial:
                        # Have the next image after experience sampling fade in from grey
                        grey = np.zeros((256, 256))
                        image1 = stim_set[gradcpt_trial]
                        image2 = np.zeros((256, 256))
                        if gradcpt_trial + 1 <= len(stim_set):
                            image2 = stim_set[gradcpt_trial+1]
                            
                        image2 = stim_set[gradcpt_trial+1]
                        transition_current = np.linspace(grey, image1, transition_steps, endpoint=False)
                        transition_next = np.linspace(image1, image2, transition_steps, endpoint=False)
                        do_es = True
                        probe_count += 1
                        continueRoutine = False
                    else:
                        # Update stimulus set
                        image1 = stim_set[gradcpt_trial]
                        image2 = np.zeros((256, 256))
                        if gradcpt_trial + 1 <= len(stim_set):
                            image2 = stim_set[gradcpt_trial+1]
                
                        # Prepare for next trial
                        transition_current = transition_next
                        # Generate a transition array of shape (transition_steps, 256, 256)
                        # containing one transition image per each frame of the transition
                        transition_next = np.linspace(image1, image2, transition_steps, endpoint=False)
                
                
                
                # Each new frame within a trial, 
                # update the image to the corresponding image from the transition array
                img = transition_current[transition_step]
                
                frame_count += 1
                transition_step += 1
                
                # If a key was pressed
                keys = kb.getKeys(keyList = ['j', 'f'], waitRelease=False, clear=True)
                
                if keys:
                    # Assuming only one key press possible per frame
                    # (and grabbing the first key pressed if not)
                    rt = (datetime.now() - trial_start_time).total_seconds()
                    key = keys[0].name
                    # Keeps from writing out an omission
                    pressed_key = True
                    gradcpt_data.append(save_gradcpt_data(rt, key))
                    kb.clearEvents()
                    
                    
                
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                
                
                # *gradcpt_image* updates
                
                # if gradcpt_image is starting this frame...
                if gradcpt_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    gradcpt_image.frameNStart = frameN  # exact frame index
                    gradcpt_image.tStart = t  # local t and not account for scr refresh
                    gradcpt_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(gradcpt_image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'gradcpt_image.started')
                    # update status
                    gradcpt_image.status = STARTED
                    gradcpt_image.setAutoDraw(True)
                
                # if gradcpt_image is active this frame...
                if gradcpt_image.status == STARTED:
                    # update params
                    gradcpt_image.setImage(img, log=False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    gradcpt_stim.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in gradcpt_stim.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "gradcpt_stim" ---
            for thisComponent in gradcpt_stim.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for gradcpt_stim
            gradcpt_stim.tStop = globalClock.getTime(format='float')
            gradcpt_stim.tStopRefresh = tThisFlipGlobal
            thisExp.addData('gradcpt_stim.stopped', gradcpt_stim.tStop)
            # the Routine "gradcpt_stim" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            ES_loop = data.TrialHandler2(
                name='ES_loop',
                nReps=6.0, 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=[None], 
                seed=None, 
            )
            thisExp.addLoop(ES_loop)  # add the loop to the experiment
            thisES_loop = ES_loop.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisES_loop.rgb)
            if thisES_loop != None:
                for paramName in thisES_loop:
                    globals()[paramName] = thisES_loop[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            for thisES_loop in ES_loop:
                currentLoop = ES_loop
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisES_loop.rgb)
                if thisES_loop != None:
                    for paramName in thisES_loop:
                        globals()[paramName] = thisES_loop[paramName]
                
                # --- Prepare to start Routine "ES_gradcpt" ---
                # create an object to store info about Routine ES_gradcpt
                ES_gradcpt = data.Routine(
                    name='ES_gradcpt',
                    components=[es_text_component_gradcpt, es_gradcpt_slider, es_gradcpt_key_resp, low_anchor_gradcpt, es_gradcpt_mouse, high_anchor_gradcpt, enter_advance_gradcpt],
                )
                ES_gradcpt.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from ES_gradcpt_code
                if not do_es:
                    ES_loop.finished = True
                    continueRoutine=False
                else:
                    # Random mouse initialization
                    jitter = random.uniform(-.05, .05)
                    es_gradcpt_mouse.setPos(newPos = (jitter, -.4))
                    win.mouseVisible = True
                    item_onset = (datetime.now() - routine_start).total_seconds()
                    # Shuffle non-confidence questions
                    # (conf rating always last)
                    if not es_trial:
                        non_conf = es_items[:-1]
                        random.shuffle(non_conf)
                        cur_items = non_conf + [es_items[-1]]
                        next_es_trial = round(random.uniform(gradcpt_trial+30, gradcpt_trial+45))
                        
                    es_text = cur_items[es_trial]['text']
                    es_low_anchor = cur_items[es_trial]['low_anchor']
                    es_high_anchor = cur_items[es_trial]['high_anchor']
                
                es_text_component_gradcpt.setText(es_text)
                es_gradcpt_slider.reset()
                # create starting attributes for es_gradcpt_key_resp
                es_gradcpt_key_resp.keys = []
                es_gradcpt_key_resp.rt = []
                _es_gradcpt_key_resp_allKeys = []
                low_anchor_gradcpt.setText(es_low_anchor)
                # setup some python lists for storing info about the es_gradcpt_mouse
                gotValidClick = False  # until a click is received
                high_anchor_gradcpt.setText(es_high_anchor)
                # store start times for ES_gradcpt
                ES_gradcpt.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                ES_gradcpt.tStart = globalClock.getTime(format='float')
                ES_gradcpt.status = STARTED
                thisExp.addData('ES_gradcpt.started', ES_gradcpt.tStart)
                ES_gradcpt.maxDuration = None
                # keep track of which components have finished
                ES_gradcptComponents = ES_gradcpt.components
                for thisComponent in ES_gradcpt.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "ES_gradcpt" ---
                # if trial has changed, end Routine now
                if isinstance(ES_loop, data.TrialHandler2) and thisES_loop.thisN != ES_loop.thisTrial.thisN:
                    continueRoutine = False
                ES_gradcpt.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *es_text_component_gradcpt* updates
                    
                    # if es_text_component_gradcpt is starting this frame...
                    if es_text_component_gradcpt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        es_text_component_gradcpt.frameNStart = frameN  # exact frame index
                        es_text_component_gradcpt.tStart = t  # local t and not account for scr refresh
                        es_text_component_gradcpt.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(es_text_component_gradcpt, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'es_text_component_gradcpt.started')
                        # update status
                        es_text_component_gradcpt.status = STARTED
                        es_text_component_gradcpt.setAutoDraw(True)
                    
                    # if es_text_component_gradcpt is active this frame...
                    if es_text_component_gradcpt.status == STARTED:
                        # update params
                        pass
                    
                    # *es_gradcpt_slider* updates
                    
                    # if es_gradcpt_slider is starting this frame...
                    if es_gradcpt_slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        es_gradcpt_slider.frameNStart = frameN  # exact frame index
                        es_gradcpt_slider.tStart = t  # local t and not account for scr refresh
                        es_gradcpt_slider.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(es_gradcpt_slider, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'es_gradcpt_slider.started')
                        # update status
                        es_gradcpt_slider.status = STARTED
                        es_gradcpt_slider.setAutoDraw(True)
                    
                    # if es_gradcpt_slider is active this frame...
                    if es_gradcpt_slider.status == STARTED:
                        # update params
                        pass
                    
                    # *es_gradcpt_key_resp* updates
                    waitOnFlip = False
                    
                    # if es_gradcpt_key_resp is starting this frame...
                    if es_gradcpt_key_resp.status == NOT_STARTED and es_gradcpt_slider.rating:
                        # keep track of start time/frame for later
                        es_gradcpt_key_resp.frameNStart = frameN  # exact frame index
                        es_gradcpt_key_resp.tStart = t  # local t and not account for scr refresh
                        es_gradcpt_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(es_gradcpt_key_resp, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'es_gradcpt_key_resp.started')
                        # update status
                        es_gradcpt_key_resp.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(es_gradcpt_key_resp.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(es_gradcpt_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    if es_gradcpt_key_resp.status == STARTED and not waitOnFlip:
                        theseKeys = es_gradcpt_key_resp.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
                        _es_gradcpt_key_resp_allKeys.extend(theseKeys)
                        if len(_es_gradcpt_key_resp_allKeys):
                            es_gradcpt_key_resp.keys = _es_gradcpt_key_resp_allKeys[-1].name  # just the last key pressed
                            es_gradcpt_key_resp.rt = _es_gradcpt_key_resp_allKeys[-1].rt
                            es_gradcpt_key_resp.duration = _es_gradcpt_key_resp_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    
                    # *low_anchor_gradcpt* updates
                    
                    # if low_anchor_gradcpt is starting this frame...
                    if low_anchor_gradcpt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        low_anchor_gradcpt.frameNStart = frameN  # exact frame index
                        low_anchor_gradcpt.tStart = t  # local t and not account for scr refresh
                        low_anchor_gradcpt.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(low_anchor_gradcpt, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'low_anchor_gradcpt.started')
                        # update status
                        low_anchor_gradcpt.status = STARTED
                        low_anchor_gradcpt.setAutoDraw(True)
                    
                    # if low_anchor_gradcpt is active this frame...
                    if low_anchor_gradcpt.status == STARTED:
                        # update params
                        pass
                    # *es_gradcpt_mouse* updates
                    
                    # if es_gradcpt_mouse is starting this frame...
                    if es_gradcpt_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        es_gradcpt_mouse.frameNStart = frameN  # exact frame index
                        es_gradcpt_mouse.tStart = t  # local t and not account for scr refresh
                        es_gradcpt_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(es_gradcpt_mouse, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.addData('es_gradcpt_mouse.started', t)
                        # update status
                        es_gradcpt_mouse.status = STARTED
                        es_gradcpt_mouse.mouseClock.reset()
                        prevButtonState = es_gradcpt_mouse.getPressed()  # if button is down already this ISN'T a new click
                    
                    # *high_anchor_gradcpt* updates
                    
                    # if high_anchor_gradcpt is starting this frame...
                    if high_anchor_gradcpt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        high_anchor_gradcpt.frameNStart = frameN  # exact frame index
                        high_anchor_gradcpt.tStart = t  # local t and not account for scr refresh
                        high_anchor_gradcpt.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(high_anchor_gradcpt, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'high_anchor_gradcpt.started')
                        # update status
                        high_anchor_gradcpt.status = STARTED
                        high_anchor_gradcpt.setAutoDraw(True)
                    
                    # if high_anchor_gradcpt is active this frame...
                    if high_anchor_gradcpt.status == STARTED:
                        # update params
                        pass
                    
                    # *enter_advance_gradcpt* updates
                    
                    # if enter_advance_gradcpt is starting this frame...
                    if enter_advance_gradcpt.status == NOT_STARTED and es_gradcpt_slider.rating:
                        # keep track of start time/frame for later
                        enter_advance_gradcpt.frameNStart = frameN  # exact frame index
                        enter_advance_gradcpt.tStart = t  # local t and not account for scr refresh
                        enter_advance_gradcpt.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(enter_advance_gradcpt, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'enter_advance_gradcpt.started')
                        # update status
                        enter_advance_gradcpt.status = STARTED
                        enter_advance_gradcpt.setAutoDraw(True)
                    
                    # if enter_advance_gradcpt is active this frame...
                    if enter_advance_gradcpt.status == STARTED:
                        # update params
                        pass
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        ES_gradcpt.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in ES_gradcpt.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "ES_gradcpt" ---
                for thisComponent in ES_gradcpt.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for ES_gradcpt
                ES_gradcpt.tStop = globalClock.getTime(format='float')
                ES_gradcpt.tStopRefresh = tThisFlipGlobal
                thisExp.addData('ES_gradcpt.stopped', ES_gradcpt.tStop)
                ES_loop.addData('es_gradcpt_slider.response', es_gradcpt_slider.getRating())
                ES_loop.addData('es_gradcpt_slider.rt', es_gradcpt_slider.getRT())
                # check responses
                if es_gradcpt_key_resp.keys in ['', [], None]:  # No response was made
                    es_gradcpt_key_resp.keys = None
                ES_loop.addData('es_gradcpt_key_resp.keys',es_gradcpt_key_resp.keys)
                if es_gradcpt_key_resp.keys != None:  # we had a response
                    ES_loop.addData('es_gradcpt_key_resp.rt', es_gradcpt_key_resp.rt)
                    ES_loop.addData('es_gradcpt_key_resp.duration', es_gradcpt_key_resp.duration)
                # store data for ES_loop (TrialHandler)
                # Run 'End Routine' code from save_data_es_gradcpt
                if do_es:
                    slider_rating = es_gradcpt_slider.rating
                    slider_rt = es_gradcpt_slider.rt
                    enter_rt = es_gradcpt_key_resp.rt
                    es_gradcpt_data.append({
                        'subject': expInfo['participant'],
                        'total_runtime_mins': (datetime.now() - experiment_start_time).total_seconds() / 60,
                        'gradcpt_trial': gradcpt_trial+1,
                        'probe': probe_count,
                        # Index starting at 1
                        'trial': es_trial + 1,
                        'item_name': cur_items[es_trial]['item_name'],
                        'low_anchor': cur_items[es_trial]['low_anchor'],
                        'high_anchor': cur_items[es_trial]['high_anchor'],
                        'item_onset_s': item_onset,
                        'response': slider_rating,
                        'slider_rt': slider_rt,
                        'enter_rt': enter_rt
                       })
                        
                    # Changing es_trial here bc it affects the index of es_items
                    es_trial += 1
                    win.mouseVisible = False
                # the Routine "ES_gradcpt" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "ES_isi_gradcpt" ---
                # create an object to store info about Routine ES_isi_gradcpt
                ES_isi_gradcpt = data.Routine(
                    name='ES_isi_gradcpt',
                    components=[es_gradcpt_isi_text],
                )
                ES_isi_gradcpt.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from es_gradcpt_code
                if not do_es:
                    continueRoutine=False
                    ES_loop.finished = True
                    
                
                
                # store start times for ES_isi_gradcpt
                ES_isi_gradcpt.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                ES_isi_gradcpt.tStart = globalClock.getTime(format='float')
                ES_isi_gradcpt.status = STARTED
                thisExp.addData('ES_isi_gradcpt.started', ES_isi_gradcpt.tStart)
                ES_isi_gradcpt.maxDuration = None
                # keep track of which components have finished
                ES_isi_gradcptComponents = ES_isi_gradcpt.components
                for thisComponent in ES_isi_gradcpt.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "ES_isi_gradcpt" ---
                # if trial has changed, end Routine now
                if isinstance(ES_loop, data.TrialHandler2) and thisES_loop.thisN != ES_loop.thisTrial.thisN:
                    continueRoutine = False
                ES_isi_gradcpt.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *es_gradcpt_isi_text* updates
                    
                    # if es_gradcpt_isi_text is starting this frame...
                    if es_gradcpt_isi_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        es_gradcpt_isi_text.frameNStart = frameN  # exact frame index
                        es_gradcpt_isi_text.tStart = t  # local t and not account for scr refresh
                        es_gradcpt_isi_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(es_gradcpt_isi_text, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'es_gradcpt_isi_text.started')
                        # update status
                        es_gradcpt_isi_text.status = STARTED
                        es_gradcpt_isi_text.setAutoDraw(True)
                    
                    # if es_gradcpt_isi_text is active this frame...
                    if es_gradcpt_isi_text.status == STARTED:
                        # update params
                        pass
                    
                    # if es_gradcpt_isi_text is stopping this frame...
                    if es_gradcpt_isi_text.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > es_gradcpt_isi_text.tStartRefresh + es_isi_time-frameTolerance:
                            # keep track of stop time/frame for later
                            es_gradcpt_isi_text.tStop = t  # not accounting for scr refresh
                            es_gradcpt_isi_text.tStopRefresh = tThisFlipGlobal  # on global time
                            es_gradcpt_isi_text.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'es_gradcpt_isi_text.stopped')
                            # update status
                            es_gradcpt_isi_text.status = FINISHED
                            es_gradcpt_isi_text.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        ES_isi_gradcpt.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in ES_isi_gradcpt.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "ES_isi_gradcpt" ---
                for thisComponent in ES_isi_gradcpt.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for ES_isi_gradcpt
                ES_isi_gradcpt.tStop = globalClock.getTime(format='float')
                ES_isi_gradcpt.tStopRefresh = tThisFlipGlobal
                thisExp.addData('ES_isi_gradcpt.stopped', ES_isi_gradcpt.tStop)
                # the Routine "ES_isi_gradcpt" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                thisExp.nextEntry()
                
            # completed 6.0 repeats of 'ES_loop'
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            # --- Prepare to start Routine "prepare" ---
            # create an object to store info about Routine prepare
            prepare = data.Routine(
                name='prepare',
                components=[prepare_text],
            )
            prepare.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from prepare_code
            if trials.finished:
                continueRoutine = False
            prepare_start_time = datetime.now()
            timer = 5
            # store start times for prepare
            prepare.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prepare.tStart = globalClock.getTime(format='float')
            prepare.status = STARTED
            thisExp.addData('prepare.started', prepare.tStart)
            prepare.maxDuration = None
            # keep track of which components have finished
            prepareComponents = prepare.components
            for thisComponent in prepare.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "prepare" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            prepare.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 5.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from prepare_code
                timer = 5 - round((datetime.now() - prepare_start_time).total_seconds())
                
                # *prepare_text* updates
                
                # if prepare_text is starting this frame...
                if prepare_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prepare_text.frameNStart = frameN  # exact frame index
                    prepare_text.tStart = t  # local t and not account for scr refresh
                    prepare_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prepare_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prepare_text.started')
                    # update status
                    prepare_text.status = STARTED
                    prepare_text.setAutoDraw(True)
                
                # if prepare_text is active this frame...
                if prepare_text.status == STARTED:
                    # update params
                    prepare_text.setText('The next images will appear in {} seconds.'.format(timer), log=False)
                
                # if prepare_text is stopping this frame...
                if prepare_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > prepare_text.tStartRefresh + 5-frameTolerance:
                        # keep track of stop time/frame for later
                        prepare_text.tStop = t  # not accounting for scr refresh
                        prepare_text.tStopRefresh = tThisFlipGlobal  # on global time
                        prepare_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'prepare_text.stopped')
                        # update status
                        prepare_text.status = FINISHED
                        prepare_text.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    prepare.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prepare.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prepare" ---
            for thisComponent in prepare.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prepare
            prepare.tStop = globalClock.getTime(format='float')
            prepare.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prepare.stopped', prepare.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if prepare.maxDurationReached:
                routineTimer.addTime(-prepare.maxDuration)
            elif prepare.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-5.000000)
            thisExp.nextEntry()
            
        # completed N_trials * 2 repeats of 'trials'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "block_rest" ---
        # create an object to store info about Routine block_rest
        block_rest = data.Routine(
            name='block_rest',
            components=[block_rest_text, block_rest_key_resp],
        )
        block_rest.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from write_data_gradcpt
        write_data(es_gradcpt_data, 'ESGRADCPT')
        write_data(gradcpt_data, 'GRADCPT')
        
        
        # create starting attributes for block_rest_key_resp
        block_rest_key_resp.keys = []
        block_rest_key_resp.rt = []
        _block_rest_key_resp_allKeys = []
        # store start times for block_rest
        block_rest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        block_rest.tStart = globalClock.getTime(format='float')
        block_rest.status = STARTED
        thisExp.addData('block_rest.started', block_rest.tStart)
        block_rest.maxDuration = None
        # keep track of which components have finished
        block_restComponents = block_rest.components
        for thisComponent in block_rest.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "block_rest" ---
        # if trial has changed, end Routine now
        if isinstance(blocks, data.TrialHandler2) and thisBlock.thisN != blocks.thisTrial.thisN:
            continueRoutine = False
        block_rest.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *block_rest_text* updates
            
            # if block_rest_text is starting this frame...
            if block_rest_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                block_rest_text.frameNStart = frameN  # exact frame index
                block_rest_text.tStart = t  # local t and not account for scr refresh
                block_rest_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(block_rest_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'block_rest_text.started')
                # update status
                block_rest_text.status = STARTED
                block_rest_text.setAutoDraw(True)
            
            # if block_rest_text is active this frame...
            if block_rest_text.status == STARTED:
                # update params
                pass
            
            # *block_rest_key_resp* updates
            waitOnFlip = False
            
            # if block_rest_key_resp is starting this frame...
            if block_rest_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                block_rest_key_resp.frameNStart = frameN  # exact frame index
                block_rest_key_resp.tStart = t  # local t and not account for scr refresh
                block_rest_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(block_rest_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'block_rest_key_resp.started')
                # update status
                block_rest_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(block_rest_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(block_rest_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if block_rest_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = block_rest_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _block_rest_key_resp_allKeys.extend(theseKeys)
                if len(_block_rest_key_resp_allKeys):
                    block_rest_key_resp.keys = _block_rest_key_resp_allKeys[-1].name  # just the last key pressed
                    block_rest_key_resp.rt = _block_rest_key_resp_allKeys[-1].rt
                    block_rest_key_resp.duration = _block_rest_key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                block_rest.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in block_rest.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "block_rest" ---
        for thisComponent in block_rest.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for block_rest
        block_rest.tStop = globalClock.getTime(format='float')
        block_rest.tStopRefresh = tThisFlipGlobal
        thisExp.addData('block_rest.stopped', block_rest.tStop)
        # check responses
        if block_rest_key_resp.keys in ['', [], None]:  # No response was made
            block_rest_key_resp.keys = None
        blocks.addData('block_rest_key_resp.keys',block_rest_key_resp.keys)
        if block_rest_key_resp.keys != None:  # we had a response
            blocks.addData('block_rest_key_resp.rt', block_rest_key_resp.rt)
            blocks.addData('block_rest_key_resp.duration', block_rest_key_resp.duration)
        # the Routine "block_rest" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed N_blocks repeats of 'blocks'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "debrief" ---
    # create an object to store info about Routine debrief
    debrief = data.Routine(
        name='debrief',
        components=[debrief_text, debrief_resp],
    )
    debrief.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from debrief_code
    write_data(es_data, 'ES', 'post')
    
    if lab_visit == 1:
        debrief = 'That concludes this session of the experiment. We look forward to seeing you again in session 2! Press the spacebar to continue.'
    elif lab_visit == 2:
        debrief = ('That concludes this session of the experiment. Feel free to read the '
        'text below to learn more about the purpose of this study. Or press the space bar '
        'to end the experiment.\n\nWe are interested in how performance on a task relates '
        'to the contents of mental experience. When one is actively engaged in a task, '
        'thoughts tend to be constrained to the task at hand. However, when one is performing a '
        "task but not actually engaged in the task (maybe not trying very hard), then one's thoughts "
        "could be all over the place. We're interested in whether how someone responds when they're not engaged "
        "in the task will reflect the nature of thoughts happening in their mind. For example, "
        "if someone is not engaged in the task and is responding quickly, that might suggest "
        "they are having thoughts all related to one topic. However, if that person is showing "
        "highly variable response times, then it's possible that their thoughts are highly varied too, "
        "in the sense that they are freely moving and jumping between topics. This would help researchers "
        "better understand the relationship between task control and the dynamics of mental experience, perhaps "
        "benefiting those who sufer from dysregulated mental experience (eg, rumination, ADHD).\n\nPress the space bar to "
        "conclude the experiment.")
    debrief_text.setText(debrief)
    # create starting attributes for debrief_resp
    debrief_resp.keys = []
    debrief_resp.rt = []
    _debrief_resp_allKeys = []
    # store start times for debrief
    debrief.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    debrief.tStart = globalClock.getTime(format='float')
    debrief.status = STARTED
    thisExp.addData('debrief.started', debrief.tStart)
    debrief.maxDuration = None
    # keep track of which components have finished
    debriefComponents = debrief.components
    for thisComponent in debrief.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "debrief" ---
    debrief.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *debrief_text* updates
        
        # if debrief_text is starting this frame...
        if debrief_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            debrief_text.frameNStart = frameN  # exact frame index
            debrief_text.tStart = t  # local t and not account for scr refresh
            debrief_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(debrief_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'debrief_text.started')
            # update status
            debrief_text.status = STARTED
            debrief_text.setAutoDraw(True)
        
        # if debrief_text is active this frame...
        if debrief_text.status == STARTED:
            # update params
            pass
        
        # *debrief_resp* updates
        waitOnFlip = False
        
        # if debrief_resp is starting this frame...
        if debrief_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            debrief_resp.frameNStart = frameN  # exact frame index
            debrief_resp.tStart = t  # local t and not account for scr refresh
            debrief_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(debrief_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'debrief_resp.started')
            # update status
            debrief_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(debrief_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(debrief_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if debrief_resp.status == STARTED and not waitOnFlip:
            theseKeys = debrief_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _debrief_resp_allKeys.extend(theseKeys)
            if len(_debrief_resp_allKeys):
                debrief_resp.keys = _debrief_resp_allKeys[-1].name  # just the last key pressed
                debrief_resp.rt = _debrief_resp_allKeys[-1].rt
                debrief_resp.duration = _debrief_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            debrief.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in debrief.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "debrief" ---
    for thisComponent in debrief.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for debrief
    debrief.tStop = globalClock.getTime(format='float')
    debrief.tStopRefresh = tThisFlipGlobal
    thisExp.addData('debrief.stopped', debrief.tStop)
    # check responses
    if debrief_resp.keys in ['', [], None]:  # No response was made
        debrief_resp.keys = None
    thisExp.addData('debrief_resp.keys',debrief_resp.keys)
    if debrief_resp.keys != None:  # we had a response
        thisExp.addData('debrief_resp.rt', debrief_resp.rt)
        thisExp.addData('debrief_resp.duration', debrief_resp.duration)
    thisExp.nextEntry()
    # the Routine "debrief" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Thanks" ---
    # create an object to store info about Routine Thanks
    Thanks = data.Routine(
        name='Thanks',
        components=[thanks_text],
    )
    Thanks.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for Thanks
    Thanks.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Thanks.tStart = globalClock.getTime(format='float')
    Thanks.status = STARTED
    thisExp.addData('Thanks.started', Thanks.tStart)
    Thanks.maxDuration = None
    # keep track of which components have finished
    ThanksComponents = Thanks.components
    for thisComponent in Thanks.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Thanks" ---
    Thanks.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *thanks_text* updates
        
        # if thanks_text is starting this frame...
        if thanks_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            thanks_text.frameNStart = frameN  # exact frame index
            thanks_text.tStart = t  # local t and not account for scr refresh
            thanks_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(thanks_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'thanks_text.started')
            # update status
            thanks_text.status = STARTED
            thanks_text.setAutoDraw(True)
        
        # if thanks_text is active this frame...
        if thanks_text.status == STARTED:
            # update params
            pass
        
        # if thanks_text is stopping this frame...
        if thanks_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > thanks_text.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                thanks_text.tStop = t  # not accounting for scr refresh
                thanks_text.tStopRefresh = tThisFlipGlobal  # on global time
                thanks_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'thanks_text.stopped')
                # update status
                thanks_text.status = FINISHED
                thanks_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Thanks.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Thanks.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Thanks" ---
    for thisComponent in Thanks.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Thanks
    Thanks.tStop = globalClock.getTime(format='float')
    Thanks.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Thanks.stopped', Thanks.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if Thanks.maxDurationReached:
        routineTimer.addTime(-Thanks.maxDuration)
    elif Thanks.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
