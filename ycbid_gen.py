import sys, os, time, datetime, platform
import numpy as np
import operator as op
from functools import reduce
sys.path.insert(0, "/helper")
import RESSOURCES
import gazebo as gz
import sdfgzobject as sdfo
import pandas as pd
from PIL import Image
import argparse


#set recursion depth a little bit higher
sys.setrecursionlimit(2000)


MODELPATH = '../models/'
SAVE_PATH = os.path.dirname(os.path.realpath(__file__)) + '/output/'

DFCOLUMNS = ['object_in_focus', 'distance', 'roll', 'pitch', 'yaw',
             'first_occluder', '1distance', '1roll', '1pitch', '1yaw',
             'second_occluder', '2distance', '2roll', '2pitch', '2yaw',
             'third_occluder', '3distance', '3roll', '3pitch', '3yaw',
             'occlusion', 'occlusion_left', 'occlusion_right','n_occluders',
             'eye', 'scale', 'lighting', 'filepath', 'segmentation_map','id']

X_TARGET_POS = 0.5
N_MAX_OCCLUDERS = 3
N_SIMULATION_STEPS = 35


def ncr(n, r):
    # n! / (r!(n-r)!)
    r = min(r, n - r)
    if r == -1:
        return 0
    if r == 0:
        return 1
    numer = reduce(op.mul, range(n, n - r, -1))
    denom = reduce(op.mul, range(1, r + 1))
    return numer // denom


def ck(N, k):
    n = k - 1
    while ncr(n, k) < N:
        n += 1
    return n - 1


def number_to_k_combination(N, k):
    #  https://en.wikipedia.org/wiki/Combinatorial_number_system
    s = set()
    N += 1
    while k > 0:
        n = ck(N, k)
        s.add(n)
        N -= ncr(n, k)
        k -= 1
    return frozenset(s)


def sample_n_occluders_sets(n, target_object_index, total_number_object, number_occluders_in_set):
    """ given the index of a 'target' object, the total number of objects in the set and the number of occluders to be
    rendered for every 'target' object, returns a set S1 containing n sets S2, S2 being sets of occluders to be
    placed in front of the target object."""
    as_numbers = set()
    maxi = ncr(total_number_object - 1, number_occluders_in_set)
    while len(as_numbers) < n:
        new = np.random.randint(0, maxi)
        if new not in as_numbers:
            as_numbers.add(new)
    as_combi_not_shifted = {number_to_k_combination(x, number_occluders_in_set) for x in as_numbers}
    as_combi_shifted = set()
    for combi in as_combi_not_shifted:
        shifted_combi = set()
        for obj in combi:
            if obj >= target_object_index:
                shifted_combi.add(obj + 1)
            else:
                shifted_combi.add(obj)
        as_combi_shifted.add(frozenset(shifted_combi))
    return frozenset(as_combi_shifted)


def do(doable, done, chosen_objects, max_objects, search_limit=2):
    # doable: set S1 of sets S2 of objects that could be rendered if chosen_objects was enriched by the missing objects
    # done: set S1 of sets S2 of objects that are already included in chosen_objects
    # chosen_objects: objects that have been picked so far
    # max_objects: maximum amount of objects tht chosen_objects can contain
    # search_limit: a threshold to limit the search space/time
    #
    # simple case, if we can't chose any object anymore, return
    if len(chosen_objects) == max_objects:
        return done, chosen_objects
    # if we can still chose objects, then we want to find the bests objects to add to our set of chosen_objects
    best = 0
    best_done_after, best_chosen_objects_after = done, chosen_objects
    sorted_doable = sorted(doable, key=lambda x: len(x & chosen_objects), reverse=True)
    # for every set of objects that could be rendered if we would chose the missing objects,
    for limit, element in zip(range(search_limit), sorted_doable):
        # chose the missing objects
        new_chosen_objects = element | chosen_objects
        # determine all the other sets that could be rendered with those new objects
        new_done = {x for x in doable if x.issubset(new_chosen_objects)}
        new_done |= done
        # determine all the other sets that could still be rendered if we would chose other objects again
        new_doable = {x for x in doable if len(x | new_chosen_objects) <= max_objects and x not in new_done}
        # get the best objects to be chosen, given the fact that we chose 'new_chosen_objects'
        done_after, chosen_objects_after = do(new_doable, new_done, new_chosen_objects, max_objects)
        # if the amount of object set renderable when chosing 'new_chosen_objects' is bigger than when chosing
        # already explored set of new chosen objects
        if len(done_after) > best:
            # remember that those new_chosen_objects are the best we found so far
            best = len(done_after)
            best_done_after, best_chosen_objects_after = done_after, chosen_objects_after
    return best_done_after, best_chosen_objects_after


def select_n_best_objects_from_set(n, objects_sets):
    # objects_sets: a set of sets containing target + occluders
    return do(objects_sets, set(), set(), n)


def sample_target_occluders_pairs(n_sets_per_target, total_number_object, number_occluders_in_set):
    # generates a set containing one element per object, each element being a pair (target, occluders_sets),
    # occluders_sets is a set of n_sets_per_target sets, each being a set of occluders to be rendered in front of target
    s = set()
    for target in range(total_number_object):
        occluders = sample_n_occluders_sets(n_sets_per_target, target, total_number_object, number_occluders_in_set)
        s.add((target, occluders))
    return s



def select_n_best_objects(n, target_occluders_pairs):
    # transforms the target_occluders_pairs into a set of sets S2, S2 containing all the objects (target + occluders)
    # needed to render one scene
    s = set()
    for target, occluders in target_occluders_pairs:
        for occ in occluders:
            s.add(frozenset(occ | {target}))
    # pass this new created set to select_n_best_objects_from_set in order to determine which are the bests objects
    done, objects = select_n_best_objects_from_set(n, s)
    return objects


class Indent:
    indent = 0

    def __init__(self, string):
        self.string = string

    def __enter__(self):
        self.t0 = time.time()
        print(Indent.indent * "|    " + self.string)
        Indent.indent += 1

    def __exit__(self, type, value, traceback):
        Indent.indent -= 1
        print(Indent.indent * "|    " + "|done  ({0:.3f}s)".format(time.time() - self.t0))



def distancetoangle(object_distance):
    ''' distancetoangle takes the object_distance defined by initializing an object
    and returns the angle needed to adjust vergence of the robot'''
    new_x = object_distance - RESSOURCES.X_EYES_POSITION
    return - np.arctan(RESSOURCES.Y_EYES_DISTANCE /
                       np.sqrt(new_x**2 + RESSOURCES.Y_EYES_DISTANCE**2)) * 360 / (2 * np.pi)


def pixeltoangle(pixel_offset):
    ''' transforms an pixel_offset in pixels to an angle usable for vergence '''
    return pixel_offset * 90 / 320


def get_init_pose(x_dist, rand=False):
    x_pos = x_dist
    y_pos = 0
    z_pos = RESSOURCES.Z_DEFAULT_SCREEN_POS - 0.05
    roll_pos = (np.random.random() * 2. * np.pi)
    pitch_pos = (np.random.random() * 2. * np.pi)
    yaw_pos = (np.random.random() * 2. * np.pi)
    if rand:
        return [x_pos, y_pos, z_pos, roll_pos, pitch_pos ,yaw_pos]
    else:
        return [x_pos, y_pos, z_pos, 0, 0, yaw_pos]

def convolve(arr1, arr2):
    assert(len(arr1.shape) == 3)
    assert(len(arr2.shape) == 3)
    assert(arr1.shape[1] == arr2.shape[1])
    arrlen = arr1.shape[1]
    rlen = arrlen * 2 - 1
    result = np.ones(rlen) * 20
    for i in range(220, 420):
        subarr1 = arr1[:,-i-1:] if i < arrlen else arr1[:,:rlen-i+1]
        subarr2 = arr2[:,i-arrlen:] if i > arrlen else arr2[:,:i+1]
        result[i] = np.sqrt(np.mean((subarr1 - subarr2)**2))
    return result

def perform_vergence(universe, robot, screen):
    left_picture = np.copy(robot.cam_left.get_image())
    right_picture = np.copy(robot.cam_right.get_image())
    # sum over the channels
    convolved_array = convolve(left_picture, right_picture)
    vdiff = np.argmin(convolved_array) - 320 # > should be at 320
    robot.vergence.set_relative_positions([pixeltoangle(vdiff)])
    pass

def initialize_target(universe, robot, screen, gzobjects, target):
    before = get_image_before(robot)
    target_pose = get_init_pose(X_TARGET_POS)
    gzobjects[target].set_positions(target_pose[0], target_pose[1], target_pose[2], target_pose[3], target_pose[4], target_pose[5])
    wait_until_view_changed(universe, robot, before)
    # pretune vergence to distance of objects
    robot.vergence.set_positions([distancetoangle(target_pose[0])])
    # reset screen to position in front
    before = get_image_before(robot)
    screen.set_positions(1,0,RESSOURCES.Z_DEFAULT_SCREEN_POS,0,0.5*np.pi,np.pi)
    universe.time.step_simulation(N_SIMULATION_STEPS)
    # perform vergence with convolution of both inputs
    for i in range(3):
        perform_vergence(universe, robot, screen)
    target_pixels_left, target_pixels_right = get_object_pixels(universe, robot, screen)

    n_target_pixels_left = len(np.where(target_pixels_left > 0.)[0])
    n_target_pixels_right = len(np.where(target_pixels_right > 0.)[0])

    return target_pose, n_target_pixels_left, n_target_pixels_right

def tidy_up(universe, robot, screen, gzobjects, target, occ):
    before = get_image_before(robot)
    for o in occ:
        gzobjects[o].set_positions(-2., 0., 0., 0., 0., 0.)
    gzobjects[target].set_positions(-2., 0., 0., 0., 0., 0.)
    universe.time.step_simulation(N_SIMULATION_STEPS)

def measure_occlusion(universe, robot, screen, target_object, target_pose, n_target_pixels_left, n_target_pixels_right):
    target_object.set_positions(x=-3, y=target_pose[1],
                            z=target_pose[2], roll=target_pose[3], pitch=target_pose[4], yaw=target_pose[5])
    universe.time.step_simulation(N_SIMULATION_STEPS)



    occluders_left = np.copy(robot.cam_left.get_image())
    occluders_right = np.copy(robot.cam_right.get_image())

    before = get_image_before(robot)
    target_object.set_positions(x=target_pose[0], y=target_pose[1],
                            z=target_pose[2], roll=target_pose[3], pitch=target_pose[4], yaw=target_pose[5])
    universe.time.step_simulation(N_SIMULATION_STEPS)

    alltogether_left = np.copy(robot.cam_left.get_image())
    alltogether_right = np.copy(robot.cam_right.get_image())

    alltogether_left[np.where(occluders_left > 0.)] = 0
    alltogether_right[np.where(occluders_right > 0.)] = 0

    occluded_target_indices_left = np.where(np.sum(alltogether_left, axis=2) > 0.)
    occluded_target_indices_right = np.where(np.sum(alltogether_right, axis=2) > 0.)

    occluded_target_indices_left = np.where(np.sum(alltogether_left, axis=2) > 0.)
    occluded_target_indices_right = np.where(np.sum(alltogether_right, axis=2) > 0.)

    occlusion_percentage_left = (n_target_pixels_left - len(occluded_target_indices_left[0]))/n_target_pixels_left
    occlusion_percentage_right = (n_target_pixels_right - len(occluded_target_indices_right[0]))/n_target_pixels_right
    occlusion_percentage_avg = (occlusion_percentage_left + occlusion_percentage_right) / 2.

    print(" " * 80 + "\r" + "occ. avg: {:.3f}".format(occlusion_percentage_avg), end="\r")
    universe.time.step_simulation(N_SIMULATION_STEPS)

    return occlusion_percentage_left, occlusion_percentage_right, occlusion_percentage_avg

def dichotomy(universe, robot, screen, target_object, target_pose, occluder_object, occluder_pose, new_occluder_pose, desired_occlusion, limit, n_target_pixels_left, n_target_pixels_right, iteration=0):
    _,_,current_occlusion = measure_occlusion(universe, robot, screen, target_object, target_pose, n_target_pixels_left, n_target_pixels_right)

    # move occluder to new occluder pose
    occluder_object.set_positions(x=new_occluder_pose[0], y=new_occluder_pose[1],
                            z=new_occluder_pose[2], roll=new_occluder_pose[3], pitch=new_occluder_pose[4], yaw=new_occluder_pose[5])
    universe.time.step_simulation(N_SIMULATION_STEPS)

    _,_,new_occlusion = measure_occlusion(universe, robot, screen, target_object, target_pose, n_target_pixels_left, n_target_pixels_right)
    if iteration > 100:
        best_pose = None
    elif (new_occlusion - desired_occlusion) < ((-1)*limit):
        pose_between = occluder_pose[:]
        pose_between[1] = (occluder_pose[1] + new_occluder_pose[1])/2.
        iteration += 1
        best_pose = dichotomy(universe, robot, screen, target_object, target_pose, occluder_object, occluder_pose, pose_between, desired_occlusion, limit, n_target_pixels_left, n_target_pixels_right, iteration)
    elif (new_occlusion - desired_occlusion) > limit:
        pose_beyond = occluder_pose[:]
        pose_beyond[1] = new_occluder_pose[1] + (occluder_pose[1] + new_occluder_pose[1])/2.
        iteration += 1
        best_pose = dichotomy(universe, robot, screen, target_object, target_pose, occluder_object, new_occluder_pose, pose_beyond, desired_occlusion, limit, n_target_pixels_left, n_target_pixels_right, iteration)
    else:
        best_pose = new_occluder_pose

    return best_pose

def recursive_identifier(savedir, filename, desired_occlusion, identifier):
    protofname = "{}{}{}p_id{}_left.jpg".format(savedir, filename, int(desired_occlusion * 100.), identifier)
    if os.path.isfile(protofname):

        identifier , number = identifier.rsplit('_', 1)
        number = int(number)
        number += 1
        identifier += ("_" + str(number))

        return recursive_identifier(savedir, filename, desired_occlusion, identifier)
    else:
        return identifier

def recursive_file_iteration(fpath):
    if os.path.isfile(fpath):

        fpath, number = fpath.rsplit('.', 1)[0].rsplit("_", 1)
        number = int(number)
        number += 1
        fpath += ("_" + str(number) + ".gzip")

        return recursive_file_iteration(fpath)
    else:
        return fpath

def manage_output(universe, robot, screen, dataframe, gzobjects, target, target_pose, occluder_set, occluder_pose_dict, desired_occlusion, n_target_pixels_left, n_target_pixels_right, debug_images=False):
    occlusion_l, occlusion_r, occlusion_m = measure_occlusion(universe, robot, screen, gzobjects[target], target_pose, n_target_pixels_left, n_target_pixels_right)

    # get semantic_sementation_array before screen is moved and plot it
    seg_left, seg_right = get_semantic_segmentation_array(universe, robot, screen, gzobjects, target, target_pose, occluder_set, occluder_pose_dict)

    screen.set_positions(-3,0,RESSOURCES.Z_DEFAULT_SCREEN_POS,0,0.5*np.pi,np.pi)
    universe.time.step_simulation(N_SIMULATION_STEPS)

    filename = filename_from_index[target]
    number_of_occluders = len(occluder_set)
    # generate identifier
    identifier = "{}_".format(target)

    for o in occluder_set:
        identifier += str(o) + "_"
    identifier += "1"

    # directory management
    savedir = "{}YCB_database2/{}/{}occ/".format(SAVE_PATH, filename, number_of_occluders)
    os.makedirs(os.path.dirname(
        savedir), exist_ok=True)

    # check if files already exist
    identifier = recursive_identifier(savedir, filename, desired_occlusion, identifier)


    with Indent("saving pictures id: {}".format(identifier)):


        # save segmentation-map
        segmentation_map_savefile = "{}{}{}p_id{}.npz".format(savedir, filename, int(desired_occlusion * 100.), identifier)
        np.savez_compressed(segmentation_map_savefile, segmentation_left=seg_left, segmentation_right=seg_right)
        # configure and save images
        im_left = Image.fromarray(
            robot.cam_left.get_image())
        im_left_fine_savefile = "{}{}{}p_id{}_left.jpg".format(savedir, filename, int(desired_occlusion * 100.), identifier)
        im_left.save(im_left_fine_savefile)

        im_right = Image.fromarray(
            robot.cam_right.get_image())
        im_right_fine_savefile = "{}{}{}p_id{}_right.jpg".format(savedir, filename, int(desired_occlusion * 100.), identifier)
        im_right.save(im_right_fine_savefile)

        if debug_images:
            (ana.anaglyph(robot.cam_left.get_image(), robot.cam_right.get_image(), ana.half_color_anaglyph)).save(
                savedir + str(int(desired_occlusion * 100.)) + 'p_' + 'id' + identifier + '_anaglyph.jpeg')

        o_pose_list = []
        o_name_list = []
        for o in occluder_set:
            o_pose_list.append(occluder_pose_dict[o]) # => [0,0,0,0,0,0])
            o_name_list.append(filename_from_index[o]) # => 'filename'
        # insert into DataFrame
        o1name, o1dist, o1roll, o1pitch, o1yaw = (o_name_list[0], o_pose_list[0][0], o_pose_list[0][3], o_pose_list[0][4],o_pose_list[0][5])  if len(occluder_set) > 0 else ('', np.NaN, np.NaN, np.NaN, np.NaN)
        o2name, o2dist, o2roll, o2pitch, o2yaw = (o_name_list[1], o_pose_list[1][0], o_pose_list[1][3], o_pose_list[1][4],o_pose_list[1][5])  if len(occluder_set) > 1 else ('', np.NaN, np.NaN, np.NaN, np.NaN)
        o3name, o3dist, o3roll, o3pitch, o3yaw = (o_name_list[2], o_pose_list[2][0], o_pose_list[2][3], o_pose_list[2][4],o_pose_list[2][5])  if len(occluder_set) > 2 else ('', np.NaN, np.NaN, np.NaN, np.NaN)


        dataframe = dataframe.append(pd.DataFrame([
            [filename, target_pose[0], target_pose[3], target_pose[4], target_pose[5],
            o1name, o1dist, o1roll, o1pitch, o1yaw,
            o2name, o2dist, o2roll, o2pitch, o2yaw,
            o3name, o3dist, o3roll, o3pitch, o3yaw,
            occlusion_m, occlusion_l, occlusion_r, number_of_occluders,
            'left', 'fine', 'highlight', im_left_fine_savefile, segmentation_map_savefile, identifier],

            [filename, target_pose[0], target_pose[3], target_pose[4], target_pose[5],
            o1name, o1dist, o1roll, o1pitch, o1yaw,
            o2name, o2dist, o2roll, o2pitch, o2yaw,
            o3name, o3dist, o3roll, o3pitch, o3yaw,
            occlusion_m, occlusion_l, occlusion_r, number_of_occluders,
            'right', 'fine', 'highlight', im_right_fine_savefile, segmentation_map_savefile, identifier]

        ], columns=DFCOLUMNS), ignore_index=True)
    return dataframe

def universe_test(universe, robot, screen, dataframe):
    return (hasattr(robot, 'head') and hasattr(robot, 'cam_left') and hasattr(robot, 'cam_right'))

def setup_universe(viewerclient=True):
    with Indent("creating universe"):
        WORLD_PATH = "path/to/icub/world/"
        universe = gz.Universe(WORLD_PATH, client=viewerclient, viewer=viewerclient, paused=True, verbose=True)
        universe.time.step_simulation(200)
        robot = gz.Robot(with_colors=True)
        screen = gz.Screen(universe.world, "black_screen", 10, 5, 5, 1, 0, 0.94,
            texture=np.zeros([200,200,3]))
        dataframe = pd.DataFrame(columns=DFCOLUMNS)
        time.sleep(10)

        if universe_test(universe, robot, screen, dataframe):
            return universe, robot, screen, dataframe
        else:
            robot.close()
            universe.close()
            del(screen)
            print('caught setup error')
            return setup_universe(viewerclient)


def get_object_pixels(universe, robot, screen):
    # take pictures left right
    picture_left = np.copy(robot.cam_left.get_image())
    picture_right = np.copy(robot.cam_right.get_image())

    # merge colorchannels and set everything that is not 0 to 255
    semantic_pixels_left = np.sum(picture_left, axis=2)
    semantic_pixels_left[semantic_pixels_left > 0.] = 1.

    semantic_pixels_right = np.sum(picture_right, axis=2)
    semantic_pixels_right[semantic_pixels_right > 0.] = 1.
    
    # hand back arrays 2x [240,320]
    return semantic_pixels_left, semantic_pixels_right

def get_semantic_segmentation_array(universe, robot, screen, gzobjects, target, target_pose, occluder_set, occluder_pose_dict):
    # create array of desired size
    semantic_segmentation_array_left = np.zeros([240,320,len(occluder_set)+1])
    semantic_segmentation_array_right = np.zeros([240,320,len(occluder_set)+1])

    # store occluders in the back
    for o in occluder_set:
        gzobjects[o].set_positions(-2., occluder_pose_dict[o][1],occluder_pose_dict[o][2],occluder_pose_dict[o][3],occluder_pose_dict[o][4],occluder_pose_dict[o][5])
        universe.time.step_simulation(N_SIMULATION_STEPS)

    # get object pixels
    semantic_segmentation_array_left[:,:,0], semantic_segmentation_array_right[:,:,0] = get_object_pixels(universe, robot, screen)
    # move target to back
    gzobjects[target].set_positions(-2., target_pose[1], target_pose[2], target_pose[3], target_pose[4], target_pose[5])
    universe.time.step_simulation(N_SIMULATION_STEPS)

    # loop through occluders
    # i = 1 b/c 0 is reserved for the target
    i = 1
    for o in occluder_set:
      # move occluder to front
      gzobjects[o].set_positions(occluder_pose_dict[o][0],occluder_pose_dict[o][1],occluder_pose_dict[o][2],occluder_pose_dict[o][3],occluder_pose_dict[o][4],occluder_pose_dict[o][5])
      universe.time.step_simulation(N_SIMULATION_STEPS)
      # get pixels
      semantic_segmentation_array_left[:,:,i], semantic_segmentation_array_right[:,:,i] = get_object_pixels(universe, robot, screen)
      # move occluder to back
      gzobjects[o].set_positions(-2.,occluder_pose_dict[o][1],occluder_pose_dict[o][2],occluder_pose_dict[o][3],occluder_pose_dict[o][4],occluder_pose_dict[o][5])
      universe.time.step_simulation(N_SIMULATION_STEPS)
      i += 1

    # move all to front
    gzobjects[target].set_positions(target_pose[0], target_pose[1], target_pose[2], target_pose[3], target_pose[4], target_pose[5])
    universe.time.step_simulation(N_SIMULATION_STEPS)

    for o in occluder_set:
        gzobjects[o].set_positions(occluder_pose_dict[o][0],occluder_pose_dict[o][1],occluder_pose_dict[o][2],occluder_pose_dict[o][3],occluder_pose_dict[o][4],occluder_pose_dict[o][5])
        universe.time.step_simulation(N_SIMULATION_STEPS)
    # return array
    return semantic_segmentation_array_left, semantic_segmentation_array_right


def create_log(logname):
    datestring = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')
    filepath = SAVE_PATH + "ycbid/logs/{}log_{}_{}.txt".format(logname, platform.node(), datestring)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    return filepath

def log_error(target, occluder_set, message, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    targetname = filename_from_index[target]
    occludernames = ""
    for o in occluder_set:
        occludernames += filename_from_index[o] + ", "
    error_log = open(filepath, "a")
    error_log.write("[ERROR] {} \t target: {}, occluders: {} \n".format(message, targetname, occludernames))
    print("[ERROR] {}".format(message))
    error_log.close()
    pass


def generate_pictures(target_occluders_pairs, objects, desired_occlusion):
    # generates all the scene that can be done with the given set of object,
    # returns all the scenes that have not been generated
    n_object_done = 0
    U, R, S, DF = setup_universe(viewerclient=False)
    errorfile = create_log("error")

    with Indent("loading objects"):
        gzobjects = load_objects(U, R, objects)
        if gzobjects is None:
            R.close()
            U.close()
            del(S)
            return None
    with Indent("generating objects {}".format(str(objects))):
        new_target_occluders_pairs = set()
        for target, occluders in target_occluders_pairs:
            current_target_remaining_occluders = set()
            if target in objects:
                with Indent("using target {}".format(target)):


                    for occ in occluders:
                        if occ.issubset(objects):
                            # move target to front, get pixels
                            target_pose, n_t_px_l, n_t_px_r = initialize_target(U, R, S, gzobjects, target)
                            occluder_number = 1
                            occlusion_successful = True
                            # create a dict for storing occluder purposes
                            occluder_pose_dict = {}
                            for o in occ:
                                occluder_distance = np.arange(0.4, 0.2, (0.4 - 0.2) / (-1 * N_MAX_OCCLUDERS))[occluder_number - 1].item()
                                occluder_y_pose = 0.4*(-1)**occluder_number
                                #occluder_y_pose =  np.random.choice([-1,1]).item()
                                occlusion_goal = (desired_occlusion / len(occ))*occluder_number

                                # initialize occluder
                                occluder_pose_dict[o] = get_init_pose(occluder_distance)
                                before = get_image_before(R)
                                gzobjects[o].set_positions(occluder_pose_dict[o][0],occluder_pose_dict[o][1],occluder_pose_dict[o][2],occluder_pose_dict[o][3],occluder_pose_dict[o][4],occluder_pose_dict[o][5])
                                wait_until_view_changed(U, R, before)

                                _,_,initial_occlusion = measure_occlusion(U, R, S, gzobjects[target], target_pose, n_t_px_l, n_t_px_r)
                                if initial_occlusion > occlusion_goal:
                                    best_pose = dichotomy(U, R, S, gzobjects[target], target_pose, gzobjects[o], occluder_pose_dict[o], [occluder_distance,occluder_y_pose, RESSOURCES.Z_DEFAULT_SCREEN_POS - 0.05,0,0,0], occlusion_goal, 0.02, n_t_px_l, n_t_px_r)

                                    if best_pose:
                                        occluder_pose_dict[o] = best_pose
                                        occluder_number +=1
                                    else:
                                        log_error(target, occ, "dichotomy algorithm failed", errorfile)
                                        occlusion_successful = False
                                else:
                                    log_error(target, occ, 'maximum occlusion lower than desired', errorfile)
                                    occlusion_successful = False
                            with Indent("Manage Output"):
                                if occlusion_successful:
                                    DF = manage_output(U, R, S, DF, gzobjects, target, target_pose, occ, occluder_pose_dict, desired_occlusion, n_t_px_l, n_t_px_r)
                            with Indent("Tidy up Scene"):
                                tidy_up(U, R, S, gzobjects, target, occ)
                            n_object_done += 1
                        else:
                            current_target_remaining_occluders.add(occ)

                    # make sure target is removed
                    gzobjects[target].set_positions(-2., 0., 0., 0., 0., 0.)
                    U.time.step_simulation(N_SIMULATION_STEPS)


            else:
                current_target_remaining_occluders = occluders
            new_target_occluders_pairs.add((target, frozenset(current_target_remaining_occluders)))
    # write to files
    os.makedirs(os.path.dirname(
        SAVE_PATH + 'ycbid/' + 'dataframes/' + str(args.n_occluders)+'occ/' + str(int(args.desired_occlusion*100)) +'p/'), exist_ok=True)
    DF_path = SAVE_PATH + 'ycbid/' + 'dataframes/' + str(args.n_occluders)+'occ/' + str(int(args.desired_occlusion*100)) +'p/' + 'YCBdb2_datastruct_1.gzip'
    DF_path = recursive_file_iteration(DF_path)
    with Indent("save to file: {}".format(DF_path)):
        DF.to_pickle(DF_path)

    with Indent("closing universe, done {} objects".format(n_object_done)):
        R.close()
        U.close()
        del(S)
    return new_target_occluders_pairs


def target_occluders_pairs_is_empty(target_occluders_pairs):
    for target, occluders in target_occluders_pairs:
        if len(occluders) != 0:
            return False
    return True


def generate(target_occluders_pairs, max_objects_simultaneously, desired_occlusion):
    # generates all the scenes in target_occluders_pairs,
    # loading only max_objects_simultaneously objects at the same time
    while not target_occluders_pairs_is_empty(target_occluders_pairs):
        with Indent("Searching best objects to load"):
            objects = select_n_best_objects(max_objects_simultaneously, target_occluders_pairs)
        new_target_occluders_pairs = generate_pictures(target_occluders_pairs, objects, desired_occlusion)
        if new_target_occluders_pairs is not None:
            target_occluders_pairs = new_target_occluders_pairs
        else:
            print("object loading failed")


def same_images(before, after):
    # criterium to determine if two images are the same
    err = np.mean((after - before)**2)
    #print(err)
    return err < 0.1


def wait_until_view_changed(universe, robot, before):
    # generates gazebo iterations until the view of the robot changed, or until the maximum of iteration is reached
    universe.time.step_simulation(100)
    after = robot.cam_left.get_image()
    i = 0
    while same_images(before, after):
        if i == 100:
            return False
        i += 1
        universe.time.step_simulation(100)
        after = robot.cam_left.get_image()
    return True


filename_from_index = []
for filename in os.listdir(MODELPATH):
    if filename.startswith('G'):
        filename_from_index.append(filename)
print(filename_from_index)


def get_image_before(robot):
    return np.copy(robot.cam_left.get_image())


def load_one_object(universe, robot, obj):
    # loads one object, check it's creation was successful, move it to the stock and check that moving the object worked
    # returns the SDFGzObject instance if it worked
    filename = filename_from_index[obj]
    before = get_image_before(robot)
    gzobj = sdfo.SDFGzObject(universe.world, filename, MODELPATH + filename + '/model.sdf',
                             x=0.3, y=0., z=RESSOURCES.Z_DEFAULT_SCREEN_POS - 0.05,
                             roll=0., pitch=0., yaw=0.)
    gzobj.set_positions(x=0.3, y=0., z=RESSOURCES.Z_DEFAULT_SCREEN_POS - 0.05, roll=0., pitch=0., yaw=0.)
    worked = wait_until_view_changed(universe, robot, before)
    if worked:
        before = get_image_before(robot)
        #plt.imshow(before)
        #plt.show()
        gzobj.set_positions(-2., 0., 0., 0., 0., 0.)
        if wait_until_view_changed(universe, robot, before):
            return gzobj
    return None


def load_objects(universe, robot, objects):
    # loads a set of objects in the simulator
    gzobjects = {}
    for o in objects:
        gzobj = load_one_object(universe, robot, o)
        if gzobj is None:
            return None
        gzobjects[o] = gzobj
    return gzobjects

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Commandline Inputs')
    parser.add_argument('--scenes', type=int, default=10)
    parser.add_argument('--n_occluders', type=int, default=2)
    parser.add_argument('--objects_in_universe', type=int, default=6)
    parser.add_argument('--desired_occlusion', type=float, default=0.2)

    args = parser.parse_args()

    target_occluders_pairs = sample_target_occluders_pairs(args.scenes, 79, args.n_occluders)
    generate(target_occluders_pairs, args.objects_in_universe, args.desired_occlusion)
