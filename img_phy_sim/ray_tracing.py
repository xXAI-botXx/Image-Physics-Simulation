"""
Definitions:

Rays:
[
    [[x1, x2, y1, y2], [x1, x2, y1, y2], [x1, x2, y1, y2]],  # Ray 1
    [[x1, x2, y1, y2]]  # Ray 2
]

Direction is given in Degree, where 0 degree is the top (north) direction. Therefore:
- 0° north
- 90° east
- 180° south
- 270° west

=> [0, 360)

Start Positions are given by (relative width position, relative height position)
"""
from img_phy_sim.img import open, get_width_height

import math

import numpy as np
import cv2
# from shapely.geometry import LineString, Point


# --------------
# >>> Helper <<<
# --------------

def get_linear_degree_range(step_size=10, offset=0):
    degree_range = list(range(0, 360, step_size))
    return list(map(lambda x: x+offset, degree_range))


def degree_to_vector(degree):
    rad = math.radians(degree)
    return [math.cos(rad), math.sin(rad)]

def vector_to_degree(vector):
    x, y = vector
    degree = math.degrees(math.atan2(y, x))  # atan2 returns angles between -180° and 180°
    return int( degree % 360 ) 


# ----------------------
# >>> Core Functions <<<
# ----------------------
def get_walls(img):
    # detect edges and contours
    edges = cv2.Canny((img*255).astype(np.uint8), 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # convert contours to line segments
    walls = []
    for c in contours:
        for i in range(len(c)-1):
            x1, y1 = c[i][0]
            x2, y2 = c[i+1][0]
            walls += [[x1, y1, x2, y2]]
    return walls



def get_all_pixel_coordinates_in_between(x1, y1, x2, y2):
    """
    Returns a list of all pixel coordinates (x, y) between (x1, y1) and (x2, y2)
    using Bresenham's line algorithm.

    https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    """
    coordinates = []

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1

    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1

    if dx > dy:
        err = dx / 2.0
        while x != x2:
            coordinates += [(x, y)]
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            coordinates += [(x, y)]
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    coordinates += [(x2, y2)]  # include the last point
    return coordinates



def get_wall_map(img, thickness=1):
    IMG_WIDTH, IMG_HEIGHT =  get_width_height(img)
    wall_map = np.full((IMG_HEIGHT, IMG_WIDTH), np.inf, dtype=np.uint16)  # uint16 to get at least 360 degree/value range

    # detect edges and contours
    edges = cv2.Canny((img*255).astype(np.uint8), 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # convert contours to line segments
    for c in contours:
        for i in range(len(c)-1):
            x1, y1 = c[i][0]
            x2, y2 = c[i+1][0]
            dy = y2 - y1
            dx = x2 - x1
            angle = math.atan2(dy, dx)
            angle_deg = math.degrees(angle)
            for x, y in get_all_pixel_coordinates_in_between(x1, y1, x2, y2):
                # wall_map[y, x] = int(angle_deg) % 360

                for tx in range(-thickness, thickness+1):
                    for ty in range(-thickness, thickness+1):
                        nx, ny = x+tx, y+ty
                        if 0 <= nx < IMG_WIDTH and 0 <= ny < IMG_HEIGHT:
                            wall_map[ny, nx] = int(angle_deg) % 360
    return wall_map


# def point_line_distance(point, line_start, line_end):
#     x0, y0 = point
#     x1, y1 = line_start
#     x2, y2 = line_end
#     num = abs((x2 - x1)*(y1 - y0) - (x1 - x0)*(y2 - y1))
#     den = np.hypot(x2 - x1, y2 - y1)
#     return num / den



def update_pixel_position(direction_in_degree, cur_position, target_line):
    # 1. Calc distance from point to target line

    # perpendicular vector to line (points toward line)
    point = np.array(cur_position)
    line_start_point = np.array(target_line[0:2])
    line_end_point = np.array(target_line[2:4])

    # projection along the line -> throw the point vector vertical/perpendicular on the line and see where it cuts with normed AP to AB
    # t is the length from point to the line, therefore it gets normed
    t = np.dot(point - line_start_point, line_end_point - line_start_point) / (np.dot(line_end_point - line_start_point, line_end_point - line_start_point) + 1e-8)
    
    # limit it to the line id needed -> because we don't want smaller or bigger values than that
    #   -> 0 would be point A
    #   -> 1 would be point B 
    t = np.clip(t, 0, 1)

    # get closest point by applying the found t as lentgh from startpoint in the line vector direction
    closest = line_start_point + t * (line_end_point - line_start_point)

    # get the final vector to the line
    to_line = closest - point  # vector from current pos to closest point on line
    
    # 2. Calc vector to the degree
    # movement vector based on angle
    rad = math.radians(direction_in_degree)
    move_dir = np.array([math.cos(rad), math.sin(rad)])
    
    # 3. Combine vector to the line and degree vector
    # combine movement towards direction and towards line
    combined = move_dir + to_line * 0.5  # weighting factor
    
    # pick pixel step (continuous to discrete) -> [-1, 0, 1]
    step_x = np.sign(combined[0])
    step_y = np.sign(combined[1])
    
    # clamp to [-1, 1], if bigger/smaller
    step_x = int(np.clip(step_x, -1, 1))
    step_y = int(np.clip(step_y, -1, 1))
    
    return (cur_position[0] + step_x, cur_position[1] + step_y)



def calc_reflection(collide_vector, wall_vector):
    # normalize both
    collide_vector = np.array(collide_vector, dtype=float)
    collide_vector /= np.linalg.norm(collide_vector)
    wall_vector = np.array(wall_vector, dtype=float)
    wall_vector /= np.linalg.norm(wall_vector)

    # calculate the normal of the wall
    normal_wall_vector_1 = np.array([-wall_vector[1], wall_vector[0]])  # rotated +90°
    normal_wall_vector_2 = np.array([wall_vector[1], -wall_vector[0]])  # rotated -90°

    # decide which vector is the right one
    #   -> dot product tells which normal faces the incoming vector
    #   -> dor product shows how similiar 2 vectors are => smaller 0 means they show against each other => right vector
    if np.dot(collide_vector, normal_wall_vector_1) < 0:
        normal_wall_vector = normal_wall_vector_1
    else:
        normal_wall_vector = normal_wall_vector_2
    
    # calc the reflection
    return collide_vector - 2 * np.dot(collide_vector, normal_wall_vector) * normal_wall_vector


def trace_beam(abs_position, img, wall_map, direction_in_degree, max_depth=3, ray_depth=0):
    direction_in_degree = direction_in_degree % 360
    IMG_WIDTH, IMG_HEIGHT =  get_width_height(img)

    ray = []
    current_ray_line = [abs_position[0], abs_position[1], abs_position[0], abs_position[1]]

    # calculate a target line to update the pixels
    #   target vector
    dx = math.cos(math.radians(direction_in_degree))
    dy = math.sin(math.radians(direction_in_degree))
    target_line = [abs_position[0], abs_position[1], abs_position[0], abs_position[1]]
    while (0 <= target_line[2] <= IMG_WIDTH) and (0 <= target_line[3] <= IMG_HEIGHT):
        target_line[2] += 0.01 * dx
        target_line[3] += 0.01 * dy

    # update current ray
    current_position = abs_position
    while True:
        # update position
        current_position = update_pixel_position(direction_in_degree=direction_in_degree, cur_position=current_position, target_line=target_line)

        # check if ray is at end
        if not (0 <= current_position[0] < IMG_WIDTH and 0 <= current_position[1] < IMG_HEIGHT):
            ray += [current_ray_line]
            break

        next_pixel = img[int(current_position[1]), int(current_position[0])]

        # check if hit building
        if float(next_pixel) == 0.0:
            ray += [current_ray_line]

            # get building wall reflection angle
            building_angle = wall_map[int(current_position[0]), int(current_position[1])]
            wall_vector = degree_to_vector(building_angle)

            # calc new direct vector
            new_direction = calc_reflection(collide_vector=degree_to_vector(direction_in_degree), wall_vector=wall_vector)
            new_direction_in_degree = vector_to_degree(new_direction)

            # start new beam calculation
            if ray_depth <= max_depth:
                ray += trace_beam(abs_position=current_ray_line[2:4], img=img, wall_map=wall_map, direction_in_degree=new_direction_in_degree, ray_depth=ray_depth+1)
            
            break
        else:
            # update current ray
            current_ray_line[2] = current_position[0]
            current_ray_line[3] = current_position[1]
    
    return ray



def trace_beams(rel_position, img_src, directions_in_degree):
    img = open(src=img_src, should_scale=True, should_print=False)
    IMG_WIDTH, IMG_HEIGHT =  get_width_height(img)
    abs_position = (rel_position[0] * IMG_WIDTH, rel_position[1] * IMG_HEIGHT)

    wall_map = get_wall_map(img, thickness=1)

    rays = []
    for direction_in_degree in directions_in_degree:
        rays += [trace_beam(abs_position=abs_position, img=img, wall_map=wall_map, direction_in_degree=direction_in_degree)]

    return rays


def get_max_width_height(rays):
    max_x = 0
    max_y = 0
    for ray in rays:
        for x1, y1, x2, y2 in ray:
            max_x = max(max_x, x1)
            max_x = max(max_x, x2)

            max_y = max(max_y, y1)
            max_y = max(max_y, y2)
    return max_x, max_y

def scale_rays(rays, max_x=None, max_y=None, new_max_x=None, new_max_y=None):
    if max_x is None or max_y is None:
        max_x, max_y = get_max_width_height(rays)

    scaled_rays = []
    for ray in rays:
        scaled_ray = []
        for x1, y1, x2, y2 in ray:
            x1 /= max_x
            y1 /= max_y
            x2 /= max_x
            y2 /= max_y

            if new_max_x is not None and new_max_y is not None:
                x1 *= new_max_x
                y1 *= new_max_y
                x2 *= new_max_x
                y2 *= new_max_y

            scaled_ray += [[x1, y1, x2, y2]]
        scaled_rays += [scaled_ray]
    return scaled_rays

def draw_rays(rays, individual_ray_outpout=False, as_channels=True, 
              img_background=None, ray_value=255, ray_thickness=1, 
              img_shape=(256, 256), dtype=float, standard_value=0,
              should_scale_rays_to_image=True):
    # prepare background image
    if img_background is None:
        img = np.full(shape=img_shape, fill_value=dtype(standard_value), dtype=dtype)
    else:
        img = img_background.copy()

    # rescale rays to fit inside image bounds if desired
    height, width = img.shape[:2]
    if should_scale_rays_to_image:
        max_x, max_y = get_max_width_height(rays)
        rays = scale_rays(rays, max_x, max_y, new_max_x=width-1, new_max_y=height-1)

    nrays = len(rays)
    if individual_ray_outpout:
        if as_channels:
            img = np.repeat(img[..., None], nrays, axis=-1)
            # img_shape += (nrays, )
            # img = np.full(shape=img_shape, fill_value=dtype(standard_value), dtype=dtype)
        else:
            imgs = [np.copy(img) for _ in range(nrays)]

    # draw on image
    for idx, ray in enumerate(rays):
        for x1, y1, x2, y2 in ray:
            start_point = (int(x1), int(y1))
            end_point = (int(x2), int(y2))
            
            if individual_ray_outpout:
                if as_channels:
                    layer = np.ascontiguousarray(img[..., idx])
                    cv2.line(layer, start_point, end_point, ray_value, ray_thickness)
                    img[..., idx] = layer
                    # cv2.line(img[..., idx], start_point, end_point, ray_value, ray_thickness)
                    # if img.ndim == 3:
                    #     cv2.line(img[:, :, idx], start_point, end_point, ray_value, ray_thickness)
                    # else:
                    #     cv2.line(img[:, :, :, idx], start_point, end_point, ray_value, ray_thickness)
                else:
                    cv2.line(imgs[idx], start_point, end_point, ray_value, ray_thickness)
            else:
                cv2.line(img, start_point, end_point, ray_value, ray_thickness)


    if individual_ray_outpout and not as_channels:
        return imgs
    else:
        return img




def get_rays(rel_position, img_src, directions_in_degree,
             rays, individual_ray_outpout=False, as_channels=True, 
             img_background=None, ray_value=255, ray_thickness=1, 
             img_shape=(256, 256), dtype=float, standard_value=0,
             should_scale_rays_to_image=True):
    rays = trace_beams(rel_position=rel_position, img_src=img_src, directions_in_degree=directions_in_degree)

    img = draw_rays(rays, individual_ray_outpout=individual_ray_outpout, as_channels=as_channels, 
                    img_background=img_background, ray_value=ray_value, ray_thickness=ray_thickness, 
                    img_shape=img_shape, dtype=dtype, standard_value=standard_value,
                    should_scale_rays_to_image=should_scale_rays_to_image)
    
    return img



