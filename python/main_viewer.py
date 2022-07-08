import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import scipy.io
import scipy.spatial.transform
import os
import math

path_data = '../owl/'



fn_cam_params = path_data +'/cam_params.mat'

fn_vertex_shader = 'rendering_vs_geometry_only.c'
fn_fragment_shader = 'rendering_full_fs.c'
fn_quad_vertex_shader = 'rendering_screen_vs.c'
fn_quad_fragment_shader = 'rendering_screen_fs.c'

fn_object_models =\
    ['output_pbrdf/meshCurrent_clustering.mat',]


for i in range(len(fn_object_models)):
    fn_object_models[i] = path_data+fn_object_models[i]

selected_view = 0

intensity_scale_levels = 2**(np.array(range(-30,30))/2)
intensity_scale_level = 30
intensity_scale = intensity_scale_levels[intensity_scale_level]
scale_levels = 2**(np.array(range(-10,10))/4)
scale_level = 10
scale = intensity_scale_levels[intensity_scale_level]
light_stokes_base = np.array([1.0, 1.0, 0, 0]) # same with hardware setting
view_stokes_base = np.array([1.0, 1.0, 0.0, 0])/2 # same with hardware setting
light_stokes = intensity_scale * light_stokes_base
view_stokes = view_stokes_base
color = np.array([1.0, 1.0, 1.0])
select_rendering = np.array([1.0, 1.0, 1.0])
rendering_mode = 0 # 0 for rendering, 1:normal


prev_pos = None
H=0
W=0
H_screen=0
W_screen=0
prev_3d_pos = None
box_center = None
is_drag = False
prev_extrinsic = None
prev_light_extrinsic = None
extrinsic = None
light_extrinsic = None
ctrl_pressed = False
shift_pressed = False
box_size = None
rot_theta = 0
cam_mat = None
shader = None
clicked_screen_x = 0
clicked_screen_y = 0
window_title = "capture setting"

def rotation_matrix(axis, angle):
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    R = scipy.spatial.transform.Rotation.from_rotvec(angle * axis)
    return R.as_matrix()

def calc_3d_pos(xpos,ypos):
    sphere_size = min(H,W)/2
    x = (xpos-W/2)/sphere_size
    y = (ypos-H/2)/sphere_size
    radius = math.sqrt(x**2+y**2)
    radius = radius%4
    if radius<1:
        z = -math.sqrt(1-radius)
    elif radius<2:
        z = math.sqrt(radius-1)
        x = x/radius*math.sqrt(1-z**2)
        y = y/radius*math.sqrt(1-z**2)
    elif radius<3:
        z = -math.sqrt(3-radius)
        x = -x/radius*math.sqrt(1-z**2)
        y = -y/radius*math.sqrt(1-z**2)
    else:
        z = math.sqrt(radius-3)
        x = -x/radius*math.sqrt(1-z**2)
        y = -y/radius*math.sqrt(1-z**2)
    return np.array([x,y,z])

def find_shortest_rotation(init_pos,final_pos):
    axis = np.cross(init_pos,final_pos)
    sin_theta = np.sqrt(np.dot(axis,axis))
    cos_theta = np.dot(init_pos,final_pos)
    theta = np.arctan2(sin_theta,cos_theta)
    return rotation_matrix(axis, theta)

def find_model_rotation(obj_center, initial_pos, final_pos):
    translation1 = np.concatenate((np.eye(3), 0 - obj_center.reshape((3, 1))), 1)
    translation1 = np.concatenate((translation1, np.array([[0, 0, 0, 1]])), 0)
    rotation = find_shortest_rotation(initial_pos, final_pos)
    rotation = np.concatenate((rotation, np.zeros((3, 1))), 1)
    rotation = np.concatenate((rotation, np.array([[0, 0, 0, 1]])), 0)
    translation2 = np.concatenate((np.eye(3), obj_center.reshape((3, 1))), 1)
    translation2 = np.concatenate((translation2, np.array([[0, 0, 0, 1]])), 0)
    return translation2 @ rotation @ translation1

def find_model_rotation_with_axis(obj_center, axis, theta):
    translation1 = np.concatenate((np.eye(3), 0 - obj_center.reshape((3, 1))), 1)
    translation1 = np.concatenate((translation1, np.array([[0, 0, 0, 1]])), 0)
    rotation = rotation_matrix(axis, theta)
    rotation = np.concatenate((rotation, np.zeros((3, 1))), 1)
    rotation = np.concatenate((rotation, np.array([[0, 0, 0, 1]])), 0)
    translation2 = np.concatenate((np.eye(3), obj_center.reshape((3, 1))), 1)
    translation2 = np.concatenate((translation2, np.array([[0, 0, 0, 1]])), 0)
    return translation2 @ rotation @ translation1

def find_model_translation(translation):
    translation_mat = np.concatenate((np.eye(3), translation.reshape((3, 1))), 1)
    translation_mat = np.concatenate((translation_mat, np.array([[0, 0, 0, 1]])), 0)
    return translation_mat

def mouse_button_callback(window, button, action, mods):
    global prev_pos,prev_3d_pos,is_drag, prev_extrinsic, extrinsic, prev_light_extrinsic, light_extrinsic, box_center, \
        clicked_screen_x, clicked_screen_y, W, H
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        prev_pos = glfw.get_cursor_pos(window)
        clicked_screen_x = prev_pos[0]//W
        clicked_screen_y = prev_pos[1]//H
        prev_3d_pos = calc_3d_pos(prev_pos[0]-clicked_screen_x*W,prev_pos[1]-clicked_screen_y*H)
        is_drag = True
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
        now_pos = glfw.get_cursor_pos(window)
        if not (prev_pos[0] == now_pos[0] and prev_pos[1] == now_pos[1]):
            if ctrl_pressed:
                displacement = np.array([now_pos[0] - prev_pos[0], now_pos[1] - prev_pos[1], 0]) / min(H, W) * min(box_size)
                displacement = np.transpose(prev_extrinsic[:,0:3])@displacement.reshape((3, 1))
                extrinsic = prev_extrinsic @ find_model_translation(displacement)
                prev_extrinsic = extrinsic.copy()
            elif shift_pressed:
                now_3d_pos = calc_3d_pos(now_pos[0]-clicked_screen_x*W,now_pos[1]-clicked_screen_y*H)
                light_extrinsic = prev_light_extrinsic@find_model_rotation(box_center,
                                                                           np.transpose(prev_light_extrinsic[:,0:3])@now_3d_pos,
                                                                           np.transpose(prev_light_extrinsic[:,0:3])@prev_3d_pos)
                prev_light_extrinsic = light_extrinsic.copy()

            else:
                now_3d_pos = calc_3d_pos(now_pos[0]-clicked_screen_x*W,now_pos[1]-clicked_screen_y*H)
                model_lotation =find_model_rotation(box_center,
                                                    np.transpose(prev_extrinsic[:,0:3])@prev_3d_pos,
                                                    np.transpose(prev_extrinsic[:,0:3])@now_3d_pos)
                extrinsic = prev_extrinsic@model_lotation
                prev_extrinsic = extrinsic.copy()
                light_extrinsic = prev_light_extrinsic @ model_lotation

                prev_light_extrinsic = light_extrinsic.copy()
        is_drag = False


def cursor_pos_callback(window, xpos, ypos):
    global prev_pos, prev_3d_pos, is_drag, prev_extrinsic, extrinsic, prev_light_extrinsic, light_extrinsic
    if is_drag and not (prev_pos[0]==xpos and prev_pos[1]==ypos):
        if ctrl_pressed:
            displacement = np.array([xpos-prev_pos[0],ypos-prev_pos[1],0])/min(H,W)*min(box_size)
            displacement = np.transpose(prev_extrinsic[:,0:3])@displacement.reshape((3, 1))
            extrinsic = prev_extrinsic@find_model_translation(displacement)
        elif shift_pressed:
            light_extrinsic = prev_light_extrinsic @ find_model_rotation(box_center,
                                                             np.transpose(prev_light_extrinsic[:, 0:3]) @ calc_3d_pos(xpos-clicked_screen_x*W,
                                                                                                                ypos-clicked_screen_y*H),
                                                             np.transpose(prev_light_extrinsic[:, 0:3]) @ prev_3d_pos)
        else:
            model_lotation = find_model_rotation(box_center,
                                                 np.transpose(prev_extrinsic[:, 0:3]) @ prev_3d_pos,
                                                 np.transpose(prev_extrinsic[:, 0:3]) @ calc_3d_pos(
                                                                 xpos - clicked_screen_x * W,
                                                                 ypos - clicked_screen_y * H))
            extrinsic = prev_extrinsic @ model_lotation
            light_extrinsic = prev_light_extrinsic @ model_lotation

def key_callback(window, key, scancode, action, mods):
    global ctrl_pressed,rot_theta,extrinsic,prev_extrinsic,select_rendering,shift_pressed,rendering_mode,\
        printscreen_iter,single_scattering_model,light_stokes,view_stokes,light_stokes_base,view_stokes_base, \
        window_title
    if key == glfw.KEY_LEFT_CONTROL and action == glfw.PRESS:
        ctrl_pressed=True
    if key == glfw.KEY_LEFT_CONTROL and action == glfw.RELEASE:
        ctrl_pressed=False
    if key == glfw.KEY_LEFT_SHIFT and action == glfw.PRESS:
        shift_pressed=True
    if key == glfw.KEY_LEFT_SHIFT and action == glfw.RELEASE:
        shift_pressed=False
    if key == glfw.KEY_1 and action == glfw.PRESS:
        rendering_mode = 0
        glfw.set_window_title(window,window_title)
    if key == glfw.KEY_2 and action == glfw.PRESS:
        rendering_mode = 1
        glfw.set_window_title(window,"normal")
    if key == glfw.KEY_KP_0 and action == glfw.PRESS and rendering_mode == 0:
        light_stokes_base = np.array([1.0, 1.0, 0, 0]) # same with hardware setting
        view_stokes_base = np.array([1.0, 1.0, 0.0, 0])/2 # same with hardware setting
        light_stokes = intensity_scale * light_stokes_base
        view_stokes = view_stokes_base
        window_title = "capture setting"
        glfw.set_window_title(window,window_title)
    if key == glfw.KEY_KP_1 and action == glfw.PRESS and rendering_mode == 0:
        light_stokes_base = intensity_scale * np.array([1.0, 0, 0, 0])
        view_stokes_base = np.array([0, 0, 1.0, 0])
        light_stokes = intensity_scale * light_stokes_base
        view_stokes = view_stokes_base
        window_title = "M(2,0)"
        glfw.set_window_title(window,window_title)
    if key == glfw.KEY_KP_2 and action == glfw.PRESS and rendering_mode == 0:
        light_stokes_base = intensity_scale * np.array([0, 1.0, 0, 0])
        view_stokes_base = np.array([0, 0, 1.0, 0])
        light_stokes = intensity_scale * light_stokes_base
        view_stokes = view_stokes_base
        window_title = "M(2,1)"
        glfw.set_window_title(window,window_title)
    if key == glfw.KEY_KP_3 and action == glfw.PRESS and rendering_mode == 0:
        light_stokes_base = intensity_scale * np.array([0, 0, 1.0, 0])
        view_stokes_base = np.array([0, 0, 1.0, 0])
        light_stokes = intensity_scale * light_stokes_base
        view_stokes = view_stokes_base
        window_title = "M(2,2)"
        glfw.set_window_title(window,window_title)
    if key == glfw.KEY_KP_4 and action == glfw.PRESS and rendering_mode == 0:
        light_stokes_base = intensity_scale * np.array([1.0, 0, 0, 0])
        view_stokes_base = np.array([0, 1.0, 0, 0])
        light_stokes = intensity_scale * light_stokes_base
        view_stokes = view_stokes_base
        window_title = "M(1,0)"
        glfw.set_window_title(window,window_title)
    if key == glfw.KEY_KP_5 and action == glfw.PRESS and rendering_mode == 0:
        light_stokes_base = intensity_scale * np.array([0, 1.0, 0, 0])
        view_stokes_base = np.array([0, 1.0, 0, 0])
        light_stokes = intensity_scale * light_stokes_base
        view_stokes = view_stokes_base
        window_title = "M(1,1)"
        glfw.set_window_title(window,window_title)
    if key == glfw.KEY_KP_6 and action == glfw.PRESS and rendering_mode == 0:
        light_stokes_base = intensity_scale * np.array([0, 0, 1.0, 0])
        view_stokes_base = np.array([0, 1.0, 0, 0])
        light_stokes = intensity_scale * light_stokes_base
        view_stokes = view_stokes_base
        window_title = "M(1,2)"
        glfw.set_window_title(window,window_title)
    if key == glfw.KEY_KP_7 and action == glfw.PRESS and rendering_mode == 0:
        light_stokes_base = intensity_scale * np.array([1.0, 0, 0, 0])
        view_stokes_base = np.array([1.0, 0, 0, 0])
        light_stokes = intensity_scale * light_stokes_base
        view_stokes = view_stokes_base
        window_title = "M(0,0)"
        glfw.set_window_title(window,window_title)
    if key == glfw.KEY_KP_8 and action == glfw.PRESS and rendering_mode == 0:
        light_stokes_base = intensity_scale * np.array([0, 1.0, 0, 0])
        view_stokes_base = np.array([1.0, 0, 0, 0])
        light_stokes = intensity_scale * light_stokes_base
        view_stokes = view_stokes_base
        window_title = "M(0,1)"
        glfw.set_window_title(window,window_title)
    if key == glfw.KEY_KP_9 and action == glfw.PRESS and rendering_mode == 0:
        light_stokes_base = intensity_scale * np.array([0, 0, 1.0, 0])
        view_stokes_base = np.array([1.0, 0, 0, 0])
        light_stokes = intensity_scale * light_stokes_base
        view_stokes = view_stokes_base
        window_title = "M(0,2)"
        glfw.set_window_title(window,window_title)

def scroll_callback(window, xoffset, yoffset):
    global intensity_scale_level,intensity_scale_levels,intensity_scale,light_stokes,\
        scale_level,scale_levels,scale,shift_pressed
    if shift_pressed:
        intensity_scale_level+= int(np.floor(yoffset))
        if intensity_scale_level<0:
            intensity_scale_level=0
        elif intensity_scale_level>=len(intensity_scale_levels):
            intensity_scale_level = len(intensity_scale_levels)-1
        intensity_scale = intensity_scale_levels[intensity_scale_level]
        light_stokes = intensity_scale * light_stokes_base
    else:
        scale_level += int(np.floor(yoffset))
        if scale_level < 0:
            scale_level = 0
        elif scale_level >= len(scale_levels):
            scale_level = len(scale_levels) - 1
        scale = scale_levels[scale_level]



def make_bounding_box(position):
    min_pos = position[0,:].copy()
    max_pos = position[0,:].copy()
    for i in range(1,position.shape[0]):
        for j in range(3):
            if min_pos[j]>position[i,j]:
                min_pos[j] = position[i, j]
            if max_pos[j]<position[i,j]:
                max_pos[j] = position[i, j]
    return min_pos,max_pos


def main():
    global cam_mat, H, W, H_screen, W_screen, box_center, prev_extrinsic, prev_light_extrinsic, extrinsic, \
        light_extrinsic, box_size, rot_theta, shader, single_scattering_model_left, single_scattering_model_right, \
        window_title
    os.makedirs(path_data, exist_ok=True)

    cam_mat = scipy.io.loadmat(fn_cam_params)
    H = cam_mat['H'][0, 0].astype(int)
    W = cam_mat['W'][0, 0].astype(int)

    # initialize glfw
    if not glfw.init():
        return

    window = glfw.create_window(W, H, window_title, None, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window,cursor_pos_callback)
    glfw.set_key_callback(window, key_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    # shader
    vs_file = open(fn_vertex_shader, mode='r')
    vertex_shader = vs_file.read()
    vs_file.close()

    fs_file = open(fn_fragment_shader, mode='r')
    fragment_shader = fs_file.read()
    fs_file.close()



    quad_vs_file = open(fn_quad_vertex_shader, mode='r')
    quad_vertex_shader = quad_vs_file.read()
    quad_vs_file.close()

    quad_fs_file = open(fn_quad_fragment_shader, mode='r')
    quad_fragment_shader = quad_fs_file.read()
    quad_fs_file.close()

    quad_shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(quad_vertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(quad_fragment_shader, GL_FRAGMENT_SHADER))


    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    # load mat data
    obj_mat = []
    position_data = []
    rho_data = []
    normal_data = []
    eta_data = []
    ks1_data = []
    m1_data = []
    rho1_data = []
    m2_data = [[] for i in range(len(fn_object_models))]
    faces = []
    for i in range(len(fn_object_models)):
        obj_mat.append(scipy.io.loadmat(fn_object_models[i]))

        position_data.append(obj_mat[i]['meshCurrent']['vertices'][0, 0])
        position_data[i] = np.transpose(position_data[i])
        rho_data.append(obj_mat[i]['meshCurrent']['rho'][0, 0])
        rho_data[i] = np.float32(np.transpose(rho_data[i]))
        normal_data.append(obj_mat[i]['meshCurrent']['normals'][0, 0])
        normal_data[i] = np.float32(np.transpose(normal_data[i]))
        eta_data.append(obj_mat[i]['meshCurrent']['eta'][0, 0])
        eta_data[i] = np.float32(np.transpose(eta_data[i]))
        ks1_data.append(obj_mat[i]['meshCurrent']['ks1'][0, 0])
        ks1_data[i] = np.float32(np.transpose(ks1_data[i]))
        m1_data.append(obj_mat[i]['meshCurrent']['m1'][0, 0])
        m1_data[i] = np.float32(np.transpose(m1_data[i]))
        rho1_data.append(obj_mat[i]['meshCurrent']['ks2'][0, 0])
        rho1_data[i] = np.float32(np.transpose(rho1_data[i]))
        m2_data[i] = obj_mat[i]['meshCurrent']['m2'][0, 0]
        m2_data[i] = np.float32(np.transpose(m2_data[i]))
        faces.append(obj_mat[i]['meshCurrent']['faces'][0, 0])
        faces[i] = np.reshape(np.transpose(faces[i]), -1)
        faces[i] -= 1

        if i==0:
            min_pos, max_pos = make_bounding_box(position_data[i])
            box_center = (min_pos+max_pos)/2
            box_size = (max_pos-min_pos)

    quad = [   -1, -1, 0.1,
                1, -1, 0.1,
                1,  1, 0.1,
               -1,  1, 0.1,]
    quad = np.array(quad, dtype = np.float32)
    quad_tex_coord = [   0, 0,
                1, 0,
                1,  1,
               0,  1,]
    quad_tex_coord = np.array(quad_tex_coord, dtype = np.float32)
    quad_indices = [0, 1, 2,
               2, 3, 0]

    quad_indices = np.array(quad_indices, dtype= np.uint32)

    if len(fn_object_models)>1:
        vao_obj = glGenVertexArrays(len(fn_object_models))
        VBO = glGenBuffers(len(fn_object_models))
        EBO = glGenBuffers(len(fn_object_models))
        VBO_rho = glGenBuffers(len(fn_object_models))
        VBO_normal = glGenBuffers(len(fn_object_models))
        VBO_eta = glGenBuffers(len(fn_object_models))
        VBO_ks1 = glGenBuffers(len(fn_object_models))
        VBO_m1 = glGenBuffers(len(fn_object_models))
        VBO_rho1 = glGenBuffers(len(fn_object_models))
        VBO_m2 = glGenBuffers(len(fn_object_models))
    else:
        vao_obj = [glGenVertexArrays(1)]
        VBO = [glGenBuffers(1)]
        EBO = [glGenBuffers(1)]
        VBO_rho = [glGenBuffers(1)]
        VBO_normal = [glGenBuffers(1)]
        VBO_eta = [glGenBuffers(1)]
        VBO_ks1 = [glGenBuffers(1)]
        VBO_m1 = [glGenBuffers(1)]
        VBO_rho1 = [glGenBuffers(1)]
        VBO_m2 = [glGenBuffers(1)]
    for i in range(len(fn_object_models)):
        glBindVertexArray(vao_obj[i])
        glBindBuffer(GL_ARRAY_BUFFER, VBO[i])
        glBufferData(GL_ARRAY_BUFFER, position_data[i].nbytes, position_data[i], GL_STATIC_DRAW)
        glBindAttribLocation(shader, 0, "position")
        position = glGetAttribLocation(shader, "position")
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE,
                              position_data[i].dtype.itemsize*3, ctypes.c_void_p(0))
        glEnableVertexAttribArray(position)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[i])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces[i].nbytes, faces[i], GL_STATIC_DRAW)


        glBindBuffer(GL_ARRAY_BUFFER, VBO_rho[i])
        glBufferData(GL_ARRAY_BUFFER, rho_data[i].nbytes, rho_data[i], GL_STATIC_DRAW)
        glBindAttribLocation(shader, 2, "rho")
        rho = glGetAttribLocation(shader, "rho")
        glVertexAttribPointer(rho, 3, GL_FLOAT, GL_FALSE,
                              rho_data[i].dtype.itemsize*3, ctypes.c_void_p(0))
        glEnableVertexAttribArray(rho)

        glBindBuffer(GL_ARRAY_BUFFER, VBO_normal[i])
        glBufferData(GL_ARRAY_BUFFER, normal_data[i].nbytes, normal_data[i], GL_STATIC_DRAW)
        glBindAttribLocation(shader, 1, "normal")
        normal = glGetAttribLocation(shader, "normal")
        glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE,
                              normal_data[i].dtype.itemsize * 3, ctypes.c_void_p(0))
        glEnableVertexAttribArray(normal)

        glBindBuffer(GL_ARRAY_BUFFER, VBO_eta[i])
        glBufferData(GL_ARRAY_BUFFER, eta_data[i].nbytes, eta_data[i], GL_STATIC_DRAW)
        glBindAttribLocation(shader, 3, "eta")
        eta = glGetAttribLocation(shader, "eta")
        glVertexAttribPointer(eta, 1, GL_FLOAT, GL_FALSE,
                              eta_data[i].dtype.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(eta)

        glBindBuffer(GL_ARRAY_BUFFER, VBO_ks1[i])
        glBufferData(GL_ARRAY_BUFFER, ks1_data[i].nbytes, ks1_data[i], GL_STATIC_DRAW)
        glBindAttribLocation(shader, 6, "ks1")
        ks1 = glGetAttribLocation(shader, "ks1")
        glVertexAttribPointer(ks1, 1, GL_FLOAT, GL_FALSE,
                              ks1_data[i].dtype.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(ks1)

        glBindBuffer(GL_ARRAY_BUFFER, VBO_m1[i])
        glBufferData(GL_ARRAY_BUFFER, m1_data[i].nbytes, m1_data[i], GL_STATIC_DRAW)
        glBindAttribLocation(shader, 4, "m1")
        m1 = glGetAttribLocation(shader, "m1")
        glVertexAttribPointer(m1, 1, GL_FLOAT, GL_FALSE,
                              m1_data[i].dtype.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(m1)

        glBindBuffer(GL_ARRAY_BUFFER, VBO_rho1[i])
        glBufferData(GL_ARRAY_BUFFER, rho1_data[i].nbytes, rho1_data[i], GL_STATIC_DRAW)
        glBindAttribLocation(shader, 7, "rho1")
        rho1 = glGetAttribLocation(shader, "rho1")
        glVertexAttribPointer(rho1, 3, GL_FLOAT, GL_FALSE,
                              rho1_data[i].dtype.itemsize * 3, ctypes.c_void_p(0))
        glEnableVertexAttribArray(rho1)

        glBindBuffer(GL_ARRAY_BUFFER, VBO_m2[i])
        glBufferData(GL_ARRAY_BUFFER, m2_data[i].nbytes, m2_data[i], GL_STATIC_DRAW)
        glBindAttribLocation(shader, 5, "m2")
        m2 = glGetAttribLocation(shader, "m2")
        glVertexAttribPointer(m2, 1, GL_FLOAT, GL_FALSE,
                              m2_data[i].dtype.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(m2)

    vao_quad = glGenVertexArrays(1)
    glBindVertexArray(vao_quad)

    VBO_quad = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO_quad)
    glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
    glBindAttribLocation(quad_shader, 0, "aPos")
    position = glGetAttribLocation(quad_shader, "aPos")
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE,
                          quad.dtype.itemsize*3, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)

    EBO_quad = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_quad)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, quad_indices.nbytes, quad_indices, GL_STATIC_DRAW)

    VBO_tex_quad = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO_tex_quad)
    glBufferData(GL_ARRAY_BUFFER, quad_tex_coord.nbytes, quad_tex_coord, GL_STATIC_DRAW)
    glBindAttribLocation(quad_shader, 1, "tex_coord")
    tex_coord = glGetAttribLocation(quad_shader, "tex_coord")
    glVertexAttribPointer(tex_coord, 2, GL_FLOAT, GL_FALSE,
                          quad_tex_coord.dtype.itemsize*2, ctypes.c_void_p(0))
    glEnableVertexAttribArray(tex_coord)


    if len(fn_object_models)>1:
        fbo = glGenFramebuffers(len(fn_object_models))
        quad_texture = glGenTextures(len(fn_object_models))
        rbo = glGenRenderbuffers(len(fn_object_models))
    else:
        fbo = [glGenFramebuffers(len(fn_object_models))]
        quad_texture = [glGenTextures(len(fn_object_models))]
        rbo = [glGenRenderbuffers(len(fn_object_models))]
    for i in range(len(fn_object_models)):
        glBindFramebuffer(GL_FRAMEBUFFER, fbo[i])
        glActiveTexture(GL_TEXTURE0+i)
        glBindTexture(GL_TEXTURE_2D, quad_texture[i])
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, quad_texture[i], 0)

        glBindRenderbuffer(GL_RENDERBUFFER, rbo[i])
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, W, H)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo[i])

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print("ERROR::FRAMEBUFFER:: Framebuffer is not complete!")
            glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    # rendering


    intrinsic = cam_mat['intrinsic'][selected_view, 0]
    original_extrinsic = cam_mat['extrinsic'][selected_view, 0]
    if os.path.isfile('%s/extrinsic.npy'%(path_data)):
        original_extrinsic = np.load('%s/extrinsic.npy'%(path_data))

    extrinsic = original_extrinsic.copy()
    prev_extrinsic = original_extrinsic.copy()
    light_extrinsic = original_extrinsic.copy()
    prev_light_extrinsic = original_extrinsic.copy()
    while True:

        scaled_intrinsic = intrinsic.copy()
        scaled_intrinsic[0,0] = scaled_intrinsic[0,0]*scale
        scaled_intrinsic[1,1] = scaled_intrinsic[1,1]*scale
        opencv_intrinsic = np.concatenate((np.concatenate((scaled_intrinsic, np.array([[0],[0],[0]])), axis=1), np.array([[0, 0, 0, 1]])), axis=0)
        opencv_extrinsic = np.concatenate((extrinsic, np.array([[0, 0, 0, 1]])), axis=0)
        y_v = -extrinsic[1, 0:3]
        y_l = -light_extrinsic[1, 0:3]

        glViewport(0,0,W,H)
        glUseProgram(shader)

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnable(GL_DEPTH_TEST)

        glUniform1ui(glGetUniformLocation(shader, 'H'), H)
        glUniform1ui(glGetUniformLocation(shader, 'W'), W)
        glUniform4fv(glGetUniformLocation(shader, 'view_stokes'), 1, view_stokes)
        glUniform3fv(glGetUniformLocation(shader, 'color'), 1, color)

        glUniform4fv(glGetUniformLocation(shader, 'light_stokes'), 1, light_stokes)
        glUniform3fv(glGetUniformLocation(shader, 'select_rendering'), 1, select_rendering)
        glUniform1ui(glGetUniformLocation(shader, 'rendering_mode'), rendering_mode)

        glUniform3fv(glGetUniformLocation(shader, 'lPos'), 1, np.transpose(-light_extrinsic[:,0:3])@light_extrinsic[:,3])
        glUniform3fv(glGetUniformLocation(shader, 'vPos'), 1, np.transpose(-extrinsic[:,0:3])@extrinsic[:,3])
        glUniform3fv(glGetUniformLocation(shader, 'y_l'), 1, y_l)
        glUniform3fv(glGetUniformLocation(shader, 'y_v'), 1, y_v)

        glUniformMatrix4fv(glGetUniformLocation(shader,'opencv_intrinsic'),1,GL_TRUE,opencv_intrinsic)
        glUniformMatrix4fv(glGetUniformLocation(shader,'opencv_extrinsic'),1,GL_TRUE,opencv_extrinsic)

        for i in range(len(fn_object_models)):
            glfw.poll_events()
            glBindFramebuffer(GL_FRAMEBUFFER, fbo[i])
            glClearColor(0, 0, 0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glBindVertexArray(vao_obj[i])
            glDrawElements(GL_TRIANGLES, faces[i].size, GL_UNSIGNED_INT, None)

        glUseProgram(quad_shader)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glViewport(0, 0, W, H)
        glDisable(GL_DEPTH_TEST)
        glBindVertexArray(vao_quad)
        glUniform1i(glGetUniformLocation(quad_shader,'screenTexture1'),0)
        for i in range(len(fn_object_models)):
            glActiveTexture(GL_TEXTURE0+i)
            glBindTexture(GL_TEXTURE_2D, quad_texture[i])

        glDrawElements(GL_TRIANGLES, quad_indices.size, GL_UNSIGNED_INT, None)
        glfw.swap_buffers(window)


        if glfw.window_should_close(window):
            break

    glfw.terminate()

if __name__ == "__main__":
    main()