# Gravitational lensing effect simulator
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import copy
from PIL import Image

image_path = "~\" # Your path of picture here, in format .png
scale = 110 # scale of background picture

GM =   0.01 # mass of black hole, also stands for half of schwarzchild radius
D_OB = 3.0 # distance from observer to black hole
D_SB = 3.0 # distance from source to black hole 
GRID_OB = 100 # observer definition
ANG_OB = 50 # observer viewing angle

def generate_unit_vectors(angle, grid):
    thetax,thetay = np.meshgrid(np.linspace(-0.5*angle/180*np.pi,+0.5*angle/180*np.pi,grid),
                                np.linspace(-0.5*angle/180*np.pi,+0.5*angle/180*np.pi,grid))
    x = np.tan(thetax).astype(np.float32)
    y = np.tan(thetay).astype(np.float32)
    z = np.ones_like(x).astype(np.float32)
    r = np.sqrt(x**2+y**2+z**2)
    x = x/r
    y = y/r
    z = z/r
    vectors = np.stack((x,y,z),axis=-1)
    return vectors

def accel(u):
    return -u+3*GM*u**2

def initializer(Vi,stepphi):
    u0 = 1.0/D_OB
    if Vi[0]==0 and Vi[1]==0:
        return 1/GM,0 
    v0 = Vi[2]/(np.sqrt(Vi[0]**2+Vi[1]**2)*D_OB)
    v0 += accel(u0)*0.5*stepphi
    return u0,v0

def target_hit(uf,phif):
    return((1.0/uf)*np.cos(phif)>D_SB)


def tracer(Vi,maxsteps):
    vi = copy.deepcopy(Vi)
    ms = copy.deepcopy(maxsteps)
    step = np.pi/ms
    phis = np.linspace(-np.pi,0,ms)
    u,v = initializer(vi,step)

    if vi[0]==0.0 and vi[1]==0.0:
        cos_xy = 0.0
        sin_xy = 0.0
    else:
        vi_xy = np.sqrt(vi[0]**2+vi[1]**2)
        cos_xy = vi[0]/vi_xy
        sin_xy = vi[1]/vi_xy

    for phi in phis:
        if u > 1.0/(2*GM):
            return False, None
        
        state_target = target_hit(u,phi)
        u += v*step
        v += accel(u)*step
        
        if (state_target ^ target_hit(u,phi+step)):
            x_target = cos_xy * 1.0/u * np.sin(phi+step)
            y_target = sin_xy * 1.0/u * np.sin(phi+step)
            return True, np.array([x_target,y_target])
    
    return False, None

def generate_test_array(i, j):
    arr = np.zeros((i, j), dtype=np.float32)
    arr[(np.arange(i) + np.arange(j).reshape(-1, 1)) % 2 == 0] = 1

    return arr

# test background
# background = generate_test_array(20, 20)
im_frame = Image.open(image_path)
np_frame = np.array(im_frame)


imagex = np_frame.shape[0]
imagey = np_frame.shape[1]
background = np.zeros((imagex,imagey))
background = np_frame[:,:,0]/256

vs = generate_unit_vectors(ANG_OB,GRID_OB).astype(np.float32)
img = np.zeros((GRID_OB,GRID_OB))

for i in range(GRID_OB):
    for j in range(GRID_OB):
        v = copy.deepcopy(vs[i,j])
        state, pos = tracer(v,1000)
        if state:
            pos_background = np.floor(scale*pos + np.array([imagex,imagey])*0.5).astype(int)
            if pos_background[0]<imagex and pos_background[1]<imagey and pos_background[0]>=0 and pos_background[1]>=0:
                img[i,j] = background[pos_background[0],pos_background[1]]
        else: img[i,j] = 0

plt.imshow(img.T,origin='lower',cmap=plt.get_cmap('gray'))
plt.savefig('./lensed_image.png',dpi=400)
