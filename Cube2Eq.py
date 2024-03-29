import math
import cv2
import numpy as np
def spherical_coordinates(i, j, w, h):
    """ Returns spherical coordinates of the pixel from the output image. """
    theta = 2*float(i)/float(w)-1
    phi = 2*float(j)/float(h)-1
    # phi = lat, theta = long
    return phi*(math.pi/2), theta*math.pi


def vector_coordinates(phi, theta):
    """ Returns 3D vector which points to the pixel location inside a sphere. """
    return (math.cos(phi) * math.cos(theta),  # X
            math.sin(phi),                    # Y
            math.cos(phi) * math.sin(theta))  # Z


# Assign identifiers to the faces of the cube
FACE_Z_POS = 1  # Left
FACE_Z_NEG = 2  # Right
FACE_Y_POS = 3  # Top
FACE_Y_NEG = 4  # Bottom
FACE_X_NEG = 5  # Front
FACE_X_POS = 6  # Back


def get_face(x, y, z):
    """ Uses 3D vector to find which cube face the pixel lies on. """
    largest_magnitude = max(abs(x), abs(y), abs(z))
    if largest_magnitude - abs(x) < 0.00001:
        return FACE_X_POS if x < 0 else FACE_X_NEG
    elif largest_magnitude - abs(y) < 0.00001:
        return FACE_Y_POS if y < 0 else FACE_Y_NEG
    elif largest_magnitude - abs(z) < 0.00001:
        return FACE_Z_POS if z < 0 else FACE_Z_NEG


def raw_face_coordinates(face, x, y, z):
    """
    Return coordinates with necessary sign (- or +) depending on which face they lie on.
    From Open-GL specification (chapter 3.8.10) https://www.opengl.org/registry/doc/glspec41.core.20100725.pdf
    """
    if face == FACE_X_NEG:
        xc = z
        yc = y
        ma = x
        return xc, yc, ma
    elif face == FACE_X_POS:
        xc = -z
        yc = y
        ma = x
        return xc, yc, ma
    elif face == FACE_Y_NEG:
        xc = z
        yc = -x
        ma = y
        return xc, yc, ma
    elif face == FACE_Y_POS:
        xc = z
        yc = x
        ma = y
        return xc, yc, ma
    elif face == FACE_Z_POS:
        xc = x
        yc = y
        ma = z
        return xc, yc, ma
    elif face == FACE_Z_NEG:
        xc = -x
        yc = y
        ma = z
        return xc, yc, ma


def raw_coordinates(xc, yc, ma):
    """ Return 2D coordinates on the specified face relative to the bottom-left corner of the face. Also from Open-GL spec."""
    return (float(xc)/abs(float(ma)) + 1) / 2, (float(yc)/abs(float(ma)) + 1) / 2


def face_origin_coordinates(face, n):
    """ Return bottom-left coordinate of specified face in the input image. """
    if face == FACE_X_NEG:
        return n, n
    elif face == FACE_X_POS:
        return 3*n, n
    elif face == FACE_Z_NEG:
        return 2*n, n
    elif face == FACE_Z_POS:
        return 0, n
    elif face == FACE_Y_POS:
        return 2*n, 0
    elif face == FACE_Y_NEG:
        return 2*n, 2*n


def normalized_coordinates(face, x, y, n):
    """ Return pixel coordinates in the input image where the specified pixel lies. """
    face_coords = face_origin_coordinates(face, n)
    normalized_x = math.floor(x*n)
    normalized_y = math.floor(y*n)

    # Stop out of bound behaviour
    if normalized_x < 0:
        normalized_x = 0
    elif normalized_x >= n:
        normalized_x = n-1
    if normalized_y < 0:
        normalized_x = 0
    elif normalized_y >= n:
        normalized_y = n-1

    return face_coords[0] + normalized_x, face_coords[1] + normalized_y


def find_corresponding_pixel(i, j, w, h, n):
    """
    :param i: X coordinate of output image pixel
    :param j: Y coordinate of output image pixel
    :param w: Width of output image
    :param h: Height of output image
    :param n: Height/Width of each square face
    :return: Pixel coordinates for the input image that a specified pixel in the output image maps to.
    """

    spherical = spherical_coordinates(i, j, w, h)
    vector_coords = vector_coordinates(spherical[0], spherical[1])
    face = get_face(vector_coords[0], vector_coords[1], vector_coords[2])
    raw_face_coords = raw_face_coordinates(face, vector_coords[0], vector_coords[1], vector_coords[2])

    cube_coords = raw_coordinates(raw_face_coords[0], raw_face_coords[1], raw_face_coords[2])

    return normalized_coordinates(face, cube_coords[0], cube_coords[1], n)

def CubemapToEquirectangular(inimg):
    #inimg=cv2.imread(infile)
    h,w,ch=inimg.shape

    height=int(w/3)
    width=int(2*height)
    n=int(h/3)
    outimg=np.zeros((height,width,ch),np.uint8)
      # For each pixel in output image find colour value from input image
    for ycoord in range(0, height-1):
        for xcoord in range(0, width-1):
            corrx, corry = find_corresponding_pixel(xcoord, ycoord, width, height, n)

            #outimg.putpixel((xcoord, ycoord), inimg.getpixel((corrx, corry)))
            outimg[ycoord,xcoord,0]= inimg[corry,corrx,0]
            outimg[ycoord,xcoord,1]= inimg[corry,corrx,1]
            outimg[ycoord,xcoord,2]= inimg[corry,corrx,2]
            #print(xcoord)
    outimg=cv2.cvtColor(outimg,cv2.COLOR_BGR2RGB)
    return outimg

#CubemapToEquirectangular('Factory_Catwalk_BgOut2.png')
def get_Cubemap_face(posx,posy,n):
    x=posx%(4*n)
    y=posy%(4*n)
    if x<n:
        return FACE_Z_POS
    elif x<2*n:
        return FACE_X_POS
    elif x<3*n:
        if y<n:
            return FACE_Y_POS
        elif y<2*n:
            return FACE_Z_NEG
        else:
            return FACE_Y_NEG
    else:
        return FACE_X_NEG

def find_Cubemap_corresponding_pixel(pos_x,pos_y,width,height,n):
    
    posx,posy=normalized_coordinates(get_Cubemap_face(pos_x,pos_y,n), pos_x, pos_y, n)
    for ycoord in range(0, height-1):
        for xcoord in range(0, width-1):
            corrx, corry = find_corresponding_pixel(xcoord, ycoord, width, height, n)
            if((pos_x-corrx)**2 + (pos_y-corry)**2<=100):
                #print('true')
                #print('called',pos_x,pos_y)
                #print('->',xcoord,ycoord)
                return xcoord,ycoord
            
            #outimg.putpixel((xcoord, ycoord), inimg.getpixel((corrx, corry)))
            #outimg[ycoord,xcoord,0]= inimg[corry,corrx,0]
            #outimg[ycoord,xcoord,1]= inimg[corry,corrx,1]
            #outimg[ycoord,xcoord,2]= inimg[corry,corrx,2]
            #print(xcoord)
    return 0,0