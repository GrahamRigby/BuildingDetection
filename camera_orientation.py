import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D

# Paul LoBuglio
# 1002161741
# 4 lines provided 
l1 = [[1607, 3143], [2238, 2709]]
l2 = [[3632, 3346], [3593, 2781]]

l3 = [[3058, 2524], [3428, 2544]]
l4 = [[2525, 3235], [3624, 3349]]
# params
K = np.array([[3641.7, 0, 2320],
                [0, 3641.7, 1740],
                [0, 0, 1]])


# I calculated that a street going north to south in toronto's grid has a bearing of about
# 163.417 clockwise from north. Bearing was calculated by dropping pins on google maps
# on the same side of a street going north to south, then using #https://www.movable-type.co.uk/scripts/latlong.html
street_ns_bearing = 163.417
# We subtract this from 90 degrees to this, because in our cardinal coordinate system, east points to positive x
street_d1 = 90 - 163.417
# We then find the corresponding rotation vector. Named r2 because it transform the y axis in cardinal space (north)
# into the direction of a north to south street in toronto.
street_r2 = np.array([np.cos(street_d1 * np.pi / 180), np.sin(street_d1 * np.pi / 180)])

# I also found the bearing of streets going from west to east in toronto, and as expected it is roughly perpendicular
# to the streets going north to south
street_we_bearing = 73.5192
# However, for the convenience of having a orthogonal rotation matrix I assume they are perfectly perpendicular
# and find street_r1 by rotating street_r2 90 degrees CCW
street_r1 = np.array([-street_r2[1], street_r2[0]])

# This gives us or rotation from cardinal space (the space where axis 0 points east and axis 1 points north) to
# 'street space,' and vice versa
R_card_to_street = np.hstack((street_r1[np.newaxis].T, street_r2[np.newaxis].T))
# because R is orthogonal
R_street_to_card = np.vstack((street_r1, street_r2))

if __name__ == "__main__":
    # For the report, I draw my annotation lines on the image and write it for my report.
    img = cv2.imread("great_pic.jpg")
    lines_drawn = np.copy(img)
    cv2.line(lines_drawn, tuple(l1[0]), tuple(l1[1]), (0, 0, 255), 4)
    cv2.line(lines_drawn, tuple(l2[0]), tuple(l2[1]), (0, 0, 255), 4)
    cv2.line(lines_drawn, tuple(l3[0]), tuple(l3[1]), (0, 255, 0), 4)
    cv2.line(lines_drawn, tuple(l4[0]), tuple(l4[1]), (0, 255, 0), 4)
    cv2.imwrite("lines_drawn.jpg", lines_drawn)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    W, H = img_rgb.shape[1], img_rgb.shape[0]

    # Next, I find the equation of our given lines in standard form. We can find this by taking the cross product
    # of the two points on the line in homogenous coordinates. This give us line = [a, b, c] where ax + by + c = 0

    line1 = np.cross(l1[0] + [1], l1[1] + [1])
    line2 = np.cross(l2[0] + [1], l2[1] + [1])

    line3 = np.cross(l3[0] + [1], l3[1] + [1])
    line4 = np.cross(l4[0] + [1], l4[1] + [1])

    # Similarly, we can find the intersection of each pair of lines in homogenous coordinates by taking their cross
    # product. We then cartezianize to get the vanishing point in the image plane
    vz = np.cross(line1, line2)
    vz = vz / vz[2]

    vx = np.cross(line3, line4)
    vx = vx / vx[2]

    # ------------ Plot our lines and vanishing points ------------
    plt.imshow(img_rgb)
    plt.axis([-13000, W, H, 0])
    y_interp = np.arange(0, H, 1)
    x_interp1 = (-line1[2] - y_interp * line1[1]) / line1[0]
    x_interp2 = (-line2[2] - y_interp * line2[1]) / line2[0]

    x_interp3 = (-line3[2] - y_interp * line3[1]) / line3[0]
    x_interp4 = (-line4[2] - y_interp * line4[1]) / line4[0]

    plt.plot(x_interp1, y_interp, c='red')
    plt.plot(x_interp2, y_interp, c='red')

    plt.plot(x_interp3, y_interp, c='green')
    plt.plot(x_interp4, y_interp, c='green')

    c1 = plt.Circle((vz[0], vz[1]), 50, color='b')
    c2 = plt.Circle((vx[0], vx[1]), 50, color='b')
    ax = plt.gca()
    ax.add_artist(c1)
    ax.add_artist(c2)
    #plt.show()
    # ------------------- Now, we find our rotation vectors ------------------------
    r1 = np.dot(np.linalg.inv(K), vx)
    r1 = r1 / np.linalg.norm(r1)

    r3 = np.dot(np.linalg.inv(K), vz)
    r3 = r3 / np.linalg.norm(r3)

    r2 = np.cross(r1, r3)
    r2 = r2 / np.linalg.norm(r2)
    # ------------------- We plot our new basis vectors (in street space) ------------------------
    fig = plt.figure()
    ax3 = fig.add_subplot(111, projection='3d')
    X, Y, Z = [0] * 3, [0] * 3, [0] * 3
    U, V, W = zip(r1, r2, r3)
    U2, V2, W2 = [.5, 0, 0], [0, .5, 0], [0, 0, .5]
    # plot our camera orientation vectors and street space basis in purple for reference
    ax3.quiver(X, Y, Z, U, V, W, color=['red', 'green', 'blue'])
    ax3.quiver(X, Y, Z, U2, V2, W2, color=['purple', 'purple', 'purple'])
    ax3.set_xlim([-1, 1])
    ax3.set_ylim([-1, 1])
    ax3.set_zlim([-1, 1])

    # Since we primarily care about 2D orientation, we project r1 and r3 orthogonally onto the street world xz plane
    # and disregard r2.
    r1p = np.array([r1[0], r1[2]])
    r1p = r1p / np.linalg.norm(r1p)

    r3p = np.array([r3[0], r3[2]])
    r3p = r3p / np.linalg.norm(r3p)
    # This gives us 2 matricies: one that transform 'street east' and 'street south' to the camera's x and z axes
    # respectively (i.e our change of basis from street to camera space) , and its inverse which transforms camera
    # space into 2d space.
    R_street_to_camera = np.hstack((r1p[np.newaxis].T, r3p[np.newaxis].T))
    # Because of the proection, our basis vector's aren't necessarily orthogonal so we have to computer inverse
    R_camera_to_street = np.linalg.inv(R_street_to_camera)

    # From everything we have, we easily get our camera to cardinal transformation matrix.
    R_camera_to_card = np.dot(R_street_to_card, R_camera_to_street)
    # The columns of our resulting matrix are the x and z camera axes in cardinal space, which is exactly what we want.
    # But first we normalize these vectors
    R_camera_to_card[:, 0] = R_camera_to_card[:, 0] / np.linalg.norm(R_camera_to_card[:, 0])
    R_camera_to_card[:, 1] = R_camera_to_card[:, 1] / np.linalg.norm(R_camera_to_card[:, 1])
    # We now plot these resulting axes in cardinal space
    fig2 = plt.figure()
    ax2 = fig2.gca()
    x, y = [0] * 2, [0] * 2
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax2.quiver(x, y, R_camera_to_card[0, :], R_camera_to_card[1, :],
               color=['red', 'blue'], angles='xy', scale_units='xy', scale=1)

    plt.show()