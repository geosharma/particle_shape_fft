# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Description: Determine the radius, r(theta), of a single particle
# at different inclination angles (theta)
# from a thresholded binary image and perform fft
# images are read from ../data folder and the output is placed in the
# ../plots folder

# import modulues
import numpy as np

# import operation system module os
import os

# import natsort module for natural human sorting
import natsort

# import module for image manipulation
import cv2

# import plotting module
import matplotlib.pyplot as plt

# shapely for intersection of the line with polygon (contour)
import shapely.geometry

# interpolation for smoothning the contour
import scipy.interpolate as scint

# linear algebra model from numpy
import numpy.linalg as la

# path for input files, change this path depending on where the
# files to be renamed are, if in the same directory as this script file
# then change to filepath = "./"
filepath = "../data/"
outfilepath = "../plots/"

# scale of the image, if 247 pixels = 1 mm
# scale = 1/247 mm/pixel
scale = 1.0

# name of the summary file
smryfilename = '00_fft_summary.csv'
smryfile = outfilepath + smryfilename

# read and store in array all the files in the path ending with
images = [f for f in os.listdir(filepath) if f.endswith('.png')]

# sort the files according to the numeral in the file name
images = natsort.natsorted(images)

# print the names of the sorted image names
print('Sorted filenames: ', images)

# write the input filenames and output filenames to stdio
# print('{0:30s} {1:30s}'.format('Input filenames', 'Cropped filenames'))

# generate 512 number of points from 0 to pi
angles = np.linspace(0, np.pi, num=64, endpoint=False)

# create array for storing angle values and radius
stang = np.zeros(2 * len(angles), dtype=np.float32)
strad = np.zeros(2 * len(angles), dtype=np.float32)

# format string for output heading and data
formatstrhead = "{0:10s}, {1:10s}, {2:15s}, {3:30s}"
formatstrdata = "{0:10.6f}, {1:10.3f}, {2:15.4f}, {3:30s}"

# warning messages
warningcontour = "WARNING: More than one countour found"

# open the summary file
sfile = open(smryfile, 'w')

# print the heading, A0 is the mean amplitude and FD are the Fourier
# descriptors
print("{0:15s},{1:7s},{2:7s},{3:7s},{4:7s},{5:7s},{6:7s},{7:7s},{8:7s},\
{9:7s},{10:10s}".format(
    "Filename", "A0", "FD1", "FD2", "FD3", "FD4", "FD5", "FD6", "FD7", "FD8",
    "1 - Norm"), file=sfile)

figscale = 0.75
# create figures and later clear axis and reuse again
# angle vs radius plot
fig = plt.figure(figsize=(figscale * 8, figscale * 6))

# figure to plot the fft
fig2 = plt.figure(figsize=(figscale * 8, figscale * 6))

for image in images:
    # file name without the extension
    imagenoext = os.path.splitext(image)[0]

    # read image file
    img = cv2.imread(filepath + image, cv2.IMREAD_GRAYSCALE)

    # apply Gaussian filter
    img_blr = cv2.GaussianBlur(img, (5, 5), 0)

    # threshold
    # set values equal to or above 127 to 0
    # set values below 220 to 255
    ret, img_th = cv2.threshold(img_blr, 127, 255, cv2.THRESH_BINARY_INV)

    # copy the thresholded image
    img_floodfill = img_th.copy()

    # fill holes present in the sand particle
    # mask used for flood filling
    # the size needs to be 2 pixels larger than the image
    h, w = img_th.shape
    mask = np.zeros((h+2, w+2), np.uint8)

    # floodfill from point(10, 50) to avoid the annotation
    cv2.floodFill(img_floodfill, mask, (10, 50), 255)

    # invert the floodfilled image
    img_floodfill_inv = cv2.bitwise_not(img_floodfill)

    # combine the two images to get the foreground with holes filled
    img_out = img_th | img_floodfill_inv

    # crop the image to remove the annotation
    img_crpd = img_out[40:430, 5:630]

    # create the contour around the edge of the particle
    im2, cnts, heir = cv2.findContours(img_crpd, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

    # there will be a single particle in an image, therefore
    # there should be only one contour
    if len(cnts) > 1:
        warning = warningcontour
    else:
        warning = ""

    # since the files are binary images of single particles, each image should
    # should only have one contour associated with it, only use the first
    # contour.
    cnt = cnts[0]

    # find the center of mass from the moments of the contours
    area = cv2.contourArea(cnt) * scale**2
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # fit an ellipse and get the orientation of the minor ellipse axis,
    rotatedrect = cv2.minAreaRect(cnt)

    # find the orientation of the major axis, all angles are with the particle
    # aligned along the major axis
    # find the correct reference for this
    if rotatedrect[1][0] < rotatedrect[1][1]:
        min_ellipse_ax_orien = np.radians(90.0 - rotatedrect[2])
    else:
        min_ellipse_ax_orien = np.radians(-rotatedrect[2])

    # extract the contour elements
    xcon = cnt[:, 0][:, 0]
    ycon = cnt[:, 0][:, 1]

    # smooth the contours, see if this can be done better
    tck, u = scint.splprep([xcon, ycon], k=1, s=2)
    u_new = np.linspace(u.min(), u.max(), np.int(cnt.shape[0]))
    xd, yd = scint.splev(u_new, tck)

    i = 0
    # to find the intersection points with a line drawn at a given angle
    # translate the contour to the origin, rotate the contour by the given
    # angle and find the extreme points along the x-axis as the
    # intersection points. A big assumption is that the particle is
    # convex and there area only two intersection points
    for theta in angles:

        # use the orientation of the minor ellipse axis from above and assign
        # the direction of the major ellipse axis as the starting point
        # the orientation of the particle major ellipse axis is horizontal
        angle = -min_ellipse_ax_orien + theta

        # translate to the origin (0, 0) and rotate the contour and
        # translate back to the center of mass
        xr = (xd - cX) * np.cos(angle) + (yd - cY) * np.sin(angle) + cX
        yr = -(xd - cX) * np.sin(angle) + (yd - cY) * np.cos(angle) + cY

        # the horizontal line, passing through the center of mass
        # most probably it would be more efficient to find the x-axis
        # intercepts after translating to the origin and rotating, but I
        # wanted to use shapely
        slope = np.tan(np.radians(0))
        intercept = cY - slope * cX
        pline = np.array([slope, intercept], dtype=np.float32)
        # longer horizonal line is required for intersection
        xline = np.linspace(xr.min() - 10, xr.max() + 10, endpoint=True)
        yline = np.polyval(pline, xline)

        # Use shapely to find the intersection of a line with a closed
        # polygon. Used this method because shapely was the recommended
        # method on online search, see if this can be done without using
        # shapely. Create shapely polygon from the rotated contour and
        # shapely line from the horizontal line
        shpoly = shapely.geometry.LinearRing(np.c_[xr, yr])
        shline = shapely.geometry.LineString(np.c_[xline, yline])
        points = shpoly.intersection(shline)

        # convert shapely returned points to numpy array for plotting
        shpx = np.asarray(shpoly)[:, 0]
        shpy = np.asarray(shpoly)[:, 1]
        shlx = np.asarray(shline)[:, 0]
        shly = np.asarray(shline)[:, 1]
        r1 = np.sqrt((points[0].x - cX)**2 + (points[0].y - cY)**2) * scale
        r2 = np.sqrt((points[1].x - cX)**2 + (points[1].y - cY)**2) * scale

        # store the results to the array
        stang[i] = theta
        strad[i] = r2
        # the intersection points are 180 degrees apart
        stang[len(angles) + i] = theta + np.pi
        strad[len(angles) + i] = r1
        i += 1

    # start the fft code
    # fft amplitude spectrum
    ps = np.abs(np.fft.fft(strad))
    freqs = np.fft.fftfreq(strad.size)
    idx = np.argsort(freqs)

    # print to the output file
    print("{0:15s},{1:7.6e},{2:7.6e},{3:7.6e},{4:7.6e},{5:7.6e},{6:7.6e},\
    {7:7.6e},{8:7.6e},{9:7.6e},{10:10.6e}".format(
        imagenoext, ps[0], ps[1]/ps[0], ps[2]/ps[0], ps[3]/ps[0], ps[4]/ps[0],
        ps[5]/ps[0], ps[6]/ps[0], ps[7]/ps[0], ps[8]/ps[0],
        1 - la.norm(ps[1:9]/ps[0])), file=sfile)

    # the name of the output file containing angles and radii
    outfilename = imagenoext + '_result' + '.csv'
    outfile = outfilepath + outfilename

    with open(outfile, 'w') as fout:
        print("Area [mm2]: ", area, file=fout)
        print("Equivalent area diameter [mm]: ",
              np.sqrt(4.0 * area/np.pi), file=fout)
        print("Center of mass [pixel]: ", cX, ",", cY, file=fout)
        print(formatstrhead.format("Radians", "Degrees", "Radius (mm)",
                                   "Comments"), file=fout)

        for i in range(len(stang)):
            print(formatstrdata.format(stang[i], np.degrees(stang[i]),
                                       strad[i], warning), file=fout)

    ax = fig.add_subplot(111)

    ax.plot(np.degrees(stang), strad)
    ax.set_xlabel(r"$\theta_i$ [deg]")
    ax.set_ylabel(r"r$\left( \theta_i \right)$ [mm]")
    ax.set_xlim(0, )
    # ax.set_ylim(top=1)
    # for the seconday x-axis showing corresponding degrees
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    # print(ax.get_xticks())
    ax2labels = np.radians(ax.get_xticks())
    ax2labels = np.round(ax2labels, 2)
    # print(ax2labels)
    # ax2.set_xticks(ax2tickpos)
    ax2.set_xticklabels(ax2labels)
    ax2.set_xlabel(r"$\theta_i$ [rad]")

    # plot the fft
    ax2 = fig2.add_subplot(111)
    ax2.plot(freqs[idx], ps[idx])
    ax2.set_xlim(-0.1, 0.1)
    ax2.set_xlabel(r"[1/rad]")
    ax2.set_ylabel(r"Amplitude")

    # save figures
    fig.savefig(outfilepath + imagenoext + "_plot" + '.png')
    fig2.savefig(outfilepath + imagenoext + "_fft" + '.png')

    # clear axis and prepare for another plot
    ax.cla()
    ax2.cla()

# close the summary file
sfile.close()
