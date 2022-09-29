from numpy import all, asarray, array, where, exp
from pandas import DataFrame
from skimage.filters import gaussian, median
from skimage.morphology import cube
from skimage.feature import peak_local_max
from scipy.optimize import curve_fit
from scipy.stats import multivariate_normal
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

def compute(im, options):
    beads, maxima, centers, smoothed = getCenters(im, options)
    return [getPSF(x, options) for x in beads], beads, maxima, centers, smoothed

def inside(shape, center, window):
    """
    Returns boolean if a center and its window is fully contained
    within the shape of the image on all three axes
    """
    return all([(center[i]-window[i]//2 >= 0) & (center[i] + window[i]//2 <= shape[i]) for i in range(0,3)])

def volume(im, center, window):
    if inside(im.shape, center, window):
        volume = im[(center[0]-window[0]//2):(center[0]+window[0]//2), 
                    (center[1]-window[1]//2):(center[1]+window[1]//2), 
                    (center[2]-window[2]//2):(center[2]+window[2]//2)]
        volume = volume.astype('float64')
        baseline = volume[[0,-1],[0,-1],[0,-1]].mean()
        volume = volume - baseline
        volume = volume/volume.max()
        return volume

def findBeads(im, window, thresh, min_dist=1):
    '''
    Uses a 2D gaussian filter to smooth the data with a sigma of 1
    Finds peaks that are at least separated by min distance
    Smoothing can be 'median' (slow but more accurate) or 'gaussian' (fast but more agressive).

    Returns the max projection of the smoothed image
    '''
    #smoothed = median(im, cube(3), mode='reflect') # VERY slow
    smoothed = gaussian(im, 1, output=None, mode='nearest', truncate=1.0, preserve_range=True)
    centers = peak_local_max(smoothed, min_distance=min_dist, threshold_abs=thresh, exclude_border=True)
    print(f'findBeads() done: {len(centers)} found')
    return centers, smoothed.max(axis=0)

def keepBeads(im, window, centers, options):
    centersM = asarray([[x[0]/options['pxPerUmAx'], x[1]/options['pxPerUmLat'], x[2]/options['pxPerUmLat']] for x in centers])
    print('centersM done')
    #centerDists = [nearest(x,centersM) for x in centersM] # super-slow step! Modified, see below
    distance_matrix = pairwise_distances(centersM)
    distance_matrix.sort()
    centerDists = distance_matrix[:,1]
    print('centerDists done')
    #min_distance = sum([x**2 for x in options['windowUm']])**(.5)
    min_distance = float(min(options['windowUm']))
    keep = where([x > min_distance for x in centerDists])
    centers = centers[keep[0],:]
    keep = where([inside(im.shape, x, window) for x in centers])
    print(f'keepBeads() done: {len(keep[0])} found')
    return centers[keep[0],:]

def getCenters(im, options):
    window = [options['windowUm'][0]*options['pxPerUmAx'], options['windowUm'][1]*options['pxPerUmLat'], options['windowUm'][2]*options['pxPerUmLat']]
    window = [round(x) for x in window]
    centers, smoothed = findBeads(im, window, options['thresh'])
    centers = keepBeads(im, window, centers, options)
    beads = [volume(im, x, window) for x in centers]
    maxima = [im[x[0], x[1], x[2]] for x in centers]
    print(f'getCenters() done: {len(centers)} found')
    return beads, maxima, centers, smoothed

def getPSF(bead, options):
    latProfile, axProfile = getSlices(bead)
    latFit = fit(latProfile,options['pxPerUmLat'])
    axFit = fit(axProfile,options['pxPerUmAx'])
    data = DataFrame([latFit[3], axFit[3]],index = ['FWHMlat', 'FWHMax']).T
    return data, latFit, axFit

def getSlices(average):
    latProfile = (average.mean(axis=0).mean(axis=1) + average.mean(axis=0).mean(axis=1))/2
    axProfile = (average.mean(axis=1).mean(axis=1) + average.mean(axis=2).mean(axis=1))/2
    return latProfile, axProfile

def fit(yRaw,scale):
    y = yRaw - (yRaw[0]+yRaw[-1])/2
    x = (array(range(y.shape[0])) - y.shape[0]/2)
    x = (array(range(y.shape[0])) - y.shape[0]/2)
    popt, pcov = curve_fit(gauss, x, y, p0 = [1, 0, 1, 0])
    FWHM = 2.355*popt[2]/scale
    yFit = gauss(x, *popt)
    return x, y, yFit, FWHM

def plotPSF(x,y,yFit,FWHM,scale,Max):
    plt.plot(x.astype(float)/scale,yFit/yFit.max(), lw=2);
    plt.plot(x.astype(float)/scale,y/yFit.max(),'ok');
    plt.xlim([-x.shape[0]/2/scale, x.shape[0]/2/scale])
    plt.ylim([0, 1.1])
    plt.xlabel('Distance (um)')
    plt.ylabel('Norm. intensity')
    plt.annotate('FWHM %.2f um' % FWHM,xy=(x.shape[0]/4/scale, .6), size=14)
    plt.annotate('Brightness %.2f' % Max,xy=(x.shape[0]/4/scale, .5), size=14)


def plotAvg(i):
    plt.figure(figsize=(5,5));
    plt.imshow(average[i], vmin=0, vmax=.9);
    if i==average.shape[0]/2:
        plt.plot(average.shape[1]/2, average.shape[2]/2, 'r.', ms=10);
    plt.xlim([0, average.shape[1]])
    plt.ylim([average.shape[2], 0])
    plt.axis('off');

def plotAvg(i):
    plt.figure(figsize=(5,5));
    plt.imshow(average[i], vmin=0, vmax=.9);
    if i==average.shape[0]/2:
        plt.plot(average.shape[1]/2, average.shape[2]/2, 'r.', ms=10);
    plt.xlim([0, average.shape[1]])
    plt.ylim([average.shape[2], 0])
    plt.axis('off');

def dist(x,y):
    return ((x - y)**2)[1:].sum()**(.5)

def nearest(x,centers):
    z = [dist(x,y) for y in centers if not (x == y).all()]
    return abs(array(z)).min(axis=0)

def gauss(x, a, mu, sigma, b):
    return a*exp(-(x-mu)**2/(2*sigma**2))+b
