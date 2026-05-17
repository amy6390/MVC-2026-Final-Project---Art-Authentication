import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import scipy

def convert_to_hsi(img_path):
    img=io.imread(img_path)
    converted_img=color.rgb2hsv(img)
    h, s=converted_img[:, :, 0], converted_img[:, :, 1]
    i=np.average(img, axis=2) #i=(R+G+B)/3
    i=i/256 #scaled to values between 0 and 255

    '''
    plt.imshow(h, cmap='gray')
    plt.title("Hue Distribution")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    plt.imshow(s, cmap='gray')
    plt.title("Saturation Distribution")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    plt.imshow(i, cmap='gray')
    plt.title("Intensity Distribution")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    '''
    return [h, s, i]

def calculate_first_derivatives(arr):
    fx=np.gradient(arr, axis=1)
    fy=np.gradient(arr, axis=0)

    return (fx, fy)

def calculate_curvature(fx, fy, fxx, fyy, fxy):
    #k(x, y)=-(fxx*fy^2 + fyy*fx^2 - 2*fxy*fx*fy)/(fx^2+fy^2)^1.5
    epsilon = 1e-8
    denom = (fx**2 + fy**2)**1.5
    k=-(fxx*fy**2+fyy*fx**2-2*fxy*fx*fy)/(denom+epsilon)
    return k

def smooth_image(vals):
    smoothed=scipy.ndimage.gaussian_filter(vals, sigma=3.5)
    '''
    plt.imshow(smoothed, cmap='gray')
    plt.title("Smoothed Image")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    '''
    return smoothed

def find_statistics(vals):
    mean=np.mean(vals)
    std_dev=np.std(vals)
    
    lower_bound, upper_bound=mean-std_dev, mean+std_dev
    tail_entries=(vals<lower_bound)|(vals > upper_bound)
    percentage=np.count_nonzero(tail_entries)/(vals.shape[0]*vals.shape[1]) #number of tail entries/total number of entries

    return (mean, percentage)

vangogh_names=['Skull', 'Starry Night', 'The Irises', 'Burning Cigarette', 'Sunflowers', 'The Potato Eaters', "Van Gogh's Chair", 'Cafe Terrace']
with open('Curvature - Real Painting Features.txt', 'w') as f:
    for name in vangogh_names:
        vals=convert_to_hsi(name+'.jpg')
        vector=[]
        for i in range(len(vals)):
            s=vals[i]
            fx, fy=calculate_first_derivatives(s)

            smoothed=smooth_image(s)
            fx_smoothed, fy_smoothed=calculate_first_derivatives(smoothed)
            fxx, fxy=calculate_first_derivatives(fx_smoothed)
            tmp, fyy=calculate_first_derivatives(fy_smoothed)
            k=calculate_curvature(fx_smoothed, fy_smoothed, fxx, fyy, fxy)

            all_arrs=[s, fx, fy, k]
            for i in range(len(all_arrs)):
                mean, p=find_statistics(all_arrs[i])
                vector.append(mean)
                vector.append(p)
        f.write(' '.join([str(x) for x in vector]))
        f.write('\n')

ai_names=['AI Skull', 'AI Starry Night', 'AI Irises', 'AI Skeleton', 'AI Cafe', 'AI House', 'AI Sunflower']
with open('Curvature - AI Painting Features.txt', 'w') as f:
    for name in ai_names:
        vals=convert_to_hsi(name+'.jpg')
        vector=[]
        for i in range(len(vals)):
            s=vals[i]
            fx, fy=calculate_first_derivatives(s)

            smoothed=smooth_image(s)
            fx_smoothed, fy_smoothed=calculate_first_derivatives(smoothed)
            fxx, fxy=calculate_first_derivatives(fx_smoothed)
            tmp, fyy=calculate_first_derivatives(fy_smoothed)
            k=calculate_curvature(fx_smoothed, fy_smoothed, fxx, fyy, fxy)

            all_arrs=[smoothed, fx, fy, k]
            for i in range(len(all_arrs)):
                mean, p=find_statistics(all_arrs[i])
                vector.append(mean)
                vector.append(p)
        f.write(' '.join([str(x) for x in vector]))
        f.write('\n')

#plt.imshow(np.abs(k), cmap='gray', vmin=0, vmax=np.max(k))
#plt.show()
#u, tail=find_statistics(i)
