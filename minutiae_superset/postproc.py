"""MinutiaeNet post-processing (verbatim, MIT-licensed) — TF-free standalone copy.

Extracted from external/MinutiaeNet/CoarseNet/{MinutiaeNet_utils,CoarseNet_model}.py
by scripts/_extract_postproc.py. Only numpy-2.x / skimage-0.26 compat patches applied
(np.bool->bool, np.int->int, np.lib.pad->np.pad, multichannel=False->channel_axis=None).

Original work: Dinh-Luan Nguyen, Kai Cao, Anil K. Jain, "Robust Minutiae Extractor:
Integrating Deep Networks and Fingerprint Domain Knowledge", ICB 2018.
https://github.com/luannd/MinutiaeNet  (MIT License).
"""
import math
import numpy as np
from scipy import ndimage, signal, spatial
from skimage.filters import gaussian
import cv2

def mnt_writer(mnt, image_name, image_size, file_name):
    f = open(file_name, 'w')
    f.write('%s\n'%(image_name))
    f.write('%d %d %d\n'%(mnt.shape[0], image_size[0], image_size[1]))
    for i in range(mnt.shape[0]):
        f.write('%d %d %.6f %.4f\n'%(mnt[i,0], mnt[i,1], mnt[i,2], mnt[i,3]))
    f.close()
    return


def angle_delta(A, B, max_D=np.pi*2):
    delta = np.abs(A - B)
    delta = np.minimum(delta, max_D-delta)
    return delta


def distance(y_true, y_pred, max_D=16, max_O=np.pi/6):
    D = spatial.distance.cdist(y_true[:, :2], y_pred[:, :2], 'euclidean')
    _ad = np.abs(np.reshape(y_true[:, 2], [-1, 1]) - np.reshape(y_pred[:, 2], [1, -1]))
    O = np.minimum(_ad, 2 * np.pi - _ad)  # vectorized angle_delta (scipy compat)
    return (D<=max_D)*(O<=max_O)


def nms(mnt):
    if mnt.shape[0]==0:
        return mnt
    # sort score
    mnt_sort = mnt.tolist()
    mnt_sort.sort(key=lambda x:x[3], reverse=True)
    mnt_sort = np.array(mnt_sort)
    # cal distance
    inrange = distance(mnt_sort, mnt_sort, max_D=16, max_O=np.pi/6).astype(np.float32)
    keep_list = np.ones(mnt_sort.shape[0])
    for i in range(mnt_sort.shape[0]):
        if keep_list[i] == 0:
            continue
        keep_list[i+1:] = keep_list[i+1:]*(1-inrange[i, i+1:])
    return mnt_sort[keep_list.astype(bool), :]


def fuse_nms(mnt, mnt_set_2):
    if mnt.shape[0]==0:
        return mnt
    # sort score
    all_mnt = np.concatenate((mnt, mnt_set_2))

    mnt_sort = all_mnt.tolist()
    mnt_sort.sort(key=lambda x:x[3], reverse=True)
    mnt_sort = np.array(mnt_sort)
    # cal distance
    inrange = distance(mnt_sort, mnt_sort, max_D=16, max_O=2*np.pi).astype(np.float32)
    keep_list = np.ones(mnt_sort.shape[0])
    for i in range(mnt_sort.shape[0]):
        if keep_list[i] == 0:
            continue
        keep_list[i+1:] = keep_list[i+1:]*(1-inrange[i, i+1:])
    return mnt_sort[keep_list.astype(bool), :]


def py_cpu_nms(det, thresh):
    if det.shape[0]==0:
        return det
    dets = det.tolist()
    dets.sort(key=lambda x:x[3], reverse=True)
    dets = np.array(dets)

    box_sz = 25
    x1 = np.reshape(dets[:,0],[-1,1]) -box_sz
    y1 = np.reshape(dets[:,1],[-1,1]) -box_sz
    x2 = np.reshape(dets[:,0],[-1,1]) +box_sz
    y2 = np.reshape(dets[:,1],[-1,1]) +box_sz
    scores = dets[:, 2]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep, :]


def FastEnhanceTexture(img,sigma=2.5,show=False):
    img = img.astype(np.float32)
    h, w = img.shape
    h2 = 2 ** nextpow2(h)
    w2 = 2 ** nextpow2(w)

    FFTsize = np.max([h2, w2])
    FFTsize = int(FFTsize)
    x, y = np.meshgrid(list(range(-FFTsize // 2, FFTsize // 2)), list(range(-FFTsize // 2, FFTsize // 2)))
    r = np.sqrt(x * x + y * y) + 0.0001
    r = r/FFTsize

    L = 1. / (1 + (2 * math.pi * r * sigma)** 4)
    img_low = LowpassFiltering(img, L)

    gradim1=  compute_gradient_norm(img)
    gradim1 = LowpassFiltering(gradim1,L)

    gradim2=  compute_gradient_norm(img_low)
    gradim2 = LowpassFiltering(gradim2,L)

    diff = gradim1-gradim2
    ar1 = np.abs(gradim1)
    diff[ar1>1] = diff[ar1>1]/ar1[ar1>1]
    diff[ar1 <= 1] = 0

    cmin = 0.3
    cmax = 0.7

    weight = (diff-cmin)/(cmax-cmin)
    weight[diff<cmin] = 0
    weight[diff>cmax] = 1


    u = weight * img_low + (1-weight)* img

    temp = img - u

    lim = 20

    temp1 = (temp + lim) * 255 / (2 * lim)

    temp1[temp1 < 0] = 0
    temp1[temp1 >255] = 255
    v = temp1
    if show:
        plt.imshow(v,cmap='gray')
        plt.show()
    return v


def compute_gradient_norm(input):
    input = input.astype(np.float32)

    Gx, Gy = np.gradient(input)
    out = np.sqrt(Gx * Gx + Gy * Gy) + 0.000001
    return out


def LowpassFiltering(img,L):
    h,w = img.shape
    h2,w2 = L.shape

    img = cv2.copyMakeBorder(img, 0, h2-h, 0, w2-w, cv2.BORDER_CONSTANT, value=0)

    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)

    img_fft = img_fft * L
    rec_img = np.fft.ifft2(np.fft.fftshift(img_fft))
    rec_img = np.real(rec_img)
    rec_img = rec_img[:h,:w]

    return rec_img


def nextpow2(x):
    return int(math.ceil(math.log(x, 2)))


def get_maps_STFT(img,patch_size = 64,block_size = 16, preprocess = False):
    assert len(img.shape) == 2

    nrof_dirs = 16
    ovp_size = (patch_size-block_size)//2
    if preprocess:
        img = FastEnhanceTexture(img, sigma=2.5, show=False)

    img = np.pad(img, (ovp_size,ovp_size),'symmetric')
    h,w = img.shape
    blkH = (h - patch_size)//block_size+1
    blkW = (w - patch_size)//block_size+1
    local_info = np.empty((blkH,blkW),dtype = object)

    x, y = np.meshgrid(list(range(-patch_size // 2,patch_size // 2)), list(range(-patch_size // 2,patch_size // 2)))
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    r = np.sqrt(x*x + y*y) + 0.0001


    RMIN = 3  # min allowable ridge spacing
    RMAX = 18 # maximum allowable ridge spacing
    FLOW = patch_size / RMAX
    FHIGH = patch_size / RMIN
    dRLow = 1. / (1 + (r / FHIGH) ** 4)
    dRHigh = 1. / (1 + (FLOW / r) ** 4)
    dBPass = dRLow * dRHigh  # bandpass

    dir = np.arctan2(y,x)
    dir[dir<0] = dir[dir<0] + math.pi
    dir_ind = np.floor(dir/(math.pi/nrof_dirs))
    dir_ind = dir_ind.astype(int,copy=False)
    dir_ind[dir_ind==nrof_dirs] = 0


    dir_ind_list = []
    for i in range(nrof_dirs):
        tmp = np.argwhere(dir_ind == i)
        dir_ind_list.append(tmp)


    sigma = patch_size/3
    weight = np.exp(-(x*x + y*y)/(sigma*sigma))


    for i in range(0,blkH):
        for j in range(0,blkW):
            patch =img[i*block_size:i*block_size+patch_size,j*block_size:j*block_size+patch_size].copy()
            local_info[i,j] = local_STFT(patch,weight,dBPass)
            local_info[i, j].analysis(r,dir_ind_list)


    # get the ridge flow from the local information
    dir_map,fre_map = get_ridge_flow_top(local_info)
    dir_map = smooth_dir_map(dir_map)

    return dir_map, fre_map


def smooth_dir_map(dir_map,sigma=2.0,mask = None):

    cos2Theta = np.cos(dir_map * 2)
    sin2Theta = np.sin(dir_map * 2)
    if mask is not None:
        assert (dir_map.shape[0] == mask.shape[0])
        assert (dir_map.shape[1] == mask.shape[1])
        cos2Theta[mask == 0] = 0
        sin2Theta[mask == 0] = 0

    cos2Theta = gaussian(cos2Theta, sigma, channel_axis=None, mode='reflect')
    sin2Theta = gaussian(sin2Theta, sigma, channel_axis=None, mode='reflect')

    dir_map = np.arctan2(sin2Theta,cos2Theta)*0.5


    return dir_map


def get_ridge_flow_top(local_info):

    blkH,blkW = local_info.shape
    dir_map = np.zeros((blkH,blkW)) - 10
    fre_map = np.zeros((blkH, blkW)) - 10
    for i in range(blkH):
        for j in range(blkW):
            if local_info[i,j].ori is None:
                continue

            dir_map[i,j] = local_info[i,j].ori[0] #+ math.pi*0.5
            fre_map[i,j] = local_info[i,j].fre[0]
    return dir_map,fre_map


class local_STFT:
    def __init__(self,patch,weight = None, dBPass = None):


        if weight is not None:
            patch = patch * weight
        patch = patch - np.mean(patch)
        norm = np.linalg.norm(patch)
        patch = patch / (norm+0.000001)

        f = np.fft.fft2(patch)
        fshift = np.fft.fftshift(f)
        if dBPass is not None:
            fshift = dBPass * fshift

        self.patch_FFT = fshift
        self.patch = patch
        self.ori = None
        self.fre = None
        self.confidence = None
        self.patch_size = patch.shape[0]

    def analysis(self,r,dir_ind_list=None,N=2):

        assert(dir_ind_list is not None)
        energy = np.abs(self.patch_FFT)
        energy = energy / (np.sum(energy)+0.00001)
        nrof_dirs = len(dir_ind_list)

        ori_interval = math.pi/nrof_dirs
        ori_interval2 = ori_interval/2


        pad_size = 1
        dir_norm = np.zeros((nrof_dirs + 2,))
        for i in range(nrof_dirs):
            tmp = energy[dir_ind_list[i][:, 0], dir_ind_list[i][:, 1]]
            dir_norm[i + 1] = np.sum(tmp)

        dir_norm[0] = dir_norm[nrof_dirs]
        dir_norm[nrof_dirs + 1] = dir_norm[1]

        # smooth dir_norm
        smoothed_dir_norm = dir_norm
        for i in range(1, nrof_dirs + 1):
            smoothed_dir_norm[i] = (dir_norm[i - 1] + dir_norm[i] * 4 + dir_norm[i + 1]) / 6

        smoothed_dir_norm[0] = smoothed_dir_norm[nrof_dirs]
        smoothed_dir_norm[nrof_dirs + 1] = smoothed_dir_norm[1]

        den = np.sum(smoothed_dir_norm[1:nrof_dirs + 1]) + 0.00001  # verify if den == 1
        smoothed_dir_norm = smoothed_dir_norm/den  # normalization if den == 1, this line can be removed

        ori = []
        fre = []
        confidence = []

        wenergy = energy*r
        for i in range(1, nrof_dirs+1):
            if smoothed_dir_norm[i] > smoothed_dir_norm[i-1] and smoothed_dir_norm[i] > smoothed_dir_norm[i+1]:
                tmp_ori = (i-pad_size)*ori_interval + ori_interval2 + math.pi/2
                ori.append(tmp_ori)
                confidence.append(smoothed_dir_norm[i])
                tmp_fre = np.sum(wenergy[dir_ind_list[i-pad_size][:, 0], dir_ind_list[i-pad_size][:, 1]])/dir_norm[i]
                tmp_fre = 1/(tmp_fre+0.00001)
                fre.append(tmp_fre)


        if len(confidence)>0:
            confidence = np.asarray(confidence)
            fre = np.asarray(fre)
            ori = np.asarray(ori)
            ind = confidence.argsort()[::-1]
            confidence = confidence[ind]
            fre = fre[ind]
            ori = ori[ind]
            if len(confidence) >= 2 and confidence[0]/confidence[1]>2.0:

                self.ori = [ori[0]]
                self.fre = [fre[0]]
                self.confidence = [confidence[0]]
            elif len(confidence)>N:
                fre = fre[:N]
                ori = ori[:N]
                confidence = confidence[:N]
                self.ori = ori
                self.fre = fre
                self.confidence = confidence
            else:
                self.ori = ori
                self.fre = fre
                self.confidence = confidence

    def get_features_of_topN(self,N=2):
        if self.confidence is None:
            self.border_wave = None
            return
        candi_num = len(self.ori)
        candi_num = np.min([candi_num,N])
        patch_size = self.patch_FFT.shape
        for i in range(candi_num):

            kernel = gabor_kernel(self.fre[i], theta=self.ori[i], sigma_x=10, sigma_y=10)

            kernel_f = np.fft.fft2(kernel.real, patch_size)
            kernel_f = np.fft.fftshift(kernel_f)
            patch_f = self.patch_FFT * kernel_f

            patch_f = np.fft.ifftshift(patch_f)  # *np.sqrt(np.abs(fshift)))
            rec_patch = np.real(np.fft.ifft2(patch_f))


            plt.subplot(121), plt.imshow(self.patch, cmap='gray')
            plt.title('Input patch'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(rec_patch, cmap='gray')
            plt.title('filtered patch'), plt.xticks([]), plt.yticks([])
            plt.show()

    def reconstruction(self,weight=None):
        f_ifft = np.fft.ifftshift(self.patch_FFT)  # *np.sqrt(np.abs(fshift)))
        rec_patch = np.real(np.fft.ifft2(f_ifft))
        if weight is not None:
            rec_patch = rec_patch * weight
        return rec_patch

    def gabor_filtering(self,theta,fre,weight=None):

        patch_size = self.patch_FFT.shape
        kernel = gabor_kernel(fre, theta=theta,sigma_x=4,sigma_y=4)

        f = kernel.real
        f = f - np.mean(f)
        f = f / (np.linalg.norm(f)+0.0001)


        kernel_f = np.fft.fft2(f,patch_size)
        kernel_f = np.fft.fftshift(kernel_f)
        patch_f = self.patch_FFT*kernel_f

        patch_f = np.fft.ifftshift(patch_f)  # *np.sqrt(np.abs(fshift)))
        rec_patch = np.real(np.fft.ifft2(patch_f))
        if weight is not None:
            rec_patch = rec_patch * weight
        return rec_patch


def fuse_minu_orientation(dir_map, mnt, mode=1,block_size=16):
    # mode is the way to fuse output minutiae with orientation
    # 1: use orientation; 2: use minutiae; 3: fuse average
    blkH, blkW = dir_map.shape
    dir_map = dir_map%(2*np.pi)

    if mode == 1:
        for k in range(mnt.shape[0]):
            # Choose nearest orientation
            ori_value = dir_map[int(mnt[k, 1]//block_size),int(mnt[k, 0]//block_size)]
            if 0 < mnt[k, 2] and mnt[k, 2] <= np.pi/2:
                if 0 < ori_value and ori_value <= np.pi / 2:
                    mnt[k, 2] = ori_value
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    if (ori_value - mnt[k, 2]) < (np.pi - ori_value + mnt[k, 2]):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value + np.pi
                if np.pi < ori_value and ori_value <= 3*np.pi/2:
                    mnt[k, 2] = ori_value - np.pi
                if 3*np.pi/2 < ori_value and ori_value <= 2 * np.pi:
                    if (np.pi*2 - ori_value + mnt[k, 2]) < (ori_value - np.pi - mnt[k, 2]):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value - np.pi
            if np.pi/2 < mnt[k, 2] and mnt[k, 2] <= np.pi:
                if 0 < ori_value and ori_value <= np.pi / 2:
                    if (mnt[k, 2] - ori_value) < (np.pi - ori_value + mnt[k, 2]):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value + np.pi
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    mnt[k, 2] = ori_value
                if np.pi < ori_value and ori_value <= 3*np.pi/2:
                    if (ori_value - mnt[k, 2]) < (mnt[k, 2] - ori_value + np.pi):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value - np.pi
                if 3*np.pi/2 < ori_value and ori_value <= 2 * np.pi:
                    mnt[k, 2] = ori_value - np.pi
            if np.pi < mnt[k, 2] and mnt[k, 2] <= 3*np.pi/2:
                if 0 < ori_value and ori_value <= np.pi / 2:
                    mnt[k, 2] = ori_value + np.pi
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    if (mnt[k, 2] - ori_value) < (ori_value + np.pi - mnt[k, 2]):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value + np.pi
                if np.pi < ori_value and ori_value <= 3*np.pi/2:
                    mnt[k, 2] = ori_value
                if 3*np.pi/2 < ori_value and ori_value <= 2 * np.pi:
                    if (ori_value - mnt[k, 2]) < (mnt[k, 2] - ori_value + np.pi):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value - np.pi
            if 3*np.pi/2 < mnt[k, 2] and mnt[k, 2] <= 2*np.pi:
                if 0 < ori_value and ori_value <= np.pi / 2:
                    if (np.pi - mnt[k, 2] + ori_value) < (mnt[k, 2] - np.pi - ori_value):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value + np.pi
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    mnt[k, 2] = ori_value + np.pi
                if np.pi < ori_value and ori_value <= 3*np.pi/2:
                    if (mnt[k, 2] - ori_value) < (np.pi*2 - mnt[k, 2] + ori_value - np.pi):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value - np.pi
                if 3*np.pi/2 < ori_value and ori_value <= 2 * np.pi:
                    mnt[k, 2] = ori_value


    elif mode == 2:
        return
    elif mode ==3:
        for k in range(mnt.shape[0]):
            # Choose nearest orientation

            ori_value = dir_map[int(mnt[k, 1] // block_size), int(mnt[k, 0] // block_size)]
            if 0 < mnt[k, 2] and mnt[k, 2] <= np.pi / 2:
                if 0 < ori_value and ori_value <= np.pi / 2:
                    fixed_ori = ori_value
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    if (ori_value - mnt[k, 2]) < (np.pi - ori_value + mnt[k, 2]):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value + np.pi
                if np.pi < ori_value and ori_value <= 3 * np.pi / 2:
                    fixed_ori = ori_value - np.pi
                if 3 * np.pi / 2 < ori_value and ori_value <= 2 * np.pi:
                    if (np.pi * 2 - ori_value + mnt[k, 2]) < (ori_value - np.pi - mnt[k, 2]):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value - np.pi
            if np.pi / 2 < mnt[k, 2] and mnt[k, 2] <= np.pi:
                if 0 < ori_value and ori_value <= np.pi / 2:
                    if (mnt[k, 2] - ori_value) < (np.pi - ori_value + mnt[k, 2]):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value + np.pi
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    fixed_ori = ori_value
                if np.pi < ori_value and ori_value <= 3 * np.pi / 2:
                    if (ori_value - mnt[k, 2]) < (mnt[k, 2] - ori_value + np.pi):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value - np.pi
                if 3 * np.pi / 2 < ori_value and ori_value <= 2 * np.pi:
                    fixed_ori = ori_value - np.pi
            if np.pi < mnt[k, 2] and mnt[k, 2] <= 3 * np.pi / 2:
                if 0 < ori_value and ori_value <= np.pi / 2:
                    fixed_ori = ori_value + np.pi
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    if (mnt[k, 2] - ori_value) < (ori_value + np.pi - mnt[k, 2]):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value + np.pi
                if np.pi < ori_value and ori_value <= 3 * np.pi / 2:
                    fixed_ori = ori_value
                if 3 * np.pi / 2 < ori_value and ori_value <= 2 * np.pi:
                    if (ori_value - mnt[k, 2]) < (mnt[k, 2] - ori_value + np.pi):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value - np.pi
            if 3 * np.pi / 2 < mnt[k, 2] and mnt[k, 2] <= 2 * np.pi:
                if 0 < ori_value and ori_value <= np.pi / 2:
                    if (np.pi - mnt[k, 2] + ori_value) < (mnt[k, 2] - np.pi - ori_value):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value + np.pi
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    fixed_ori = ori_value + np.pi
                if np.pi < ori_value and ori_value <= 3 * np.pi / 2:
                    if (mnt[k, 2] - ori_value) < (np.pi * 2 - mnt[k, 2] + ori_value - np.pi):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value - np.pi
                if 3 * np.pi / 2 < ori_value and ori_value <= 2 * np.pi:
                    fixed_ori = ori_value

            mnt[k, 2] = (mnt[k, 2] + fixed_ori)/2.0
    else:
        return
