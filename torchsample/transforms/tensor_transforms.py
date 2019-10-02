
import os
import random
import math
import numpy as np

import torch as th
from torch.autograd import Variable

from ..utils import th_random_choice

class Compose(object):
    """
    Composes several transforms together.
    """
    def __init__(self, transforms):
        """
        Composes (chains) several transforms together into
        a single transform

        Arguments
        ---------
        transforms : a list of transforms
            transforms will be applied sequentially
        """
        self.transforms = transforms

    def __call__(self, *inputs):
        for transform in self.transforms:
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
            inputs = transform(*inputs)
        return inputs


class RandomChoiceCompose(object):
    """
    Randomly choose to apply one transform from a collection of transforms

    e.g. to randomly apply EITHER 0-1 or -1-1 normalization to an input:
        >>> transform = RandomChoiceCompose([RangeNormalize(0,1),
                                             RangeNormalize(-1,1)])
        >>> x_norm = transform(x) # only one of the two normalizations is applied
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *inputs):
        tform = random.choice(self.transforms)
        outputs = tform(*inputs)
        return outputs


class ToTensor(object):
    """
    Converts a numpy array to torch.Tensor
    """
    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _input = th.from_numpy(_input)
            outputs.append(_input)
        return outputs if idx >= 1 else outputs[0]


class ToVariable(object):
    """
    Converts a torch.Tensor to autograd.Variable
    """
    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _input = Variable(_input)
            outputs.append(_input)
        return outputs if idx >= 1 else outputs[0]


class ToCuda(object):
    """
    Moves an autograd.Variable to the GPU
    """
    def __init__(self, device=0):
        """
        Moves an autograd.Variable to the GPU

        Arguments
        ---------
        device : integer
            which GPU device to put the input(s) on
        """
        self.device = device

    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _input = _input.cuda(self.device)
            outputs.append(_input)
        return outputs if idx >= 1 else outputs[0]


class ToFile(object):
    """
    Saves an image to file. Useful as a pass-through ransform
    when wanting to observe how augmentation affects the data

    NOTE: Only supports saving to Numpy currently
    """
    def __init__(self, root):
        """
        Saves an image to file. Useful as a pass-through ransform
        when wanting to observe how augmentation affects the data

        NOTE: Only supports saving to Numpy currently

        Arguments
        ---------
        root : string
            path to main directory in which images will be saved
        """
        if root.startswith('~'):
            root = os.path.expanduser(root)
        self.root = root
        self.counter = 0

    def __call__(self, *inputs):
        for idx, _input in inputs:
            fpath = os.path.join(self.root, 'img_%i_%i.npy'%(self.counter, idx))
            np.save(fpath, _input.numpy())
        self.counter += 1
        return inputs


class ChannelsLast(object):
    """
    Transposes a tensor so that the channel dim is last
    `HWC` and `DHWC` are aliases for this transform.    
    """
    def __init__(self, safe_check=False):
        """
        Transposes a tensor so that the channel dim is last
        `HWC` and `DHWC` are aliases for this transform.

        Arguments
        ---------
        safe_check : boolean
            if true, will check if channels are already last and, if so,
            will just return the inputs
        """
        self.safe_check = safe_check

    def __call__(self, *inputs):
        ndim = inputs[0].dim()
        if self.safe_check:
            # check if channels are already last
            if inputs[0].size(-1) < inputs[0].size(0):
                return inputs
        plist = list(range(1,ndim))+[0]

        outputs = []
        for idx, _input in enumerate(inputs):
            _input = _input.permute(*plist)
            outputs.append(_input)
        return outputs if idx >= 1 else outputs[0]

HWC = ChannelsLast
DHWC = ChannelsLast

class ChannelsFirst(object):
    """
    Transposes a tensor so that the channel dim is first.
    `CHW` and `CDHW` are aliases for this transform.
    """
    def __init__(self, safe_check=False):
        """
        Transposes a tensor so that the channel dim is first.
        `CHW` and `CDHW` are aliases for this transform.

        Arguments
        ---------
        safe_check : boolean
            if true, will check if channels are already last and, if so,
            will just return the inputs
        """
        self.safe_check = safe_check

    def __call__(self, *inputs):
        ndim = inputs[0].dim()
        if self.safe_check:
            # check if channels are already first
            if inputs[0].size(0) < inputs[0].size(-1):
                return inputs
        plist = [ndim-1] + list(range(0,ndim-1))

        outputs = []
        for idx, _input in enumerate(inputs):
            _input = _input.permute(*plist)
            outputs.append(_input)
        return outputs if idx >= 1 else outputs[0]

CHW = ChannelsFirst
CDHW = ChannelsFirst

class TypeCast(object):
    """
    Cast a torch.Tensor to a different type
    """
    def __init__(self, dtype='float'):
        """
        Cast a torch.Tensor to a different type

        Arguments
        ---------
        dtype : string or torch.*Tensor literal or list of such
            data type to which input(s) will be cast.
            If list, it should be the same length as inputs.
        """
        if isinstance(dtype, (list,tuple)):
            dtypes = []
            for dt in dtype:
                if isinstance(dt, str):
                    if dt == 'byte':
                        dt = th.ByteTensor
                    elif dt == 'double':
                        dt = th.DoubleTensor
                    elif dt == 'float':
                        dt = th.FloatTensor
                    elif dt == 'int':
                        dt = th.IntTensor
                    elif dt == 'long':
                        dt = th.LongTensor
                    elif dt == 'short':
                        dt = th.ShortTensor
                dtypes.append(dt)
            self.dtype = dtypes
        else:
            if isinstance(dtype, str):
                if dtype == 'byte':
                    dtype = th.ByteTensor
                elif dtype == 'double':
                    dtype = th.DoubleTensor
                elif dtype == 'float':
                    dtype = th.FloatTensor
                elif dtype == 'int':
                    dtype = th.IntTensor
                elif dtype == 'long':
                    dtype = th.LongTensor
                elif dtype == 'short':
                    dtype = th.ShortTensor
            self.dtype = dtype

    def __call__(self, *inputs):
        if not isinstance(self.dtype, (tuple,list)):
            dtypes = [self.dtype]*len(inputs)
        else:
            dtypes = self.dtype
        
        outputs = []
        for idx, _input in enumerate(inputs):
            _input = _input.type(dtypes[idx])
            outputs.append(_input)
        return outputs if idx >= 1 else outputs[0]


class AddChannel(object):
    """
    Adds a dummy channel to an image. 
    This will make an image of size (28, 28) to now be
    of size (1, 28, 28), for example.
    """
    def __init__(self, axis=0):
        """
        Adds a dummy channel to an image, also known as
        expanding an axis or unsqueezing a dim

        Arguments
        ---------
        axis : integer
            dimension to be expanded to singleton
        """
        self.axis = axis

    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _input = _input.unsqueeze(self.axis)
            outputs.append(_input)
        return outputs if idx >= 1 else outputs[0]

ExpandAxis = AddChannel
Unsqueeze = AddChannel

class Transpose(object):

    def __init__(self, dim1, dim2):
        """
        Swaps two dimensions of a tensor

        Arguments
        ---------
        dim1 : integer
            first dim to switch
        dim2 : integer
            second dim to switch
        """
        self.dim1 = dim1
        self.dim2 = dim2

    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _input = th.transpose(_input, self.dim1, self.dim2)
            outputs.append(_input)
        return outputs if idx >= 1 else outputs[0]


class RangeNormalize(object):
    """
    Given min_val: (R, G, B) and max_val: (R,G,B),
    will normalize each channel of the th.*Tensor to
    the provided min and max values.

    Works by calculating :
        a = (max'-min')/(max-min)
        b = max' - a * max
        new_value = a * value + b
    where min' & max' are given values, 
    and min & max are observed min/max for each channel
    
    Arguments
    ---------
    min_range : float or integer
        Min value to which tensors will be normalized
    max_range : float or integer
        Max value to which tensors will be normalized
    fixed_min : float or integer
        Give this value if every sample has the same min (max) and 
        you know for sure what it is. For instance, if you
        have an image then you know the min value will be 0 and the
        max value will be 255. Otherwise, the min/max value will be
        calculated for each individual sample and this will decrease
        speed. Dont use this if each sample has a different min/max.
    fixed_max :float or integer
        See above

    Example:
        >>> x = th.rand(3,5,5)
        >>> rn = RangeNormalize((0,0,10),(1,1,11))
        >>> x_norm = rn(x)

    Also works with just one value for min/max:
        >>> x = th.rand(3,5,5)
        >>> rn = RangeNormalize(0,1)
        >>> x_norm = rn(x)
    """
    def __init__(self, 
                 min_val, 
                 max_val):
        """
        Normalize a tensor between a min and max value

        Arguments
        ---------
        min_val : float
            lower bound of normalized tensor
        max_val : float
            upper bound of normalized tensor
        """
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _min_val = _input.min()
            _max_val = _input.max()
            a = (self.max_val - self.min_val) / (_max_val - _min_val)
            b = self.max_val- a * _max_val
            _input = _input.mul(a).add(b)
            outputs.append(_input)
        return outputs if idx >= 1 else outputs[0]


class StdNormalize(object):
    """
    Normalize torch tensor to have zero mean and unit std deviation
    """
    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _input = _input.sub(_input.mean()).div(_input.std())
            outputs.append(_input)
        return outputs if idx >= 1 else outputs[0]


class Slice2D(object):

    def __init__(self, axis=0, reject_zeros=False):
        """
        Take a random 2D slice from a 3D image along 
        a given axis. This image should not have a 4th channel dim.

        Arguments
        ---------
        axis : integer in {0, 1, 2}
            the axis on which to take slices

        reject_zeros : boolean
            whether to reject slices that are all zeros
        """
        self.axis = axis
        self.reject_zeros = reject_zeros

    def __call__(self, x, y=None):
        while True:
            keep_slice  = random.randint(0,x.size(self.axis)-1)
            if self.axis == 0:
                slice_x = x[keep_slice,:,:]
                if y is not None:
                    slice_y = y[keep_slice,:,:]
            elif self.axis == 1:
                slice_x = x[:,keep_slice,:]
                if y is not None:
                    slice_y = y[:,keep_slice,:]
            elif self.axis == 2:
                slice_x = x[:,:,keep_slice]
                if y is not None:
                    slice_y = y[:,:,keep_slice]

            if not self.reject_zeros:
                break
            else:
                if y is not None and th.sum(slice_y) > 0:
                        break
                elif th.sum(slice_x) > 0:
                        break
        if y is not None:
            return slice_x, slice_y
        else:
            return slice_x


class RandomCrop(object):

    def __init__(self, size, channels_first=False):
        """
        Randomly crop a torch tensor

        Arguments
        --------
        size : tuple or list
            dimensions of the crop

        channels_first : boolean [False]
            if input contains more dimensions that self.size (e.g. multi-channel input), this
            specifies wheter to use the pad the last dimensions (True) or first dimensions (False).
            Has no effect when len(size) == len(input.shape)
        """
        self.size = size
        self.channels_first = channels_first

    def __call__(self, *inputs):
        x_size = inputs[0].size()
        n_dim = len(x_size)
        n_size_dim = len(self.size)

        if n_size_dim < n_dim:
          shape_slice = slice(-n_size_dim, None) if self.channels_first else slice(None, n_size_dim)
        else:
          assert len(inputs[0].shape) == n_size_dim, 'Input has less dimensions than specified size.'
          shape_slice = slice(None)

        def_shape = x_size[shape_slice]

        slices = []
        for dim_idx in range(n_size_dim):
            c_idx = random.randint(0, def_shape[dim_idx] - self.size[dim_idx])
            crop = slice(c_idx, c_idx + self.size[dim_idx])
            slices.append(crop)

        if n_size_dim < n_dim:
            # Insert slice for channels (no cropping)
            slices.insert(0 if self.channels_first else len(slices), slice(None))

        # convert to tuple to allow slicing
        slices = tuple(slices)

        outputs = []
        for idx, _input in enumerate(inputs):
            assert def_shape == _input.size()[shape_slice]
            _input = _input[slices]
            outputs.append(_input)
        return outputs if idx >= 1 else outputs[0]


class SpecialCrop(object):

    def __init__(self, size, crop_type=0, channels_first=False):
        """
        Perform a special crop - one of the four corners or center crop

        Arguments
        ---------
        size : tuple or list
            dimensions of the crop

        crop_type : integer or tuple or list of integers in {-1, 0, 1} (1 for each dimension)
            -1 = bottom/right/posterior crop (i.e. slice [-size:])
            0 = center crop
            1 = top/left/anterior crop (i.e. slice [:size])

        channels_first : boolean [False]
            if input contains more dimensions that self.size (e.g. multi-channel input), this
            specifies wheter to use the pad the last dimensions (True) or first dimensions (False).
            Has no effect when len(size) == len(input.shape)
        """
        self.size = size
        if isinstance(crop_type, (list, tuple)):
            assert len(crop_type) == len(self.size), 'Length of crop_type and size must be equal'
            for dim in crop_type:
                assert dim in {-1, 0, 1}, 'crop_type must be in {-1, 0, 1}'
            self.crop_type = crop_type
        else:
            assert crop_type in {-1, 0, 1}, 'crop_type must be in {-1, 0, 1}'
            self.crop_type = [crop_type] * len(self.size)

        self.channels_first = channels_first
    
    def __call__(self, *inputs):
        x_size = inputs[0].size()
        n_dim = len(x_size)
        n_size_dim = len(self.size)

        if n_size_dim < n_dim:
            shape_slice = slice(-n_size_dim, None) if self.channels_first else slice(None, n_size_dim)
        else:
            assert len(inputs[0].shape) == n_size_dim, 'Input has less dimensions than specified size.'
            shape_slice = slice(None)

        def_shape = x_size[shape_slice]

        slices = []
        for dim_idx, crop_type in enumerate(self.crop_type):
            if crop_type == -1:  # bottom crop
                b = slice(-self.size[dim_idx], None)
                slices.append(b)
            elif crop_type == 0:  # center crop
                diff = (def_shape[dim_idx] - self.size[dim_idx]) / 2.
                c = slice(int(math.ceil(diff)), -int(math.floor(diff)))
                slices.append(c)
            elif crop_type == 1:  # top crop
                t = slice(None, self.size[dim_idx])
                slices.append(t)

        if n_size_dim < n_dim:
            # Insert slice for channels (no cropping)
            slices.insert(0 if self.channels_first else len(slices), slice(None))

        # convert to tuple to allow slicing
        slices = tuple(slices)

        outputs = []
        for idx, _input in enumerate(inputs):
            assert def_shape == _input.size()[shape_slice]
            _input = _input[slices]
            outputs.append(_input)
        return outputs if idx >= 1 else outputs[0]


class Pad(object):

    def __init__(self, size, channels_first=False):
        """
        Pads an image to the given size

        Arguments
        ---------
        size : tuple or list
            size of crop

        channels_first : boolean [False]
            if input contains more dimensions that self.size (e.g. multi-channel input), this
            specifies wheter to use the pad the last dimensions (True) or first dimensions (False).
            Has no effect when len(size) == len(input.shape)
        """
        self.size = size
        self.channels_first = channels_first

    def __call__(self, x, y=None):
        x = x.numpy()
        n_dim = len(x.shape)
        n_size_dim = len(self.size)

        if n_size_dim < n_dim:
            shape_slice = slice(-n_size_dim, None) if self.channels_first else slice(None, n_size_dim)
        else:
            assert len(inputs[0].shape) == n_size_dim, 'Input has less dimensions than specified size.'
            shape_slice = slice(None)

        x_size = x.shape[shape_slice]
        shape_diffs = [0] * len(x.shape)  # start with 0 padding in all dimensions
        # calculate shape diffs for size dimensions
        shape_diffs[shape_slice] = [int(np.ceil((i_s - d_s))) for d_s, i_s in zip(x_size, self.size)]

        shape_diffs = np.maximum(shape_diffs, 0)
        pad_sizes = [(int(np.ceil(s/2.)), int(np.floor(s/2.))) for s in shape_diffs]
        x = np.pad(x, pad_sizes, mode='constant')
        if y is not None:
            y = y.numpy()
            y = np.pad(y, pad_sizes, mode='constant')
            return th.from_numpy(x), th.from_numpy(y)
        else:
            return th.from_numpy(x)


class PadNumpy(object):

    def __init__(self, size, channels_first=False):
        """
        Pads a Numpy image to the given size
        Return a Numpy image / image pair

        Arguments
        ---------
        size : tuple or list
            size of crop
        channels_first : boolean [False]
            if input contains more dimensions that self.size (e.g. multi-channel input), this
            specifies wheter to use the pad the last dimensions (True) or first dimensions (False).
            Has no effect when len(size) == len(inputs[0].shape)
        """
        self.size = size
        self.channels_first = channels_first

    def __call__(self, *inputs):
        n_dim = len(inputs[0].shape)
        n_size_dim = len(self.size)

        if n_size_dim < n_dim:
            shape_slice = slice(-n_size_dim, None) if self.channels_first else slice(None, n_size_dim)
        else:
            assert len(inputs[0].shape) == n_size_dim, 'Input has less dimensions than specified size.'
            shape_slice = slice(None)

        x_size = inputs[0].shape[shape_slice]
        shape_diffs = [0] * len(inputs[0].shape)  # start with 0 padding in all dimensions
        # calculate shape diffs for size dimensions
        shape_diffs[shape_slice] = [int(np.ceil((i_s - d_s))) for d_s, i_s in zip(x_size, self.size)]
        shape_diffs = np.maximum(shape_diffs, 0)
        pad_sizes = [(int(np.ceil(s/2.)), int(np.floor(s/2.))) for s in shape_diffs]
        outputs = []
        for idx, _input in enumerate(inputs):
            assert x_size == _input.shape[shape_slice]
            _input = np.pad(_input, pad_sizes, mode='minimum')
            outputs.append(_input)

        return outputs if idx >= 1 else outputs[0]


class PadFactorNumpy(object):

    def __init__(self, factor, n_size_dim=None, channels_first=False):
        """
        Pads a Numpy image (WxHxC) and makes sure that it is divisable by 2^factor
        Return a Numpy image

        Arguments
        ---------
        factor : tuple(int, int, int)
            division factor (to make sure that strided convs and
            transposed conv produce similar feature maps)
        n_size_dim : int [None]
            specifies the number of dimensions which need to be padded. If n_size_dim = None, all dimensions are padded.
        channels_first : boolean [False]
            if input contains more dimensions that self.size (e.g. multi-channel input), this
            specifies wheter to use the pad the last dimensions (True) or first dimensions (False).
            Has no effect when n_size_dim is None or equal to len(inputs[0].shape)
        """
        self.factor = np.array(factor, dtype=np.float32)
        self.n_size_dim = n_size_dim
        self.channels_first = channels_first

    def __call__(self, *inputs):
        n_dim = len(inputs[0].shape)
        n_size_dim = self.n_size_dim if self.n_size_dim is not None else n_dim

        if n_size_dim < n_dim:
            shape_slice = slice(-n_size_dim, None) if self.channels_first else slice(None, n_size_dim)
        else:
            assert len(inputs[0].shape) == n_size_dim, 'Input has less dimensions than specified size.'
            shape_slice = slice(None)

        x_size = inputs[0].shape[shape_slice]
        new_size = np.ceil(np.divide(x_size, self.factor)) * self.factor
        pre_pad  = np.round((new_size - x_size) / 2.0).astype(np.int16)
        post_pad = ((new_size - x_size) - pre_pad).astype(np.int16)

        pad_sizes = [(0, 0)] * n_size_dim
        pad_sizes[shape_slice] = zip(pre_pad, post_pad)

        outputs = []
        for idx, _input in enumerate(inputs):
            assert x_size == _input.shape[x_slice]
            _input = np.pad(_input, pad_sizes, mode='minimum')
            outputs.append(_input)

        return outputs if idx >= 1 else outputs[0]


class RandomFlip(object):

    def __init__(self, h=True, v=False, p=0.5):
        """
        Randomly flip an image horizontally and/or vertically with
        some probability.

        Arguments
        ---------
        h : boolean
            whether to horizontally flip w/ probability p

        v : boolean
            whether to vertically flip w/ probability p

        p : float between [0,1]
            probability with which to apply allowed flipping operations
        """
        self.horizontal = h
        self.vertical = v
        self.p = p

    def __call__(self, x, y=None):
        x = x.numpy()
        if y is not None:
            y = y.numpy()
        # horizontal flip with p = self.p
        if self.horizontal:
            if random.random() < self.p:
                x = x.swapaxes(2, 0)
                x = x[::-1, ...]
                x = x.swapaxes(0, 2)
                if y is not None:
                    y = y.swapaxes(2, 0)
                    y = y[::-1, ...]
                    y = y.swapaxes(0, 2)
        # vertical flip with p = self.p
        if self.vertical:
            if random.random() < self.p:
                x = x.swapaxes(1, 0)
                x = x[::-1, ...]
                x = x.swapaxes(0, 1)
                if y is not None:
                    y = y.swapaxes(1, 0)
                    y = y[::-1, ...]
                    y = y.swapaxes(0, 1)
        if y is None:
            # must copy because torch doesnt current support neg strides
            return th.from_numpy(x.copy())
        else:
            return th.from_numpy(x.copy()),th.from_numpy(y.copy())


class RandomOrder(object):
    """
    Randomly permute the channels of an image
    """
    def __call__(self, *inputs):
        order = th.randperm(inputs[0].dim())
        outputs = []
        for idx, _input in enumerate(inputs):
            _input = _input.index_select(0, order)
            outputs.append(_input)
        return outputs if idx >= 1 else outputs[0]

