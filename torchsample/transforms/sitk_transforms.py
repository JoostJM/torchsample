
import numpy as np
import SimpleITK as sitk
import torch as th


class SimpleITKtoTensor(object):
  def __call__(self, *inputs):
    outputs = []
    for idx, _input in enumerate(inputs):
      _output = sitk.GetArrayFromImage(_input)
      if _input.GetNumberOfComponentsPerPixel() == 1:
        _output = np.expand_dims(_output, -1)
      outputs.append(th.from_numpy(_output))
    return outputs if idx >= 1 else outputs[0]


class NormalizeSimpleITK(object):

  def __init__(self, norm_flag=True):
    """
    :param norm_flag: [bool] list of flags for normalisation
    """
    self.norm_flag = norm_flag

  def __call__(self, *inputs):
    # prepare the normalisation flag
    if isinstance(self.norm_flag, bool):
      norm_flag = [self.norm_flag] * len(inputs)
    else:
      norm_flag = self.norm_flag

    nif = sitk.NormalizeImageFilter()

    outputs = []
    for norm_flag, _input in zip(norm_flag, inputs):
      if norm_flag:
        outputs.append(nif.Execute(_input))
      else:
        outputs.append(_input)

    return outputs[0] if len(outputs) == 1 else outputs


class NormalizePercentileSimpleITK(object):

  def __init__(self,
               min_val=0.0,
               max_val=1.0,
               perc_threshold=(1.0, 95.0),
               norm_flag=True):
    """
    Normalize a SimpleITK Image between a min and max value
    :param min_val: (float) lower bound of normalized tensor
    :param max_val: (float) upper bound of normalized tensor
    :param perc_threshold: (float, float) percentile of image intensities used for scaling
    :param norm_flag: [bool] list of flags for normalisation
    """

    self.min_val = min_val
    self.max_val = max_val
    self.perc_threshold = perc_threshold
    self.norm_flag = norm_flag

  def __call__(self, *inputs):
    # prepare the normalisation flag
    if isinstance(self.norm_flag, bool):
      norm_flag = [self.norm_flag] * len(inputs)
    else:
      norm_flag = self.norm_flag

    outputs = []
    for norm_flag, _input in zip(norm_flag, inputs):
      if norm_flag:
        # determine the percentiles and threshold the outliers
        im_arr = sitk.GetArrayFromImage(_input)
        _min_val, _max_val = np.percentile(im_arr, self.perc_threshold)

        # scale the intensity values
        a = (self.max_val - self.min_val) / (_max_val - _min_val)
        im_arr -= _min_val
        im_arr *= a
        im_arr[im_arr < self.min_val] = self.min_val
        im_arr[im_arr > self.max_val] = self.max_val

        im = sitk.GetImageFromArray(im_arr)
        im.CopyInformation(_input)
        outputs.append(im)
      else:
        outputs.append(_input)

      return outputs[0] if len(outputs) == 1 else outputs


class PadSimpleITK(object):

  def __init__(self, size):
    """
    Pads a SimpleITK image to the given size
    Return a SimpleITK image / image pair

    Arguments
    ---------
    size : tuple or list
        size of crop in x, y, z
    """
    self.size = np.array(size)

  def __call__(self, *inputs):
    im_size = np.array(inputs[0].GetSize())
    shape_diffs = np.ceil(self.size - im_size).astype(int)
    shape_diffs = np.maximum(shape_diffs, 0)

    if np.max(shape_diffs) == 0:  # No padding required
      return inputs

    pif = sitk.ConstantPadImageFilter()
    pif.SetConstant(0)
    pif.SetPadLowerBound(np.ceil(shape_diffs / 2.).astype(int).tolist())
    pif.SetPadUpperBound(np.floor(shape_diffs / 2.).astype(int).tolist())

    outputs = []
    for idx, _input in enumerate(inputs):
      assert np.array_equal(im_size, np.array(_input.GetSize()))

      if _input.GetNumberOfComponentsPerPixel() > 1:
        # SimpleITK Constant pad only works for non-vector type images, so pad
        # each image channel separately, then join them again.
        channel_output = []
        for c_idx in range(_input.GetNumberOfComponentsPerPixel()):
          channel = sitk.VectorIndexSelectionCast(_input, c_idx)
          channel_output.append(pif.Execute(channel))
        outputs.append(sitk.Compose(channel_output))
      else:
        outputs.append(pif.Execute(_input))

    return outputs if idx >= 1 else outputs[0]


class PadFactorSimpleITK(object):

  def __init__(self, factor):
    """
    Pads a Numpy image (WxHxC) and makes sure that it is divisable by 2^factor
    Return a Numpy image

    Arguments
    ---------
    factor : int or tuple(int, int, int)
        division factor (to make sure that strided convs and
        transposed conv produce similar feature maps)
    """
    self.factor = np.array(factor, dtype=np.float32)

  def __call__(self, *inputs):
    im_size = np.array(inputs[0].GetSize())

    pad = self.factor - np.remainder(im_size, self.factor)  # compute the remainder
    if np.max(pad) == 0:  # No padding required
      return inputs

    pif = sitk.ConstantPadImageFilter()
    pif.SetConstant(0)
    pif.SetPadLowerBound(np.ceil(pad / 2.).astype(int).tolist())
    pif.SetPadUpperBound(np.floor(pad / 2.).astype(int).tolist())

    outputs = []
    for idx, _input in enumerate(inputs):
      assert np.array_equal(im_size, np.array(_input.GetSize()))

      if _input.GetNumberOfComponentsPerPixel() > 1:
        # SimpleITK Constant pad only works for non-vector type images, so pad
        # each image channel separately, then join them again.
        channel_output = []
        for c_idx in range(_input.GetNumberOfComponentsPerPixel()):
          channel = sitk.VectorIndexSelectionCast(_input, c_idx)
          channel_output.append(pif.Execute(channel))
        outputs.append(sitk.Compose(channel_output))
      else:
        outputs.append(pif.Execute(_input))

    return outputs if idx >= 1 else outputs[0]


class RandomFlipSimpleITK(object):

  def __init__(self, h=True, v=False, d=False, p=0.5):
    """
    Randomly flip a SipleITK image along x, y and/or z axis with
    some probability.

    Arguments
    ---------
    h : boolean
        whether to horizontally (x-axis) flip w/ probability p

    v : boolean
        whether to vertically (y-axis) flip w/ probability p

    d : boolean
        whether to depth (z-axis) flip w/ probability p

    p : float between [0,1]
        probability with which to apply allowed flipping operations
    """
    self.axes = (h, v, d)
    self.p = p

  def __call__(self, *inputs):

    fif = sitk.FlipImageFilter()

    axes = [(a and np.random.random() < self.p) for a in self.axes]

    fif.SetFlipAxes(axes)

    outputs = []
    for idx, _input in enumerate(inputs):
      outputs.append(fif.Execute(_input))

    return outputs if idx >= 1 else outputs[0]


class RandomCropSimpleITK(object):
  def __init__(self, size):
    """
    Randomly crop a torch tensor

    Arguments
    --------
    size : tuple or list
        dimensions of the crop
    """
    self.size = np.array(size)

  def __call__(self, *inputs):
    im_size = inputs[0].GetSize()
    n_dim = inputs[0].GetDimension()

    crop_lbound = []
    for dim_idx in range(n_dim):
      c_idx = np.random.randint(0, im_size[dim_idx] - self.size[dim_idx])
      crop_lbound.append(c_idx)

    crop_lbound = np.array(crop_lbound)
    crop_ubound = np.array(im_size) - self.size - crop_lbound

    cif = sitk.CropImageFilter()
    cif.SetLowerBoundaryCropSize(crop_lbound.astype('uint').tolist())
    cif.SetUpperBoundaryCropSize(crop_ubound.astype('uint').tolist())

    outputs = []
    for idx, _input in enumerate(inputs):
      assert im_size == _input.GetSize()
      outputs.append(cif.Execute(_input))
    return outputs if idx >= 1 else outputs[0]


class SpecialCropSimpleITK(object):

  def __init__(self, size, crop_type=0):
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

  def __call__(self, *inputs):
    im_size = inputs[0].GetSize()
    n_dim = inputs[0].GetDimension()
    assert n_dim == len(self.size), 'Length of specified size vector does not match dimensionality of input'

    crop_lbound = []
    crop_ubound = []
    for dim_idx, crop_type in enumerate(self.crop_type):
      if crop_type == -1:  # bottom crop
        crop_lbound.append(im_size[dim_idx] - self.size[dim_idx])
        crop_ubound.append(0)
      elif crop_type == 0:  # center crop
        diff = (im_size[dim_idx] - self.size[dim_idx]) / 2.
        crop_lbound.append(np.ceil(diff))
        crop_ubound.append(np.floor(diff))
      elif crop_type == 1:  # top crop
        crop_lbound.append(0)
        crop_ubound.append(im_size[dim_idx] - self.size[dim_idx])

    cif = sitk.CropImageFilter()
    cif.SetLowerBoundaryCropSize(np.array(crop_lbound).astype('uint').tolist())
    cif.SetUpperBoundaryCropSize(np.array(crop_ubound).astype('uint').tolist())

    outputs = []
    for idx, _input in enumerate(inputs):
      assert im_size == _input.GetSize()
      outputs.append(cif.Execute(_input))
    return outputs if idx >= 1 else outputs[0]


class RandomAffineSimpleITK(object):
  def __init__(self,
               rotation_range=None,
               translation_range=None,
               shear_range=None,
               zoom_range=None,
               interp=sitk.sitkLinear,
               lazy=False):
    """
    Perform an affine transforms with various sub-transforms, using
    only one interpolation and without having to instantiate each
    sub-transform individually.

    Arguments
    ---------
    rotation_range : one integer or float
        image will be rotated randomly between (-degrees, degrees)

    translation_range : float or 3-tuple of float between [0, 1)
        first value:
            fractional bounds of total depth to shift image
            image will be depth shifted between
            (-depth_range * depth_dimension, depth_range * depth_dimension)
        second value:
            fractional bounds of total width to shift image
            Image will be vertically shifted between
            (-width_range * width_dimension, width_range * width_dimension)
        third value:
            fractional bounds of total heigth to shift image
            image will be horizontally shifted between
            (-height_range * height_dimension, height_range * height_dimension)

    shear_range : float
        image will be sheared randomly between (-degrees, degrees)

    zoom_range : list/tuple with two floats between [0, infinity).
        first float should be less than the second
        lower and upper bounds on percent zoom.
        Anything less than 1.0 will zoom in on the image,
        anything greater than 1.0 will zoom out on the image.
        e.g. (0.7, 1.0) will only zoom in,
             (1.0, 1.4) will only zoom out,
             (0.7, 1.4) will randomly zoom in or out

    interp : sitk enumerated value for interpolator
        type of interpolation to use. You can provide a different
        type of interpolation for each input, e.g. if you have two
        inputs then you can say `interp=[sitk.sitkLinear,sitk.sitkNearestNeighbor]
    
    lazy    : boolean
        if true, only create the affine transform and return that
        if false, perform the transform on the Image and return the Image
    """
    self.transforms = []
    if rotation_range is not None:
        rotation_tform = RandomRotateSimpleITK(rotation_range, lazy=True)
        self.transforms.append(rotation_tform)

    if translation_range is not None:
        translation_tform = RandomTranslateSimpleITK(translation_range, lazy=True)
        self.transforms.append(translation_tform)

    if shear_range is not None:
        shear_tform = RandomShearSimpleITK(shear_range, lazy=True)
        self.transforms.append(shear_tform)

    if zoom_range is not None:
        zoom_tform = RandomZoomSimpleITK(zoom_range, lazy=True)
        self.transforms.append(zoom_tform)

    self.interp = interp
    self.lazy = lazy

    if len(self.transforms) == 0:
        raise Exception('Must give at least one transform parameter')

  def __call__(self, *inputs):
    # collect all of the lazily returned tform matrices
    tform = self.transforms[0](inputs[0])
    for tf in self.transforms[1:]:
      tform.AddTransform(tf(inputs[0]))
    self.tform_matrix = tform

    if self.lazy:
      return tform
    else:
      return AffineSimpleITK(tform, self.interp)(*inputs)


class AffineComposeSimpleITK(object):

  def __init__(self,
               transforms,
               interp=sitk.sitkLinear):
    """
    Apply a collection of explicit affine transforms to an input image,
    and to a target image if necessary

    Arguments
    ---------
    transforms : list or tuple
        each element in the list/tuple should be an affine transform.
        currently supported transforms:
            - RotateSimpleITK()
            - TranslateSimpleITK()
            - ShearSimpleITK()
            - ZoomSimpleITK()

    interp : sitk enumerated value for interpolator
        type of interpolation to use. You can provide a different
        type of interpolation for each input, e.g. if you have two
        inputs then you can say `interp=[sitk.sitkLinear,sitk.sitkNearestNeighbor]

    """
    self.transforms = transforms
    self.interp = interp
    # set transforms to lazy so they only return the tform matrix
    for t in self.transforms:
      t.lazy = True

  def __call__(self, *inputs):
    # collect all of the lazily returned tform matrices
    tform = self.transforms[0](inputs[0])
    for tf in self.transforms[1:]:
      tform.AddTransform(tf(inputs[0]))
    self.tform_matrix = tform

    return AffineSimpleITK(tform, self.interp)(*inputs)


class AffineSimpleITK(object):
  
  def __init__(self, transform, interp=sitk.sitkLinear):
    """
    Perform an affine transforms with various sub-transforms, using
    only one interpolation and without having to instantiate each
    sub-transform individually.

    Arguments
    ---------
    transform : a SimpleITK Transform object

    interp : sitk enumerated value for interpolator
        type of interpolation to use. You can provide a different
        type of interpolation for each input, e.g. if you have two
        inputs then you can say `interp=[sitk.sitkLinear,sitk.sitkNearestNeighbor]

    """
    self.transform = transform
    self.interp = interp

  def __call__(self, *inputs):
    if not isinstance(self.interp, (tuple, list)):
      interp = [self.interp] * len(inputs)
    else:
      interp = self.interp

    rif = sitk.ResampleImageFilter()
    rif.SetReferenceImage(inputs[0])

    # It important to keep in mind that a transform in a resampling operation
    # defines the transform from the output space to the input space.
    # Therefore, set the inverse!
    rif.SetTransform(self.transform.GetInverse())

    outputs = []
    for idx, _input in enumerate(inputs):
      rif.SetInterpolator(interp[idx])
      outputs.append(rif.Execute(_input))
    return outputs if idx >= 1 else outputs[0]


class RandomRotateSimpleITK(object):

  def __init__(self,
               rotation_range,
               axis='z',
               interp=sitk.sitkLinear,
               lazy=False):
    """
    Randomly rotate an image between (-degrees, degrees). If the image
    has multiple channels, the same rotation will be applied to each channel.

    Arguments
    ---------
    rotation_range : integer or float
        image will be rotated between (-degrees, degrees) degrees

    axis: string in {'x', 'y', 'z'}
        axis for rotation. This axis will be fixed.

    interp : sitk enumerated value for interpolator
        type of interpolation to use. You can provide a different
        type of interpolation for each input, e.g. if you have two
        inputs then you can say `interp=[sitk.sitkLinear,sitk.sitkNearestNeighbor]

    lazy    : boolean
        if true, only create the affine transform and return that
        if false, perform the transform on the Image and return the Image
    """
    self.rotation_range = rotation_range
    self.axis = axis
    self.interp = interp
    self.lazy = lazy

  def __call__(self, *inputs):
    degree = np.random.uniform(-self.rotation_range, self.rotation_range)

    if self.lazy:
      return RotateSimpleITK(degree, axis=self.axis, lazy=True)(inputs[0])
    else:
      outputs = RotateSimpleITK(degree, axis=self.axis,
                         interp=self.interp)(*inputs)
      return outputs


class RandomChoiceRotateSimpleITK(object):

  def __init__(self,
               values,
               axis='z',
               p=None,
               interp=sitk.sitkLinear,
               lazy=False):
    """
    Randomly rotate an image from a list of values. If the image
    has multiple channels, the same rotation will be applied to each channel.

    Arguments
    ---------
    values : a list or tuple
        the values from which the rotation value will be sampled

    axis: string in {'x', 'y', 'z'}
        axis for rotation. This axis will be fixed.

    p : a list or tuple the same length as `values`
        the probabilities of sampling any given value. Must sum to 1.
        If omitted, uniform probability across choices is used.

    interp : sitk enumerated value for interpolator
        type of interpolation to use. You can provide a different
        type of interpolation for each input, e.g. if you have two
        inputs then you can say `interp=[sitk.sitkLinear,sitk.sitkNearestNeighbor]

    lazy    : boolean
        if true, only create the affine transform and return that
        if false, perform the transform on the Image and return the Image
    """
    #if isinstance(values, (list, tuple)):
    #  values = th.FloatTensor(values)
    self.values = values
    self.axis = axis
    if p is not None:
      if abs(1.0 - sum(p)) > 1e-3:
        raise ValueError('Probs must sum to 1')
      if len(p) != len(values):
        raise ValueError('Length of p must be equal to length of values')
    self.p = p
    self.interp = interp
    self.lazy = lazy

  def __call__(self, *inputs):
    degree = np.random.choice(self.values, p=self.p)
    if self.lazy:
      return RotateSimpleITK(degree, axis=self.axis, lazy=True)(inputs[0])
    else:
      outputs = RotateSimpleITK(degree, axis=self.axis,
                         interp=self.interp)(*inputs)
      return outputs


class RotateSimpleITK(object):

  def __init__(self,
               value,
               axis='z',
               interp=sitk.sitkLinear,
               lazy=False):
    """
    Randomly rotate an image between (-degrees, degrees). If the image
    has multiple channels, the same rotation will be applied to each channel.

    Arguments
    ---------
    value : integer or float
        image will be rotated degrees

    axis: string in {'x', 'y', 'z'}
        axis for rotation. This axis will be fixed.

    interp : sitk enumerated value for interpolator
        type of interpolation to use. You can provide a different
        type of interpolation for each input, e.g. if you have two
        inputs then you can say `interp=[sitk.sitkLinear,sitk.sitkNearestNeighbor]

    lazy    : boolean
        if true, only create the affine transform and return that
        if false, perform the transform on the Image and return the Image
    """
    self.value = value
    self.axis = axis
    self.interp = interp
    self.lazy = lazy

  def __call__(self, *inputs):
    ndim = inputs[0].GetDimension()
    tform = sitk.AffineTransform(ndim)

    theta = np.pi / 180 * self.value
    if ndim == 2 or self.axis == 'z':  # in case of 2D always rotated in first 2 axes
      # boolean pre=False indicates the rotation needs to be applied after any current tform content
      tform.Rotate(0, 1, theta, False)
    elif self.axis == 'y':
      tform.Rotate(0, 2, theta, False)
    elif self.axis == 'x':
      tform.Rotate(1, 2, theta, False)
    else:
      raise ValueError('axis %s not in {"x", "y", "z"}' % self.axis)

    tform = sitk.Transform(tform)
    if self.lazy:
      return tform
    else:
      return AffineSimpleITK(tform, self.interp)(*inputs)


class RandomTranslateSimpleITK(object):

  def __init__(self,
               translation_range,
               interp=sitk.sitkLinear,
               lazy=False):
    """
    Randomly translate an image some fraction of total height and/or
    some fraction of total width. If the image has multiple channels,
    the same translation will be applied to each channel. Assumes CDWH
    ordering.

    Arguments
    ---------
    translation_range : float or 3-tuple of float between [0, 1)
        first value:
            fractional bounds of total depth to shift image
            image will be depth shifted between
            (-depth_range * depth_dimension, depth_range * depth_dimension)
        second value:
            fractional bounds of total width to shift image 
            Image will be vertically shifted between 
            (-width_range * width_dimension, width_range * width_dimension)
        third value:
            fractional bounds of total heigth to shift image
            image will be horizontally shifted between
            (-height_range * height_dimension, height_range * height_dimension)

    interp : sitk enumerated value for interpolator
        type of interpolation to use. You can provide a different
        type of interpolation for each input, e.g. if you have two
        inputs then you can say `interp=[sitk.sitkLinear,sitk.sitkNearestNeighbor]

    lazy    : boolean
        if true, only create the affine transform and return that
        if false, perform the transform on the Image and return the Image
    """
    if isinstance(translation_range, float):
      translation_range = (translation_range, translation_range, translation_range)
    self.depth_range = translation_range[0]
    self.width_range = translation_range[1]
    self.height_range = translation_range[2]
    self.interp = interp
    self.lazy = lazy

  def __call__(self, *inputs):
    # height shift
    random_height = np.random.uniform(-self.height_range, self.height_range)
    # width shift
    random_width = np.random.uniform(-self.width_range, self.width_range)
    # depth shift
    random_depth = np.random.uniform(-self.depth_range, self.depth_range)

    if self.lazy:
      return TranslateSimpleITK([random_depth, random_width, random_height],
                                lazy=True)(inputs[0])
    else:
      outputs = TranslateSimpleITK([random_depth, random_width, random_height],
                                   interp=self.interp)(*inputs)
      return outputs


class RandomChoiceTranslateSimpleITK(object):

  def __init__(self,
               values,
               p=None,
               interp=sitk.sitkLinear,
               lazy=False):
    """
    Randomly translate an image some fraction of total height and/or
    some fraction of total width from a list of potential values. 
    If the image has multiple channels,
    the same translation will be applied to each channel.

    Arguments
    ---------
    values : a list or tuple
        the values from which the translation value will be sampled

    p : a list or tuple the same length as `values`
        the probabilities of sampling any given value. Must sum to 1.

    interp : sitk enumerated value for interpolator
        type of interpolation to use. You can provide a different
        type of interpolation for each input, e.g. if you have two
        inputs then you can say `interp=[sitk.sitkLinear,sitk.sitkNearestNeighbor]

    lazy    : boolean
        if true, only create the affine transform and return that
        if false, perform the transform on the Image and return the Image
    """
    self.values = values
    if p is not None:
      if abs(1.0 - sum(p)) > 1e-3:
        raise ValueError('Probs must sum to 1')
      if len(p) != len(values):
        raise ValueError('Length of p must be equal to length of values')
    self.p = p
    self.interp = interp
    self.lazy = lazy

  def __call__(self, *inputs):
    random_height = np.random.choice(self.values, p=self.p)
    random_width = np.random.choice(self.values, p=self.p)
    random_depth = np.random.choice(self.values, p=self.p)

    if self.lazy:
      return TranslateSimpleITK([random_depth, random_width, random_height],
                                lazy=True)(inputs[0])
    else:
      outputs = TranslateSimpleITK([random_depth, random_width, random_height],
                                   interp=self.interp)(*inputs)
      return outputs


class TranslateSimpleITK(object):

  def __init__(self,
               value,
               interp=sitk.sitkLinear,
               lazy=False):
    """
    Arguments
    ---------
    value : float or 3-tuple of float
        if single value, both horizontal, vertical and depth translation
        will be this value * total height/width. Thus, value should
        be a fraction of total height/width with range (-1, 1)

    interp : sitk enumerated value for interpolator
        type of interpolation to use. You can provide a different
        type of interpolation for each input, e.g. if you have two
        inputs then you can say `interp=[sitk.sitkLinear,sitk.sitkNearestNeighbor]
    
    lazy    : boolean
        if true, only create the affine transform and return that
        if false, perform the transform on the Image and return the Image

    """
    if not isinstance(value, (tuple, list)):
      value = (value, value, value)

    if value[0] > 1 or value[0] < -1:
      raise ValueError('Translation must be between -1 and 1')
    if value[1] > 1 or value[1] < -1:
      raise ValueError('Translation must be between -1 and 1')
    if value[2] > 1 or value[2] < -1:
      raise ValueError('Translation must be between -1 and 1')

    self.depth_range = value[0]
    self.width_range = value[1]
    self.height_range = value[2]
    self.interp = interp
    self.lazy = lazy

  def __call__(self, *inputs):
    ndim = inputs[0].GetDimension()
    im_size = inputs[0].GetSize()
    tz = self.depth_range * im_size[0]
    ty = self.width_range * im_size[1]
    tx = self.height_range * im_size[2]

    tform = sitk.AffineTransform(ndim)
    tform.Translate((tx, ty, tz))

    tform = sitk.Transform(tform)
    if self.lazy:
      return tform
    else:
      return AffineSimpleITK(tform, self.interp)(*inputs)


class RandomShearSimpleITK(object):

  def __init__(self,
               shear_range,
               interp=sitk.sitkLinear,
               lazy=False):
    """
    Randomly shear an image with radians (-shear_range, shear_range)

    Arguments
    ---------
    shear_range : float
        radian bounds on the shear transform

    interp : sitk enumerated value for interpolator
        type of interpolation to use. You can provide a different
        type of interpolation for each input, e.g. if you have two
        inputs then you can say `interp=[sitk.sitkLinear,sitk.sitkNearestNeighbor]

    lazy    : boolean
        if true, only create the affine transform and return that
        if false, perform the transform on the Image and return the Image
    """
    self.shear_range = shear_range
    self.interp = interp
    self.lazy = lazy

  def __call__(self, *inputs):
    shear_x = np.random.uniform(-self.shear_range, self.shear_range)
    shear_y = np.random.uniform(-self.shear_range, self.shear_range)

    if self.lazy:
      return ShearSimpleITK([shear_x, shear_y],
                            lazy=True)(inputs[0])
    else:
      outputs = ShearSimpleITK([shear_x, shear_y],
                               interp=self.interp)(*inputs)
      return outputs


class RandomChoiceShearSimpleITK(object):

  def __init__(self,
               values,
               p=None,
               interp=sitk.sitkLinear,
               lazy=False):
    """
    Randomly shear an image with a value sampled from a list of values.

    Arguments
    ---------
    values : a list or tuple
        the values from which the rotation value will be sampled

    p : a list or tuple the same length as `values`
        the probabilities of sampling any given value. Must sum to 1.

    interp : sitk enumerated value for interpolator
        type of interpolation to use. You can provide a different
        type of interpolation for each input, e.g. if you have two
        inputs then you can say `interp=[sitk.sitkLinear,sitk.sitkNearestNeighbor]

    lazy    : boolean
        if true, only create the affine transform and return that
        if false, perform the transform on the Image and return the Image
    """
    self.values = values
    if p is not None:
      if abs(1.0 - sum(p)) > 1e-3:
        raise ValueError('Probs must sum to 1')
      if len(p) != len(values):
        raise ValueError('Length of p must be equal to length of values')
    self.p = p
    self.interp = interp
    self.lazy = lazy

  def __call__(self, *inputs):
    shear_x = np.random.choice(self.values, p=self.p)
    shear_y = np.random.choice(self.values, p=self.p)

    if self.lazy:
      return ShearSimpleITK([shear_x, shear_y],
                     lazy=True)(inputs[0])
    else:
      outputs = ShearSimpleITK([shear_x, shear_y],
                        interp=self.interp)(*inputs)
      return outputs


class ShearSimpleITK(object):

  def __init__(self,
               value,
               interp=sitk.sitkLinear,
               lazy=False):
    if isinstance(value, (list, tuple)):
      self.value = value
    else:
      self.value = (value, 0)
    self.interp = interp
    self.lazy = lazy

  def __call__(self, *inputs):
    ndim = inputs[0].GetDimension()
    tform = sitk.AffineTransform(ndim)

    shear_matrix = np.eye(3, dtype=float)

    if self.value[0] != 0:
      theta_x = np.pi / 180 * self.value[0]
      shear_matrix[0, 1] = -np.sin(theta_x)
      shear_matrix[1, 1] = np.cos(theta_x)
    if self.value[1] != 0:
      theta_y = np.pi / 180 * self.value[1]
      shear_matrix[0, 0] = np.cos(theta_y)
      shear_matrix[1, 0] = np.sin(theta_y)

    tform.SetMatrix(shear_matrix.flatten())
    tform = sitk.Transform(tform)
    if self.lazy:
      return tform
    else:
      return AffineSimpleITK(tform, self.interp)(*inputs)


class RandomZoomSimpleITK(object):

  def __init__(self,
               zoom_range,
               interp=sitk.sitkLinear,
               lazy=False):
    """
    Randomly zoom in and/or out on an image 

    Arguments
    ---------
    zoom_range : tuple or list with 2 values, both between (0, infinity)
        lower and upper bounds on percent zoom. 
        Anything less than 1.0 will zoom in on the image, 
        anything greater than 1.0 will zoom out on the image.
        e.g. (0.7, 1.0) will only zoom in, 
             (1.0, 1.4) will only zoom out,
             (0.7, 1.4) will randomly zoom in or out

    interp : sitk enumerated value for interpolator
        type of interpolation to use. You can provide a different
        type of interpolation for each input, e.g. if you have two
        inputs then you can say `interp=[sitk.sitkLinear,sitk.sitkNearestNeighbor]

    lazy    : boolean
        if true, only create the affine transform and return that
        if false, perform the transform on the Image and return the Image
    """
    if not isinstance(zoom_range, list) and not isinstance(zoom_range, tuple):
      raise ValueError('zoom_range must be tuple or list with 2 values')
    self.zoom_range = zoom_range
    self.interp = interp
    self.lazy = lazy

  def __call__(self, *inputs):
    zx = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
    zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
    zz = np.random.uniform(self.zoom_range[0], self.zoom_range[1])

    if self.lazy:
      return ZoomSimpleITK([zz, zy, zx], lazy=True)(inputs[0])
    else:
      outputs = ZoomSimpleITK([zz, zy, zx],
                       interp=self.interp)(*inputs)
      return outputs


class RandomChoiceZoomSimpleITK(object):

  def __init__(self,
               values,
               p=None,
               interp=sitk.sitkLinear,
               lazy=False):
    """
    Randomly zoom in and/or out on an image with a value sampled from
    a list of values

    Arguments
    ---------
    values : a list or tuple
        the values from which the applied zoom value will be sampled

    p : a list or tuple the same length as `values`
        the probabilities of sampling any given value. Must sum to 1.

    interp : sitk enumerated value for interpolator
        type of interpolation to use. You can provide a different
        type of interpolation for each input, e.g. if you have two
        inputs then you can say `interp=[sitk.sitkLinear,sitk.sitkNearestNeighbor]

    lazy    : boolean
        if true, only create the affine transform and return that
        if false, perform the transform on the Image and return the Image
    """
    if isinstance(values, (list, tuple)):
      values = th.FloatTensor(values)
    self.values = values
    if p is not None:
      if abs(1.0 - sum(p)) > 1e-3:
        raise ValueError('Probs must sum to 1')
      if len(p) != len(values):
        raise ValueError('Length of p must be equal to length of values')
    self.p = p
    self.interp = interp
    self.lazy = lazy

  def __call__(self, *inputs):
    zx = np.random.choice(self.values, p=self.p)
    zy = np.random.choice(self.values, p=self.p)
    zz = np.random.choice(self.values, p=self.p)

    if self.lazy:
      return ZoomSimpleITK([zz, zy, zx], lazy=True)(inputs[0])
    else:
      outputs = ZoomSimpleITK([zz, zy, zx],
                       interp=self.interp)(*inputs)
      return outputs


class ZoomSimpleITK(object):

  def __init__(self,
               value,
               interp=sitk.sitkLinear,
               lazy=False):
    """
    Arguments
    ---------
    value : float
        Fractional zoom.
        =1 : no zoom
        >1 : zoom-in (value-1)%
        <1 : zoom-out (1-value)%

    interp : sitk enumerated value for interpolator
        type of interpolation to use. You can provide a different
        type of interpolation for each input, e.g. if you have two
        inputs then you can say `interp=[sitk.sitkLinear,sitk.sitkNearestNeighbor]

    lazy    : boolean
        if true, only create the affine transform and return that
        if false, perform the transform on the Image and return the Image
    """
    self.value = value
    self.interp = interp
    self.lazy = lazy

  def __call__(self, *inputs):
    ndim = inputs[0].GetDimension()
    tform = sitk.AffineTransform(ndim)
    tform.Scale(self.value)

    tform = sitk.Transform(tform)
    if self.lazy:
      return tform
    else:
      return AffineSimpleITK(tform, self.interp)(*inputs)
