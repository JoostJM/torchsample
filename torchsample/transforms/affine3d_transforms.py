"""
Affine transforms implemented on torch tensors, and
requiring only one interpolation
"""

import math
import random
import torch as th

from ..utils import th_affine3d, th_random_choice


class RandomAffine3D(object):

    def __init__(self, 
                 rotation_range=None, 
                 translation_range=None,
                 shear_range=None, 
                 zoom_range=None,
                 interp='trilinear',
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

        interp : string in {'trilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['trilinear','nearest']

        """
        self.transforms = []
        if rotation_range is not None:
            rotation_tform = RandomRotate3D(rotation_range, lazy=True)
            self.transforms.append(rotation_tform)

        if translation_range is not None:
            translation_tform = RandomTranslate3D(translation_range, lazy=True)
            self.transforms.append(translation_tform)

        if shear_range is not None:
            shear_tform = RandomShear3D(shear_range, lazy=True)
            self.transforms.append(shear_tform) 

        if zoom_range is not None:
            zoom_tform = RandomZoom3D(zoom_range, lazy=True)
            self.transforms.append(zoom_tform)

        self.interp = interp
        self.lazy = lazy

        if len(self.transforms) == 0:
            raise Exception('Must give at least one transform parameter')

    def __call__(self, *inputs):
        # collect all of the lazily returned tform matrices
        tform_matrix = self.transforms[0](inputs[0])
        for tform in self.transforms[1:]:
            tform_matrix = tform_matrix.mm(tform(inputs[0])) 
        self.tform_matrix = tform_matrix

        if self.lazy:
            return tform_matrix
        else:
            outputs = Affine3D(tform_matrix,
                             interp=self.interp)(*inputs)
            return outputs


class Affine3D(object):

    def __init__(self, 
                 tform_matrix,
                 interp='trilinear'):
        """
        Perform an affine transforms with various sub-transforms, using
        only one interpolation and without having to instantiate each
        sub-transform individually.

        Arguments
        ---------
        tform_matrix : a 3x3 or 3x4 matrix
            affine transformation matrix to apply

        interp : string in {'trilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['trilinear','nearest']

        """
        self.tform_matrix = tform_matrix
        self.interp = interp

    def __call__(self, *inputs):
        if not isinstance(self.interp, (tuple,list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        outputs = []
        for idx, _input in enumerate(inputs):
            input_tf = th_affine3d(_input,
                                   self.tform_matrix,
                                   mode=interp[idx])
            outputs.append(input_tf)
        return outputs if idx >= 1 else outputs[0]


class Affine3DCompose(object):

    def __init__(self, 
                 transforms,
                 interp='trilinear'):
        """
        Apply a collection of explicit affine transforms to an input image,
        and to a target image if necessary

        Arguments
        ---------
        transforms : list or tuple
            each element in the list/tuple should be an affine transform.
            currently supported transforms:
                - Rotate3D()
                - Translate3D()
                - Shear3D()
                - Zoom3D()

        interp : string in {'trilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['trilinear','nearest']

        """
        self.transforms = transforms
        self.interp = interp
        # set transforms to lazy so they only return the tform matrix
        for t in self.transforms:
            t.lazy = True

    def __call__(self, *inputs):
        # collect all of the lazily returned tform matrices
        tform_matrix = self.transforms[0](inputs[0])
        for tform in self.transforms[1:]:
            tform_matrix = tform_matrix.mm(tform(inputs[0])) 

        if not isinstance(self.interp, (tuple,list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        outputs = []
        for idx, _input in enumerate(inputs):
            input_tf = th_affine3d(_input,
                                   tform_matrix,
                                   mode=interp[idx])
            outputs.append(input_tf)
        return outputs if idx >= 1 else outputs[0]


class RandomRotate3D(object):

    def __init__(self, 
                 rotation_range,
                 axis=0,
                 interp='trilinear',
                 lazy=False):
        """
        Randomly rotate an image between (-degrees, degrees). If the image
        has multiple channels, the same rotation will be applied to each channel.

        Arguments
        ---------
        rotation_range : integer or float
            image will be rotated between (-degrees, degrees) degrees

        axis: integer in (0, 1, 2)
            axis (z, y, x) for rotation. This axis will be fixed.

        interp : string in {'trilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['trilinear','nearest']

        lazy    : boolean
            if true, only create the affine transform matrix and return that
            if false, perform the transform on the tensor and return the tensor
        """
        self.rotation_range = rotation_range
        self.axis = axis
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        degree = random.uniform(-self.rotation_range, self.rotation_range)

        if self.lazy:
            return Rotate3D(degree, axis=self.axis, lazy=True)(inputs[0])
        else:
            outputs = Rotate3D(degree, axis=self.axis,
                               interp=self.interp)(*inputs)
            return outputs


class RandomChoiceRotate3D(object):

    def __init__(self, 
                 values,
                 axis=0,
                 p=None,
                 interp='trilinear',
                 lazy=False):
        """
        Randomly rotate an image from a list of values. If the image
        has multiple channels, the same rotation will be applied to each channel.

        Arguments
        ---------
        values : a list or tuple
            the values from which the rotation value will be sampled

        axis: integer in (0, 1, 2)
            axis (z, y, x) for rotation. This axis will be fixed.

        p : a list or tuple the same length as `values`
            the probabilities of sampling any given value. Must sum to 1.

        interp : string in {'trilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['trilinear','nearest']

        lazy    : boolean
            if true, only create the affine transform matrix and return that
            if false, perform the transform on the tensor and return the tensor
        """
        if isinstance(values, (list, tuple)):
            values = th.FloatTensor(values)
        self.values = values
        self.axis = axis
        if p is None:
            p = th.ones(len(values)) / len(values)
        else:
            if abs(1.0-sum(p)) > 1e-3:
                raise ValueError('Probs must sum to 1')
        self.p = p
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        degree = th_random_choice(self.values, p=self.p)

        if self.lazy:
            return Rotate3D(degree, axis=self.axis, lazy=True)(inputs[0])
        else:
            outputs = Rotate3D(degree, axis=self.axis,
                               interp=self.interp)(*inputs)
            return outputs


class Rotate3D(object):

    def __init__(self, 
                 value,
                 axis=0,
                 interp='trilinear',
                 lazy=False):
        """
        Randomly rotate an image between (-degrees, degrees). If the image
        has multiple channels, the same rotation will be applied to each channel.

        Arguments
        ---------
        value : integer or float
            image will be rotated degrees

        axis: integer in (0, 1, 2)
            axis (z, y, x) for rotation. This axis will be fixed.

        interp : string in {'trilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['trilinear','nearest']

        lazy    : boolean
            if true, only create the affine transform matrix and return that
            if false, perform the transform on the tensor and return the tensor
        """
        self.value = value
        self.axis = axis
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        if not isinstance(self.interp, (tuple,list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        theta = math.pi / 180 * self.value
        if self.axis == 0:
            rotation_matrix = th.FloatTensor([[1,               0,                0, 0],
                                              [0, math.cos(theta), -math.sin(theta), 0],
                                              [0, math.sin(theta),  math.cos(theta), 0],
                                              [0,               0,                0, 1]])
        elif self.axis == 1:
            rotation_matrix = th.FloatTensor([[math.cos(theta),  0, math.sin(theta), 0],
                                              [0,                1,               0, 0],
                                              [-math.sin(theta), 0, math.cos(theta), 0],
                                              [0,                0,               0, 1]])
        elif self.axis == 2:
            rotation_matrix = th.FloatTensor([[math.cos(theta), -math.sin(theta), 0, 0],
                                              [math.sin(theta),  math.cos(theta), 0, 0],
                                              [              0,                0, 1, 0],
                                              [              0,                0, 0, 1]])
        else:
            raise ValueError('axis out of range [0-2]')

        if self.lazy:
            return rotation_matrix
        else:
            outputs = []
            for idx, _input in enumerate(inputs):
                input_tf = th_affine3d(_input,
                                       rotation_matrix,
                                       mode=interp[idx],
                                       center=True)
                outputs.append(input_tf)
            return outputs if idx >= 1 else outputs[0]


class RandomTranslate3D(object):

    def __init__(self, 
                 translation_range,
                 interp='trilinear',
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

        interp : string in {'trilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['trilinear','nearest']

        lazy    : boolean
            if true, only create the affine transform matrix and return that
            if false, perform the transform on the tensor and return the tensor
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
        random_height = random.uniform(-self.height_range, self.height_range)
        # width shift
        random_width = random.uniform(-self.width_range, self.width_range)
        # depth shift
        random_depth = random.uniform(-self.depth_range, self.depth_range)

        if self.lazy:
            return Translate3D([random_depth, random_width, random_height],
                             lazy=True)(inputs[0])
        else:
            outputs = Translate3D([random_depth, random_width, random_height],
                                 interp=self.interp)(*inputs)
            return outputs


class RandomChoiceTranslate3D(object):

    def __init__(self,
                 values,
                 p=None,
                 interp='trilinear',
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

        interp : string in {'trilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['trilinear','nearest']

        lazy    : boolean
            if true, only create the affine transform matrix and return that
            if false, perform the transform on the tensor and return the tensor
        """
        if isinstance(values, (list, tuple)):
            values = th.FloatTensor(values)
        self.values = values
        if p is None:
            p = th.ones(len(values)) / len(values)
        else:
            if abs(1.0-sum(p)) > 1e-3:
                raise ValueError('Probs must sum to 1')
        self.p = p
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        random_height = th_random_choice(self.values, p=self.p)
        random_width = th_random_choice(self.values, p=self.p)
        random_depth = th_random_choice(self.values, p=self.p)

        if self.lazy:
            return Translate3D([random_depth, random_width, random_height],
                             lazy=True)(inputs[0])
        else:
            outputs = Translate3D([random_depth, random_width, random_height],
                                interp=self.interp)(*inputs)
            return outputs


class Translate3D(object):

    def __init__(self, 
                 value, 
                 interp='trilinear',
                 lazy=False):
        """
        Arguments
        ---------
        value : float or 3-tuple of float
            if single value, both horizontal, vertical and depth translation
            will be this value * total height/width. Thus, value should
            be a fraction of total height/width with range (-1, 1)

        interp : string in {'trilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['trilinear','nearest']

        """
        if not isinstance(value, (tuple,list)):
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
        if not isinstance(self.interp, (tuple,list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        tz = self.depth_range * inputs[0].size(1)
        ty = self.width_range * inputs[0].size(2)
        tx = self.height_range * inputs[0].size(3)

        translation_matrix = th.FloatTensor([[1, 0, 0, tz],
                                             [0, 1, 0, ty],
                                             [0, 0, 1, tx],
                                             [0, 0, 0,  1]])
        if self.lazy:
            return translation_matrix
        else:
            outputs = []
            for idx, _input in enumerate(inputs):
                input_tf = th_affine3d(_input,
                                       translation_matrix,
                                       mode=interp[idx],
                                       center=True)
                outputs.append(input_tf)
            return outputs if idx >= 1 else outputs[0]


class RandomShear3D(object):

    def __init__(self, 
                 shear_range,
                 interp='trilinear',
                 lazy=False):
        """
        Randomly shear an image with radians (-shear_range, shear_range)

        Arguments
        ---------
        shear_range : float
            radian bounds on the shear transform
        
        interp : string in {'trilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['trilinear','nearest']

        lazy    : boolean
            if false, perform the transform on the tensor and return the tensor
            if true, only create the affine transform matrix and return that
        """
        self.shear_range = shear_range
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        shear_x = random.uniform(-self.shear_range, self.shear_range)
        shear_y = random.uniform(-self.shear_range, self.shear_range)

        if self.lazy:
            return Shear3D([shear_x, shear_y],
                         lazy=True)(inputs[0])
        else:
            outputs = Shear3D([shear_x, shear_y],
                            interp=self.interp)(*inputs)
            return outputs


class RandomChoiceShear3D(object):

    def __init__(self,
                 values,
                 p=None,
                 interp='trilinear',
                 lazy=False):
        """
        Randomly shear an image with a value sampled from a list of values.

        Arguments
        ---------
        values : a list or tuple
            the values from which the rotation value will be sampled

        p : a list or tuple the same length as `values`
            the probabilities of sampling any given value. Must sum to 1.

        interp : string in {'trilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['trilinear','nearest']

        lazy    : boolean
            if false, perform the transform on the tensor and return the tensor
            if true, only create the affine transform matrix and return that
        """
        if isinstance(values, (list, tuple)):
            values = th.FloatTensor(values)
        self.values = values
        if p is None:
            p = th.ones(len(values)) / len(values)
        else:
            if abs(1.0-sum(p)) > 1e-3:
                raise ValueError('Probs must sum to 1')
        self.p = p
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        shear_x = th_random_choice(self.values, p=self.p)
        shear_y = th_random_choice(self.values, p=self.p)

        if self.lazy:
            return Shear3D([shear_x, shear_y],
                         lazy=True)(inputs[0])
        else:
            outputs = Shear3D([shear_x, shear_y],
                            interp=self.interp)(*inputs)
            return outputs 


class Shear3D(object):

    def __init__(self,
                 value,
                 interp='trilinear',
                 lazy=False):
        if isinstance(value, (list, tuple)):
            self.value = value
        else:
            self.value = (value, 0)
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        if not isinstance(self.interp, (tuple,list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        theta_x = (math.pi * self.value[0]) / 180
        theta_y = (math.pi * self.value[1]) / 180
        shear_matrix = th.FloatTensor([[1,                  0,                 0, 0],
                                       [0,  math.cos(theta_x), math.sin(theta_y), 0],
                                       [0, -math.sin(theta_x), math.cos(theta_y), 0],
                                       [0,                  0,                 0, 1]])
        if self.lazy:
            return shear_matrix
        else:
            outputs = []
            for idx, _input in enumerate(inputs):
                input_tf = th_affine3d(_input,
                                       shear_matrix,
                                       mode=interp[idx],
                                       center=True)
                outputs.append(input_tf)
            return outputs if idx >= 1 else outputs[0]


class RandomZoom3D(object):

    def __init__(self, 
                 zoom_range,
                 interp='trilinear',
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

        interp : string in {'trilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['trilinear','nearest']

        lazy    : boolean
            if false, perform the transform on the tensor and return the tensor
            if true, only create the affine transform matrix and return that
        """
        if not isinstance(zoom_range, list) and not isinstance(zoom_range, tuple):
            raise ValueError('zoom_range must be tuple or list with 2 values')
        self.zoom_range = zoom_range
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        zx = random.uniform(self.zoom_range[0], self.zoom_range[1])
        zy = random.uniform(self.zoom_range[0], self.zoom_range[1])
        zz = random.uniform(self.zoom_range[0], self.zoom_range[1])

        if self.lazy:
            return Zoom3D([zz, zy, zx], lazy=True)(inputs[0])
        else:
            outputs = Zoom3D([zz, zy, zx],
                           interp=self.interp)(*inputs)
            return outputs


class RandomChoiceZoom3D(object):

    def __init__(self, 
                 values,
                 p=None,
                 interp='trilinear',
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

        interp : string in {'trilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['trilinear','nearest']

        lazy    : boolean
            if false, perform the transform on the tensor and return the tensor
            if true, only create the affine transform matrix and return that
        """
        if isinstance(values, (list, tuple)):
            values = th.FloatTensor(values)
        self.values = values
        if p is None:
            p = th.ones(len(values)) / len(values)
        else:
            if abs(1.0-sum(p)) > 1e-3:
                raise ValueError('Probs must sum to 1')
        self.p = p
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        zx = th_random_choice(self.values, p=self.p)
        zy = th_random_choice(self.values, p=self.p)
        zz = th_random_choice(self.values, p=self.p)

        if self.lazy:
            return Zoom3D([zz, zy, zx], lazy=True)(inputs[0])
        else:
            outputs = Zoom3D([zz, zy, zx],
                           interp=self.interp)(*inputs)
            return outputs


class Zoom3D(object):

    def __init__(self,
                 value,
                 interp='trilinear',
                 lazy=False):
        """
        Arguments
        ---------
        value : float
            Fractional zoom.
            =1 : no zoom
            >1 : zoom-in (value-1)%
            <1 : zoom-out (1-value)%

        interp : string in {'trilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['trilinear','nearest']

        lazy: boolean
            If true, just return transformed
        """

        if not isinstance(value, (tuple,list)):
            value = (value, value, value)
        self.value = value
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        if not isinstance(self.interp, (tuple,list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        zz, zy, zx = self.value
        zoom_matrix = th.FloatTensor([[zz, 0,  0, 0],
                                      [0, zy,  0, 0],
                                      [0,  0, zx, 0],
                                      [0,  0,  0, 1]])

        if self.lazy:
            return zoom_matrix
        else:
            outputs = []
            for idx, _input in enumerate(inputs):
                input_tf = th_affine3d(_input,
                                       zoom_matrix,
                                       mode=interp[idx],
                                       center=True)
                outputs.append(input_tf)
            return outputs if idx >= 1 else outputs[0]


