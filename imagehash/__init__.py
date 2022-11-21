from __future__ import absolute_import, division, print_function

import sys
import onnxruntime
import numpy
from PIL import Image, ImageFilter
try:
	ANTIALIAS = Image.Resampling.LANCZOS
except AttributeError:
	# deprecated in pillow 10
	# https://pillow.readthedocs.io/en/stable/deprecations.html
	ANTIALIAS = Image.ANTIALIAS

__version__ = '4.3.1'

def _binary_array_to_hex(arr):
	bit_string = ''.join(str(b) for b in 1 * arr.flatten())
	width = int(numpy.ceil(len(bit_string) / 4))
	return '{:0>{width}x}'.format(int(bit_string, 2), width=width)


class ImageHash:
	def __init__(self, binary_array):
		# type: (NDArray) -> None
		self.hash = binary_array  # type: NDArray

	def __str__(self):
		return _binary_array_to_hex(self.hash.flatten())

	def __repr__(self):
		return repr(self.hash)

	def __sub__(self, other):
		# type: (ImageHash) -> int
		if other is None:
			raise TypeError('Other hash must not be None.')
		if self.hash.size != other.hash.size:
			raise TypeError('ImageHashes must be of the same shape.', self.hash.shape, other.hash.shape)
		return numpy.count_nonzero(self.hash.flatten() != other.hash.flatten())

	def __eq__(self, other):
		# type: (object) -> bool
		if other is None:
			return False
		return numpy.array_equal(self.hash.flatten(), other.hash.flatten())  # type: ignore

	def __ne__(self, other):
		# type: (object) -> bool
		if other is None:
			return False
		return not numpy.array_equal(self.hash.flatten(), other.hash.flatten())  # type: ignore

	def __hash__(self):
		# this returns a 8 bit integer, intentionally shortening the information
		return sum([2**(i % 8) for i, v in enumerate(self.hash.flatten()) if v])

	def __len__(self):
		# Returns the bit length of the hash
		return self.hash.size


# dynamic code for typing
try:
	# specify allowed values if possible (py3.8+)
	from typing import Literal
	WhashMode = Literal['haar', 'db4']  # type: ignore
except ImportError:
	WhashMode = str  # type: ignore

try:
	# enable numpy array typing (py3.7+)
	import numpy.typing
	NDArray = numpy.typing.NDArray[numpy.bool_]
except (AttributeError, ImportError):
	NDArray = list  # type: ignore

# type of Callable
if sys.version_info >= (3, 3):
	if sys.version_info >= (3, 9, 0) and sys.version_info <= (3, 9, 1):
		# https://stackoverflow.com/questions/65858528/is-collections-abc-callable-bugged-in-python-3-9-1
		from typing import Callable
	else:
		from collections.abc import Callable
	try:
		MeanFunc = Callable[[NDArray], float]
		HashFunc = Callable[[Image.Image], ImageHash]
	except TypeError:
		MeanFunc = Callable  # type: ignore
		HashFunc = Callable  # type: ignore
# end of dynamic code for typing

def average_hash(image, hash_size=8, mean=numpy.mean):
	# type: (Image.Image, int, MeanFunc) -> ImageHash
	
	if hash_size < 2:
		raise ValueError('Hash size must be greater than or equal to 2')

	# reduce size and complexity, then covert to grayscale
	image = image.convert('L').resize((hash_size, hash_size), ANTIALIAS)

	# find average pixel value; 'pixels' is an array of the pixel values, ranging from 0 (black) to 255 (white)
	pixels = numpy.asarray(image)
	avg = mean(pixels)

	# create string of bits
	diff = pixels > avg
	# make a hash
	return ImageHash(diff)


def phash(image, hash_size=8, highfreq_factor=4):
	if hash_size < 2:
		raise ValueError('Hash size must be greater than or equal to 2')

	import scipy.fftpack
	img_size = hash_size * highfreq_factor
	image = image.convert('L').resize((img_size, img_size), ANTIALIAS)
	pixels = numpy.asarray(image)
	dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0), axis=1)
	dctlowfreq = dct[:hash_size, :hash_size]
	med = numpy.median(dctlowfreq)
	diff = dctlowfreq > med
	return ImageHash(diff)


def phash_simple(image, hash_size=8, highfreq_factor=4):
	import scipy.fftpack
	img_size = hash_size * highfreq_factor
	image = image.convert('L').resize((img_size, img_size), ANTIALIAS)
	pixels = numpy.asarray(image)
	dct = scipy.fftpack.dct(pixels)
	dctlowfreq = dct[:hash_size, 1:hash_size + 1]
	avg = dctlowfreq.mean()
	diff = dctlowfreq > avg
	return ImageHash(diff)


def dhash(image, hash_size=8):
	if hash_size < 2:
		raise ValueError('Hash size must be greater than or equal to 2')

	image = image.convert('L').resize((hash_size + 1, hash_size), ANTIALIAS)
	pixels = numpy.asarray(image)
	# compute differences between columns
	diff = pixels[:, 1:] > pixels[:, :-1]
	return ImageHash(diff)


def dhash_vertical(image, hash_size=8):
	image = image.convert('L').resize((hash_size, hash_size + 1), ANTIALIAS)
	pixels = numpy.asarray(image)
	# compute differences between rows
	diff = pixels[1:, :] > pixels[:-1, :]
	return ImageHash(diff)


def whash(image, hash_size=8, image_scale=None, mode='haar', remove_max_haar_ll=True):
	import pywt
	if image_scale is not None:
		assert image_scale & (image_scale - 1) == 0, 'image_scale is not power of 2'
	else:
		image_natural_scale = 2**int(numpy.log2(min(image.size)))
		image_scale = max(image_natural_scale, hash_size)

	ll_max_level = int(numpy.log2(image_scale))

	level = int(numpy.log2(hash_size))
	assert hash_size & (hash_size - 1) == 0, 'hash_size is not power of 2'
	assert level <= ll_max_level, 'hash_size in a wrong range'
	dwt_level = ll_max_level - level

	image = image.convert('L').resize((image_scale, image_scale), ANTIALIAS)
	pixels = numpy.asarray(image) / 255.

	# Remove low level frequency LL(max_ll) if @remove_max_haar_ll using haar filter
	if remove_max_haar_ll:
		coeffs = pywt.wavedec2(pixels, 'haar', level=ll_max_level)
		coeffs = list(coeffs)
		coeffs[0] *= 0
		pixels = pywt.waverec2(coeffs, 'haar')

	# Use LL(K) as freq, where K is log2(@hash_size)
	coeffs = pywt.wavedec2(pixels, mode, level=dwt_level)
	dwt_low = coeffs[0]

	# Substract median and compute hash
	med = numpy.median(dwt_low)
	diff = dwt_low > med
	return ImageHash(diff)

def nhash(image):
	session = onnxruntime.InferenceSession(sys.argv[1])

	# Load output hash matrix
	seed1 = open(sys.argv[2], 'rb').read()[128:]
	seed1 = numpy.frombuffer(seed1, dtype=numpy.float32)
	seed1 = seed1.reshape([96, 128])

	# Preprocess image
	image = Image.open(sys.argv[3]).convert('RGB')
	image = image.resize([360, 360])
	arr = numpy.array(image).astype(np.float32) / 255.0
	arr = arr * 2.0 - 1.0
	arr = arr.transpose(2, 0, 1).reshape([1, 3, 360, 360])

	# Run model
	inputs = {session.get_inputs()[0].name: arr}
	outs = session.run(None, inputs)

	# Convert model output to hex hash
	hash_output = seed1.dot(outs[0].flatten())
	hash_bits = ''.join(['1' if it >= 0 else '0' for it in hash_output])
	hash_hex = '{:0{}x}'.format(int(hash_bits, 2), len(hash_bits) // 4)

	print(hash_hex)