
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def nearest_upsampling(data, scale):
  with tf.name_scope('nearest_upsampling'):
      bs, h, w, c = tf.unstack(tf.shape(data))
      output = tf.stack([data] * scale, axis=3)
      output = tf.stack([output] * scale, axis=2)
      return tf.reshape(output, [bs, h * scale, w * scale, c])

  return tf.reshape(data, [bs, h * scale, w * scale, c])


def selective_crop_and_resize(features,
                              boxes,
                              box_levels,
                              boundaries,
                              output_size=7,
                              is_gpu_inference=False):

  (batch_size, num_levels, max_feature_height, max_feature_width,
   num_filters) = features.get_shape().as_list()
  _, num_boxes, _ = boxes.get_shape().as_list()

  box_grid_x = []
  box_grid_y = []
  for i in range(output_size):
    box_grid_x.append(boxes[:, :, 1:2] +
                      (i + 0.5) * boxes[:, :, 3:4] / output_size)
    box_grid_y.append(boxes[:, :, 0:1] +
                      (i + 0.5) * boxes[:, :, 2:3] / output_size)
  box_grid_x = tf.concat(box_grid_x, axis=-1)
  box_grid_y = tf.concat(box_grid_y, axis=-1)
  tf.print("GRIDS:", box_grid_x, box_grid_y)
  # Compute indices for gather operation.
  box_grid_y0 = tf.floor(box_grid_y)
  box_grid_x0 = tf.floor(box_grid_x)
  tf.print("GRIDS-Floor:", box_grid_x0, box_grid_y0)
  box_grid_x0 = tf.maximum(0., box_grid_x0)
  box_grid_y0 = tf.maximum(0., box_grid_y0)
  tf.print("GRIDS-FLoor-Max:", box_grid_x0, box_grid_y0)
  box_gridx0x1 = tf.stack([
      tf.minimum(box_grid_x0, boundaries[:, :, 1:2]),
      tf.minimum(box_grid_x0 + 1, boundaries[:, :, 1:2])
  ],
                          axis=3)
  box_gridy0y1 = tf.stack([
      tf.minimum(box_grid_y0, boundaries[:, :, 0:1]),
      tf.minimum(box_grid_y0 + 1, boundaries[:, :, 0:1])
  ],
                          axis=3)


  tf.print("GRIDS-FLoor-Max-Min:", box_gridx0x1, box_gridy0y1)
  x_indices = tf.reshape(box_gridx0x1, [batch_size, num_boxes, output_size * 2])
  y_indices = tf.reshape(box_gridy0y1, [batch_size, num_boxes, output_size * 2])
  tf.print("X_Y INDICES", x_indices, y_indices)

  indices_dtype = tf.float32 if is_gpu_inference else tf.int32

  if not is_gpu_inference:
    x_indices = tf.cast(x_indices, tf.int32)
    y_indices = tf.cast(y_indices, tf.int32)

  height_dim_offset = max_feature_width
  level_dim_offset = max_feature_height * height_dim_offset
  batch_dim_offset = num_levels * level_dim_offset
  tf.print("OFFSETS:", height_dim_offset, level_dim_offset, batch_dim_offset)
  batch_dim_indices = (
      tf.reshape(tf.range(batch_size, dtype=indices_dtype) * batch_dim_offset, [batch_size, 1, 1, 1]) *
      tf.ones([1, num_boxes, output_size * 2, output_size * 2], dtype=indices_dtype)
  )

  box_level_indices = (
      tf.reshape(box_levels * level_dim_offset, [batch_size, num_boxes, 1, 1]) *
      tf.ones([1, 1, output_size * 2, output_size * 2], dtype=indices_dtype)
  )

  height_indices = (
      tf.reshape(y_indices * height_dim_offset, [batch_size, num_boxes, output_size * 2, 1]) *
      tf.ones([1, 1, 1, output_size * 2], dtype=indices_dtype)
  )

  width_indices = (
      tf.reshape(x_indices, [batch_size, num_boxes, 1, output_size * 2]) *
      tf.ones([1, 1, output_size * 2, 1], dtype=indices_dtype)
  )
  tf.print("INDICES:", batch_dim_indices, box_level_indices, height_indices, width_indices)
  # TODO(hongjunchoi): Remove the need for temporary variables as
  # temporary variables with

  if True:
      batch_dim_indices = tf.cast(batch_dim_indices, tf.float32)
      box_level_indices = tf.cast(box_level_indices, tf.float32)
      height_indices = tf.cast(height_indices, tf.float32)
      width_indices = tf.cast(width_indices, tf.float32)

      indices = tf.add_n([
          batch_dim_indices,
          box_level_indices,
          height_indices,
          width_indices,
      ])

      indices = tf.cast(indices, tf.int32)

  else:  # TODO: Restore this API int32 dtype will be supported on GPUs.
      indices = tf.add_n([
          batch_dim_indices,
          box_level_indices,
          height_indices,
          width_indices,
      ])

  tf.print("Added INDICES:", indices)
  if batch_size == 1:

    indices = tf.reshape(indices, [1, -1])

    if is_gpu_inference:
      indices = tf.cast(indices, dtype=tf.int32)


    tf.print("Added Reshaped INDICES:", indices)
    features = tf.reshape(features, [1, -1, num_filters])

    features_per_box = tf.gather(features, indices, axis=1)

  else:
    indices = tf.reshape(indices, [-1])

    if is_gpu_inference:
      indices = tf.cast(indices, dtype=tf.int32)

    features = tf.reshape(features, [-1, num_filters])
    features_per_box = tf.gather(features, indices)

  features_per_box = tf.reshape(
      features_per_box,
      [batch_size, num_boxes, output_size * 2, output_size * 2, num_filters]
  )
  tf.print("Reshaped Features:", features)
  tf.print("Reshaped features_per_box:", features_per_box)

  ly = box_grid_y - box_grid_y0
  lx = box_grid_x - box_grid_x0
  hy = 1.0 - ly
  hx = 1.0 - lx
  kernel_x = tf.reshape(tf.stack([hx, lx], axis=3), [batch_size, num_boxes, 1, output_size * 2])
  kernel_y = tf.reshape(tf.stack([hy, ly], axis=3), [batch_size, num_boxes, output_size * 2, 1])
  tf.print("KERNEL COORDS:", kernel_x, kernel_y)
  # Use implicit broadcast to generate the interpolation kernel. The
  # multiplier `4` is for avg pooling.
  interpolation_kernel = kernel_y * kernel_x * 4
  tf.print("Interpolate Kernel:", interpolation_kernel)
  # Interpolate the gathered features with computed interpolation kernels.
  features_per_box *= tf.cast(tf.expand_dims(interpolation_kernel, axis=4), dtype=features_per_box.dtype)
  tf.print("Interpolated features_per_box:", features_per_box)
  features_per_box = tf.reshape(
      features_per_box,
      [batch_size * num_boxes, output_size * 2, output_size * 2, num_filters]
  )
  tf.print("Reshaped Interpolated features_per_box:", features_per_box)
  features_per_box = tf.nn.avg_pool2d(features_per_box, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
  features_per_box = tf.reshape(features_per_box, [batch_size, num_boxes, output_size, output_size, num_filters])
  tf.print("Final Reshaped Averagepooled Interpolated features_per_box:", features_per_box)
  return features_per_box


def multilevel_crop_and_resize(features,
                               boxes,
                               output_size=7,
                               is_gpu_inference=False):

  with tf.name_scope('multilevel_crop_and_resize'):
      levels = features.keys()
      min_level = min(levels)
      max_level = max(levels)
      _, max_feature_height, max_feature_width, _ = (features[min_level].get_shape().as_list())
      tf.print("Features-Dict:", features)
      tf.print("BOXES:", boxes)
      # Stack feature pyramid into a features_all of shape
      # [batch_size, levels, height, width, num_filters].
      features_all = []
      for level in range(min_level, max_level + 1):
        features_all.append(tf.image.pad_to_bounding_box(features[level], 0, 0, max_feature_height, max_feature_width))

      features_all = tf.stack(features_all, axis=1)
      tf.print("FEATURES_ALL:", features_all)
      # Assign boxes to the right level.
      box_width = tf.squeeze(boxes[:, :, 3:4] - boxes[:, :, 1:2], axis=-1)
      box_height = tf.squeeze(boxes[:, :, 2:3] - boxes[:, :, 0:1], axis=-1)
      tf.print("Box Sizes:", box_height, box_width)
      areas_sqrt = tf.sqrt(box_height * box_width)

      levels = tf.math.floordiv(tf.math.log(tf.divide(areas_sqrt, 224.0)), tf.math.log(2.0)) + 4.0
      tf.print("LEVELS-1:", levels)
      if not is_gpu_inference:
        levels = tf.cast(levels, dtype=tf.int32)

      # Map levels between [min_level, max_level].
      levels = tf.minimum(
          float(max_level) if is_gpu_inference else max_level,
          tf.maximum(levels, float(min_level) if is_gpu_inference else min_level)
      )
      tf.print("LEVELS-2:", levels)
      # Project box location and sizes to corresponding feature levels.
      scale_to_level = tf.cast(
          tf.pow(tf.constant(2.0), levels if is_gpu_inference else tf.cast(levels, tf.float32)),
          dtype=boxes.dtype
      )
      tf.print("scale_to_level:", scale_to_level)
      boxes /= tf.expand_dims(scale_to_level, axis=2)
      tf.print("BOXES-After-Scaling:", boxes)
      box_width /= scale_to_level
      box_height /= scale_to_level
      tf.print("Box Sizes After Scaling:", box_height, box_width)
      boxes = tf.concat(
          [boxes[:, :, 0:2],
          tf.expand_dims(box_height, -1),
          tf.expand_dims(box_width, -1)],
          axis=-1
      )
      tf.print("BOXES-After-Scaling-Combine-Height-Width:", boxes)
      # Map levels to [0, max_level-min_level].
      levels -= min_level
      tf.print("level final:", levels)
      level_strides = tf.pow([[2.0]], levels if is_gpu_inference else tf.cast(levels, tf.float32))
      tf.print("level strides:", level_strides)
      boundary = tf.cast(
          tf.concat(
              [
                  tf.expand_dims([[tf.cast(max_feature_height, tf.float32)]] / level_strides - 1, axis=-1),
                  tf.expand_dims([[tf.cast(max_feature_width, tf.float32)]] / level_strides - 1, axis=-1),
              ],
              axis=-1
          ),
          boxes.dtype
      )
      tf.print("boundary:", boundary)

  return selective_crop_and_resize(features_all, boxes, levels, boundary, output_size, is_gpu_inference)
