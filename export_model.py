# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Exports trained model to TensorFlow frozen graph."""

import os
import sys
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from deeplab import common
from deeplab import input_preprocess
from deeplab import model


class Segmentation_PB_creator:

    def __init__(self,
                crop_size,
                num_classes,
                checkpoint_path,
                export_path,
                min_resize_value=None,
                max_resize_value=None,
                resize_factor=None,
                model_variant="xception_65",
                atrous_rates=[6,12,18],
                output_stride=128,
                inference_scales=[1,],
                image_pyramid=None,
                add_flipped_images=False,
                ):
        self.crop_size = crop_size
        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path
        self.export_path = export_path
        self.min_resize_value = min_resize_value
        self.max_resize_value = max_resize_value
        self.resize_factor = resize_factor
        self.model_variant = model_variant
        self.atrous_rates = atrous_rates
        self.output_stride = output_stride
        self.inference_scales = inference_scales
        self.image_pyramid = image_pyramid
        self.add_flipped_images = add_flipped_images
        self._INPUT_NAME = 'ImageTensor'
        self._OUTPUT_NAME = 'SemanticPredictions'



    def createPBFile(self):
      tf.logging.set_verbosity(tf.logging.INFO)
      tf.logging.info('Prepare to export model to: %s', self.export_path)

      with tf.Graph().as_default():
        input_image = tf.placeholder(tf.uint8, [1, None, None, 3], name=self._INPUT_NAME)
        image_size = tf.shape(input_image)[1:3]
        image = tf.squeeze(input_image, axis=0)
        resized_image, image, _ = input_preprocess.preprocess_image_and_label(
              image,
              label=None,
              crop_height=self.crop_size[0],
              crop_width=self.crop_size[1],
              min_resize_value=self.min_resize_value,
              max_resize_value=self.max_resize_value,
              resize_factor=self.resize_factor,
              is_training=False,
              model_variant=self.model_variant)
        resized_image_size = tf.shape(resized_image)[:2]

        image = tf.expand_dims(image, 0)

        model_options = common.ModelOptions(
            outputs_to_num_classes={common.OUTPUT_TYPE: self.num_classes},
            crop_size=self.crop_size,
            atrous_rates=self.atrous_rates,
            output_stride=self.output_stride)

        if tuple(self.inference_scales) == (1.0,):
          tf.logging.info('Exported model performs single-scale inference.')
          predictions = model.predict_labels(
              image,
              model_options=model_options,
              image_pyramid=self.image_pyramid)
        else:
          tf.logging.info('Exported model performs multi-scale inference.')
          predictions = model.predict_labels_multi_scale(
              image,
              model_options=model_options,
              eval_scales=self.inference_scales,
              add_flipped_images=self.add_flipped_images)

        predictions = tf.cast(predictions[common.OUTPUT_TYPE], tf.float32)
        # Crop the valid regions from the predictions.
        semantic_predictions = tf.slice(
            predictions,
            [0, 0, 0],
            [1, resized_image_size[0], resized_image_size[1]])
        # Resize back the prediction to the original image size.
        def _resize_label(label, label_size):
          # Expand dimension of label to [1, height, width, 1] for resize operation.
          label = tf.expand_dims(label, 3)
          resized_label = tf.image.resize_images(
              label,
              label_size,
              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
              align_corners=True)
          return tf.cast(tf.squeeze(resized_label, 3), tf.int32)
        semantic_predictions = _resize_label(semantic_predictions, image_size)
        semantic_predictions = tf.identity(semantic_predictions, name=self._OUTPUT_NAME)

        print("Before created saver")
        #saver = tf.train.Saver(tf.model_variables())
        saver = tf.train.Saver()

        tf.gfile.MakeDirs(os.path.dirname(self.export_path))
        print("Here after created saver")

        freeze_graph.freeze_graph_with_def_protos(
            tf.get_default_graph().as_graph_def(add_shapes=True),
            saver.as_saver_def(),
            self.checkpoint_path,
            self._OUTPUT_NAME,
            restore_op_name=None,
            filename_tensor_name=None,
            output_graph=self.export_path,
            clear_devices=True,
            initializer_nodes=None)

    
def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    crop_size_w = int(sys.argv[1])
    crop_size_h = int(sys.argv[2])
    crop_size = [crop_size_w,crop_size_h]
    num_classes= int(sys.argv[3])
    checkpoint_dir=sys.argv[4]
    export_path = sys.argv[5]
    file_names = [os.path.join(checkpoint_dir,i) for i in os.listdir(checkpoint_dir) if "model.ckpt" in i and ".meta" in i]
    model_file = max(file_names,key=os.path.getctime).split(".meta")[0]

    print("----------------->")
    print("w:{},h:{}".format(crop_size_w,crop_size_h))
    print("model_file:{}".format(model_file))
    print("num class:{}".format(num_classes))
    print("model_file:{}".format(model_file))
    print("export_path:{}".format(export_path))
    print("<----------------")

    seg_pb_creator = Segmentation_PB_creator(crop_size,num_classes,model_file,export_path)
    seg_pb_creator.createPBFile()







if __name__ == '__main__':
    main()
