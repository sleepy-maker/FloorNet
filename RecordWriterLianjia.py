import os
import cv2
import numpy as np
import tensorflow as tf
from plyfile import PlyData, PlyElement
from utils import getDensity, drawDensityImage
import pdb


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def read_scene_pc(file_path):
    with open(file_path, 'rb') as f:
        plydata = PlyData.read(f)
        dtype = plydata['vertex'].data.dtype
    print('dtype of file{}: {}'.format(file_path, dtype))

    points_data = np.array(plydata['vertex'].data.tolist())

    return points_data


def write_scene_pc(points, output_path):
    vertex = np.array([tuple(x) for x in points],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ])
    vertex_el = PlyElement.describe(vertex, 'vertex')
    PlyData([vertex_el]).write(output_path)  # write the new ply file


class RecordWriter:
    def __init__(self, num_points, base_dir, phase, im_size, max_num_corners=300):
        self.num_points = num_points
        self.base_dir = base_dir
        self.phase = phase
        self.im_size = im_size  # HEIGHT, WIDTH = SIZE
        self.max_num_corners = max_num_corners

        self.file_paths = self.get_pc_filepaths()

        self.writer = tf.python_io.TFRecordWriter(self.base_dir + '_' + self.phase + '.tfrecords')

    def get_pc_filepaths(self):
        filenames = os.listdir(self.base_dir)
        file_paths = [os.path.join(self.base_dir, filename) for filename in filenames]
        return file_paths

    def write(self):
        for file_path in self.file_paths:
            self.write_example(file_path)

    def write_example(self, file_path):
        points = read_scene_pc(file_path)

        axis_trans_mat = np.array([[0, 0, 1, 0],
                                   [-1, 0, 0, 0],
                                   [0, -1, 0, 0],
                                   [0, 0, 0, 1]])

        xyz = points[:, :3]
        xyz = np.concatenate([xyz, np.ones([xyz.shape[0], 1])], axis=1)

        transformed_xyz = np.matmul(axis_trans_mat, xyz.transpose([1, 0])).transpose([1, 0])
        transformed_xyz = transformed_xyz[:, :3]

        mins = transformed_xyz.min(0, keepdims=True)
        maxs = transformed_xyz.max(0, keepdims=True)

        max_range = (maxs - mins)[:, :2].max()
        padding = max_range * 0.05
        mins = (maxs + mins) / 2 - max_range / 2
        mins -= padding
        max_range += padding * 2
        transformed_xyz = (transformed_xyz - mins) / max_range

        new_points = np.concatenate([transformed_xyz, points[:, 3:6]], axis=1)
        points = new_points

        if points.shape[0] < self.num_points:
            indices = np.arange(points.shape[0])
            points = np.concatenate([points, points[np.random.choice(indices, self.num_points - points.shape[0])]], axis=0)
        else:
            sampled_indices = np.arange(points.shape[0])
            np.random.shuffle(sampled_indices)
            points = points[sampled_indices[:self.num_points]]

        # For testing purpose: draw the density image to check the quality
        # write_scene_pc(points, './test.ply')
        # density_img = drawDensityImage(getDensity(points=points))
        # cv2.imwrite('./test_density.png', density_img)

        points[:, 3:] = points[:, 3:] / 255 - 0.5

        coordinates = np.clip(np.round(points[:, :2] * self.im_size).astype(np.int32), 0, self.im_size - 1)

        points_indices = self.get_projection_indices(coordinates)

        # prepare other g.t. related inputs to be zeros for now

        corner_gt = np.zeros([self.max_num_corners, 3], dtype=np.int64)

        num_corners = 0

        icon_segmentation = np.zeros((self.im_size, self.im_size), dtype=np.uint8)

        room_segmentation = np.zeros((self.im_size, self.im_size), dtype=np.uint8)

        flags = np.zeros(2, np.int64)
        flags[0] = 1
        flags[1] = 0

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_path': _bytes_feature(file_path),
            'points': _float_feature(points.reshape(-1)),
            'point_indices': _int64_feature(points_indices.reshape(-1)),
            'corner': _int64_feature(corner_gt.reshape(-1)),
            'num_corners': _int64_feature([num_corners]),
            'icon': _bytes_feature(icon_segmentation.tostring()),
            'room': _bytes_feature(room_segmentation.tostring()),
            'flags': _int64_feature(flags),
        }))

        self.writer.write(example.SerializeToString())

    def get_projection_indices(self, coordinates):
        indices_map = np.zeros([self.num_points], dtype=np.int64)
        for i, coord in enumerate(coordinates):
            x, y = coord
            indices_map[i] = y * self.im_size + x
        return indices_map


if __name__ == '__main__':
    base_dir = '/local-scratch/cjc/FloorNet/data/Lianjia-samples'
    record_writer = RecordWriter(num_points=50000, base_dir=base_dir, phase='test', im_size=256)
    record_writer.write()
