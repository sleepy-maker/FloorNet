import tensorflow as tf

import os
import cv2
import numpy as np
import math
import pdb

from plyfile import PlyData, PlyElement
import json

from utils import getDensity, drawDensityImage



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
        self.ply_base_dir = os.path.join(self.base_dir, 'ply')
        self.annot_base_dir = os.path.join(self.base_dir, 'json')
        self.phase = phase
        self.im_size = im_size  # HEIGHT, WIDTH = SIZE
        self.max_num_corners = max_num_corners

        self.ply_paths, self.annot_paths = self.get_filepaths()

        self.writer = tf.python_io.TFRecordWriter(self.base_dir + '_' + self.phase + '.tfrecords')

    def get_filepaths(self):
        ply_filenames = sorted(os.listdir(self.ply_base_dir))
        json_filenames = sorted(os.listdir(self.annot_base_dir))

        assert len(ply_filenames) == len(json_filenames)
        ply_file_paths = [os.path.join(self.ply_base_dir, filename) for filename in ply_filenames]
        annot_file_paths = [os.path.join(self.annot_base_dir, filename) for filename in json_filenames]

        return ply_file_paths, annot_file_paths

    def write(self):
        for ply_file_path, annot_file_path in zip(self.ply_paths, self.annot_paths):
            self.write_example(ply_file_path, annot_file_path)

    def write_example(self, ply_path, annot_path):
        points = read_scene_pc(ply_path)

        xyz = points[:, :3]

        mins = xyz.min(0, keepdims=True)
        maxs = xyz.max(0, keepdims=True)

        max_range = (maxs - mins)[:, :2].max()
        padding = max_range * 0.05
        
        mins = (maxs + mins) / 2 - max_range / 2
        mins -= padding
        max_range += padding * 2

        xyz = (xyz - mins) / max_range

        new_points = np.concatenate([xyz, points[:, 3:6]], axis=1)
        points = new_points

        if points.shape[0] < self.num_points:
            indices = np.arange(points.shape[0])
            points = np.concatenate([points, points[np.random.choice(indices, self.num_points - points.shape[0])]], axis=0)
        else:
            sampled_indices = np.arange(points.shape[0])
            np.random.shuffle(sampled_indices)
            points = points[sampled_indices[:self.num_points]]

        # For testing purpose: draw the density image to check the quality
        filename, _ = os.path.splitext(os.path.basename(ply_path))
        write_scene_pc(points, './debug/{}.ply'.format(filename))
        density_img = drawDensityImage(getDensity(points=points))
        cv2.imwrite('./debug/{}_density.png'.format(filename), density_img)

        density_img = np.stack([density_img]*3, axis=2)
        annot_image = self.parse_annot(density_img, annot_path, mins, max_range)
        cv2.imwrite('./debug/{}_annot.png'.format(filename), annot_image)



        # points[:, 3:] = points[:, 3:] / 255 - 0.5

        # coordinates = np.clip(np.round(points[:, :2] * self.im_size).astype(np.int32), 0, self.im_size - 1)

        # points_indices = self.get_projection_indices(coordinates)

        # # prepare other g.t. related inputs to be zeros for now

        # corner_gt = np.zeros([self.max_num_corners, 3], dtype=np.int64)

        # num_corners = 0

        # icon_segmentation = np.zeros((self.im_size, self.im_size), dtype=np.uint8)

        # room_segmentation = np.zeros((self.im_size, self.im_size), dtype=np.uint8)

        # flags = np.zeros(2, np.int64)
        # flags[0] = 1
        # flags[1] = 0

        # example = tf.train.Example(features=tf.train.Features(feature={
        #     'image_path': _bytes_feature(file_path),
        #     'points': _float_feature(points.reshape(-1)),
        #     'point_indices': _int64_feature(points_indices.reshape(-1)),
        #     'corner': _int64_feature(corner_gt.reshape(-1)),
        #     'num_corners': _int64_feature([num_corners]),
        #     'icon': _bytes_feature(icon_segmentation.tostring()),
        #     'room': _bytes_feature(room_segmentation.tostring()),
        #     'flags': _int64_feature(flags),
        # }))

        # self.writer.write(example.SerializeToString())

    def get_projection_indices(self, coordinates):
        indices_map = np.zeros([self.num_points], dtype=np.int64)
        for i, coord in enumerate(coordinates):
            x, y = coord
            indices_map[i] = y * self.im_size + x
        return indices_map

    def parse_annot(self, img, file_path, mins, max_range):
        with open(file_path, 'r') as f:
            data = json.load(f)

        points = data['points']
        lines = data['lines']
        line_items = data['lineItems']
        areas = data['areas']
            
        point_dict = dict()
        
        for point in points:
            point_dict[point['id']] = point

        line_dict = dict()
        for line in lines:
            line_dict[line['id']] = line

        # img = np.zeros([self.im_size, self.im_size, 3], dtype=np.uint8)

        min_x = mins[0][0]
        min_y = mins[0][1]
        width = height = max_range

        # draw all corners
        for point in points:
            img_x, img_y = self._draw_corner_with_scaling(img, (point['x'], point['y']), min_x, width, min_y, height)
            point_dict[point['id']]['img_x'] = img_x
            point_dict[point['id']]['img_y'] = img_y
        
        # draw all line segments
        for line in lines:
            assert len(line['points']) == 2
            point_id_1, point_id_2 = line['points']
            start_pt = (point_dict[point_id_1]['img_x'], point_dict[point_id_1]['img_y'])
            end_pt = (point_dict[point_id_2]['img_x'], point_dict[point_id_2]['img_y'])
            # line_dict[line['id']]['img_start_pt'] = start_pt
            # line_dict[line['id']]['img_end_pt'] = end_pt
            cv2.line(img, start_pt, end_pt, (255,0,0))

        # draw all line with labels, such as doors, windows
        for line_item in line_items:
            start_pt = (line_item['startPointAt']['x'], line_item['startPointAt']['y'])
            end_pt = (line_item['endPointAt']['x'], line_item['endPointAt']['y'])
            img_start_pt = self._draw_corner_with_scaling(img, start_pt, min_x, width, min_y, height, color=(0,255,0))
            img_end_pt = self._draw_corner_with_scaling(img, end_pt, min_x, width, min_y, height, color=(0,255,0))
            cv2.line(img, img_start_pt, img_end_pt, (0, 255, 255))
            cv2.putText(img, line_item['is'], (img_start_pt[0], img_start_pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

        print(len(areas))
        return img

    def _draw_corner_with_scaling(self, img, corner, min_x, width, min_y, height, color=(0,0,255)):
        img_x = int(math.floor((corner[0] - min_x) * 1.0 / width * self.im_size))
        img_y = int(math.floor((corner[1] - min_y) * 1.0 / height * self.im_size))
        cv2.circle(img, (img_x,img_y), 2, color, -1)
        return img_x, img_y



if __name__ == '__main__':
    base_dir = '/local-scratch/cjc/Lianjia-inverse-cad/FloorNet/data/first_500/processed_test'
    record_writer = RecordWriter(num_points=50000, base_dir=base_dir, phase='test', im_size=256)
    record_writer.write()

