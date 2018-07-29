import os.path as osp
from simple_html import HTML


def writeHTML(out_path, image_indices, image_filenames, base_dir):
    h = HTML('html')
    h.p('Results')
    h.br()
    path = '.'
    t = h.table(border='1')
    for image_i in image_indices:
        r = t.tr()
        for image_filename in image_filenames:
            filename = str(image_i) + '_' + image_filename + '.png'
            img_path = osp.join(base_dir, filename)
            r.td().img(src=img_path, width='256', height='256')
    h.br()

    html_file = open(out_path, 'w')
    html_file.write(str(h))
    html_file.close()


if __name__ == '__main__':
    image_filenames = ['room_gt', 'corner_pred', 'Lianjia_pred']
    image_indices = [i for i in range(20) if i != 2 and i != 16]

    writeHTML(out_path='./Lianjia_results.html', image_indices=image_indices, image_filenames=image_filenames,
              base_dir='/Users/cjc/vision/FloorNet/Lianjia-floorplan-samples')
