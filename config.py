# add 'elephant' to pascal voc 2007 labels
labels = ('aeroplane', 'bicycle', 'bird', 'boat',
          'bottle', 'bus', 'car', 'cat',
          'chair', 'cow', 'diningtable', 'dog',
          'elephant', 'horse', 'motorbike', 'person',
          'pottedplant', 'sheep', 'sofa', 'train',
          'tvmonitor')

label_map = {k: v + 1 for v, k in enumerate(labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}

distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8',
                   '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080',
                   '#aa6e28', '#fffac8', '#800000', '#aaffc3',
                   '#808000', '#ffd8b1', '#e6beff', '#808080',
                   '#FFFFFF', '#19caad']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}
