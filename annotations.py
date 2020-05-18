import os


class Annotations:
    def __init__(self, folder: str) -> None:
        self.folder = folder

        self.ann_classes = {}

        self._list_annotations()

    def _list_annotations(self) -> None:
        ann_filename = os.path.join(self.folder, 'analysis/Annotations')

        if os.path.isfile(ann_filename):
            ann_file = open(ann_filename, 'r')
        else:
            raise RuntimeError('{}: Annotations file missing.'.format(self.folder))

    def add_class(self, class_name: str) -> None:
        pass

    def update_canvas(self) -> None:
        pass

    def save_to_file(self) -> None:
        ann_filename = os.path.join(self.folder, 'Annotations')
        ann_file = open(ann_filename, 'w')

        ann_file.write('{}\n'.format(self.folder))
        ann_file.write('Classes: {}\n'.format(self.ann_classes.keys()))

        for cls in self.ann_classes:
            ann_file.write('{{{}}}:\n{}'.format(cls, self.ann_classes[cls]))
        ann_file.close()

