import PyQt4
from PyQt4.QtGui import QImage
import os
img=QImage()

for root, dirs, files in os.walk("/home/zlz/PycharmProjects/FlapPyBird-master/assets/sprites"):
    for file in files:
        if ".py" not in file:
            img.load(file)
            img.save(file)
