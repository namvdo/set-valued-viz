from _imports import *

from normals_model import ModelConfiguration as NormalsModel
from mask_model import ModelConfiguration as MaskModel


class ConsoleInterface(ConsoleInterface):
    # call with .start(<menu function>)
    
    def __init__(self):
        self.normals_model = NormalsModel()
        self.mask_model = MaskModel()
    
    save_directory = os.path.join(WORKDIR, "saves")
    def examplemenu(self) -> (str, list):
        desc = "An example description about the available options"
        options = [
            ("Exit", self._quit),
            ]
        return desc, options



if __name__ == "__main__":
    ConsoleInterface().start("examplemenu")
