import shutil
import PyInstaller.__main__

from _imports import *

def main(target_file, icon="", onefile=True):
    # define paths
    name, ext = target_file.rsplit(".", 1)
    target_path = os.path.join(WORKDIR, target_file)
    dirpath = WORKDIR
    
    compilepath = os.path.join(dirpath, 'compiled')
    workpath = os.path.join(compilepath, "work")
    distpath = os.path.join(compilepath, "dist")
    specpath = os.path.join(compilepath, "spec")
    
    # construct arguments
    args = [
        target_path,
        f'--workpath={workpath}',
        f'--distpath={distpath}',
        f'--specpath={specpath}',
        '--clean',
        '--noconfirm',
        ]
    if icon:
        args.append("--icon="+icon)
    
    if onefile: args.append('--onefile')

    # compile
    PyInstaller.__main__.run(args)
    delete_folder(workpath) # delete temporary files

if __name__ == "__main__":
##    main("graphical_interface.py")
    main("graphical_interface3.py")
