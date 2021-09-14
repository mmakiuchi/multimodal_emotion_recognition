"""
    Name: organize_folds.py
    Description: Organize the spmel directories in the 5 folds
"""

import argparse
import shutil

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--spmel_dir", type=str, help="directory of the spmel files")
    parser.add_argument("--folds_dir", type=str, help="directory to store the folds")

    config = parser.parse_args()

    src_dir = './' + config.spmel_dir
    folds_dir = "./" + config.folds_dir

    for session in range(1,6):
        for fold in range(1, 6):
            if fold == session:
                shutil.copytree(src_dir + "/Session" + str(session), folds_dir + "/fold" + str(fold) + "/test/Session" + str(session))
            else:
                shutil.copytree(src_dir + "/Session" + str(session), folds_dir + "/fold" + str(fold) + "/train/Session" + str(session))

if __name__ == '__main__':
    main()