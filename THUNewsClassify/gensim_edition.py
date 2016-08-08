
import os

class loadFolders(object):
    def __init__(self,par_path):
        self.par_path = par_path
    def __iter__(self):
        for file in os.listdir(self.par_path):
            file_abspath = os.path.join(self.par_path, file)
            if os.path.isdir(file_abspath): # if file is a folder
                yield file_abspath

class loadFiles(object):
    def __init__(self,par_path):
        self.par_path = par_path
    def __iter__(self):
        folders = loadFolders(self.par_path)
        for folder in folders:              # level directory
            for file in os.listdir(folder):     # secondary directory
                file_path = os.path.join(folder,file)
                if os.path.isfile(file_path):
                    content = open(file_path,'rb').read().decode('utf8')
                    yield content






