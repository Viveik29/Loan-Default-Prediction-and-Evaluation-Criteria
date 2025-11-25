from setuptools import find_packages ,setup
import os


requirement = []
Hypen_E= '-e .'
def get_requirements(file_path):

    
    if os.path.isfile(file_path):
        with open(file_path) as f:
            requirement = f.read().splitlines()
            if Hypen_E in requirement:
             requirement = requirement.remove(Hypen_E)  
        return requirement
    
                  
setup(

name='ML Project',
version= '0.0.0.1',
author='vivek',
author_email= 'viveik16693@gmail.com',
packages= find_packages(),
install_requires = get_requirements('requirements.txt')


)
if __name__=="__main__":
   print(requirement)