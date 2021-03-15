Follow steps:


1) Download and Install Anaconda
2) Create a new env for python 3.8, called 'arc105' (avoiding base env, since it contains a lot of libs)
3) In anaconda prompt, chage to new env: 

> conda activate 105

and install pytorch using:

> conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

4) There are some remains dependencies to install:
in anaconda environments, search for 

> rasterio
> shapely
> pyshp
> scikit-image
> tensorboard
> scikit-learn
> tqdm
> imgaug
