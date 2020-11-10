# Semi-Supervised Classification with Graph Convolutional Networks
review of the projects is uploaded with pdf format named project_review due to the difficulties of typing equations through markdown:(
## Description of the content of the repo
gcn.py is the python code for the model.
img folder contains the equation files decribed in the review of the project.
project_review is the review and summary of the Semi-Supervised Classification with Graph Convolutional Networks.
## Re-implementation of the project
I wanted to implement graph convolutional networks, so it is different from Semi-Supervised Classification described in the paper. To use simple datasets, I used image data(Omniglot) and convert it to a graph data. However, it will not work well since it is not trained with sufficient amount of datasets.
## How to run the code
data_sampler load the data. To load the data(Omniglot), we need to assign root_dir(where data is stored) to Omniglot class.
Also, set num to 1(I think it will not work well when we assign bigger than 1 to num)
## Reference
For the code, I referred to the link below.
https://github.com/tkipf/pygcn
