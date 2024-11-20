# Derm Foundation Serving

This is a container implementation that can serve the model and is used for
Vertex AI serving.

## Description for select files and folders

*   [`Dockerfile`](./Dockerfile): This file defines the Docker image that will
    be used to serve the model. It includes the necessary dependencies, such as
    TensorFlow and the model server.
*   [`requirements.txt`](./requirements.txt): This file lists the Python
    packages that are required to run the model server.
*   [`prediction_executor.py`](./prediction_executor.py): This file contains the
    code for the prediction executor, which is responsible for loading the model
    and running predictions.
*   [`data_processing`](./data_processing): This folder contains code for
    processing the CXR foundation dataset.
*   [`prediction_container`](./prediction_container): This folder contains code
    for a container that can be used to serve predictions from any model.
