# Derm Foundation serving

This folder contains the source code and configuration necessary to serve the
model on
[Vertex AI](https://cloud.google.com/vertex-ai/docs/predictions/use-custom-container).
The implementation follows this
[container architecture](https://developers.google.com/health-ai-developer-foundations/model-serving/container-architecture).

The serving container can be used in both online and batch prediction workflows:

*   **Online predictions**: Deploy the container as a
    [REST](https://en.wikipedia.org/wiki/REST) endpoint, like a
    [Vertex AI endpoint](https://cloud.google.com/vertex-ai/docs/general/deployment).
    This allows you to access the model for real-time predictions via the REST
    [Application Programming Interface (API)](https://developers.google.com/health-ai-developer-foundations/cxr-foundation/serving-api).

*   **Batch predictions**: Use the container to run large-scale
    [Vertex AI batch prediction jobs](https://cloud.google.com/vertex-ai/docs/predictions/get-batch-predictions)
    to process many inputs at once.

## Description of select files and folders

*   [`serving_framework/`](./serving_framework): A library for
    implementing Vertex-compatible HTTP servers.

*   [`vertex_schemata/`](./vertex_schemata): Folder containing YAML files that
    define the
    [PredictSchemata](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/PredictSchemata)
    for Vertex AI endpoints.

*   [`Dockerfile`](./Dockerfile): Defines the Docker image for serving the
    model.

*   [`entrypoint.sh`](./entrypoint.sh): A bash script that is used as the Docker
    entrypoint. It sets up the necessary environment variables, copies the
    TensorFlow [SavedModel(s)](https://www.tensorflow.org/guide/saved_model)
    locally and launches the TensorFlow server and the frontend HTTP server.

*   [`predictor.py`](./predictor.py): Prepares model input, calls the model, and
    post-processes the output into the final response.

*   [`requirements.txt`](./requirements.txt): Lists the required Python
    packages.

*   [`server_gunicorn.py`](./server_gunicorn.py): Creates the HTTP server that
    launches the prediction executor.

## Dependencies

*   [`data_processing/`](../data_processing): A library for data
    retrieval and processing.
