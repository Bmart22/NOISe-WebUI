# Instructions

Running the demo:

This project requires an environement with flask installed. A suitable requirements file is available in the main folder.

        conda env create --name <env_name> --file requirements.yml

In the environment, move to the main folder. You can run the application with:

        flask --app osteo_flask run --debug

Then, open the associated link.

On the initial page, you can upload an image from your computer. Then, press the "Inference" button to run inference on the image; depending on the size of the image, this may take some time. The results will be loaded on a new page in which the predicted bounding boxes and segmentation masks are overlayed on top of the original image. The boxes and masks can then be toggled on and off with the checkboxes.

![Image of application creation page](flask_tmp/static/demo.png "Title")

