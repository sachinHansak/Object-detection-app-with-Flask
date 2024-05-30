"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""

import os
import argparse
import io
from PIL import Image

import torch
from flask import Flask, render_template, request, redirect

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import requests
import inflect

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])

# Function for rendering the image results


def predict():
    
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return
        
        def render_results_in_image(in_pil_img, in_results):
            plt.figure(figsize=(16, 10))
            plt.imshow(in_pil_img)

            ax = plt.gca()

            for prediction in in_results:

                x, y = prediction['box']['xmin'], prediction['box']['ymin']
                w = prediction['box']['xmax'] - prediction['box']['xmin']
                h = prediction['box']['ymax'] - prediction['box']['ymin']

                ax.add_patch(plt.Rectangle((x, y),
                                        w,
                                        h,
                                        fill=False,
                                        color="green",
                                        linewidth=2))
                ax.text(
                x,
                y,
                f"{prediction['label']}: {round(prediction['score']*100, 1)}%",
                color='red'
                )

            plt.axis("off")

            # Save the modified image to a BytesIO object
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png',
                        bbox_inches='tight',
                        pad_inches=0)
            img_buf.seek(0)
            modified_image = Image.open(img_buf)

            # Close the plot to prevent it from being displayed
            plt.close()

            return modified_image   

        from transformers.utils import logging
        logging.set_verbosity_error()

        from transformers import pipeline

        # Load the model through pipeline
        od_pipe = pipeline("object-detection", "facebook/detr-resnet-50")
            
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        pipeline_output = od_pipe(img)

        processed_image = render_results_in_image(
            img,
            pipeline_output)
        #processed_image.save(save_dir="static/image0.jpg")
        processed_image.save(os.path.join("static/", "image0.png"))
        return render_template("free.html", image_path="static/image0.png")
        #return redirect("static/image0.png")

    return render_template("index1.html")

# @app.route('/')
# def output():
#     return render_template('show_image.html')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing models")
    parser.add_argument("--port", default=8000, type=int, help="port number")
    args = parser.parse_args()

    app.run(port=8000)  # debug=True causes Restarting with stat
    
