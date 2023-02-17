import os
from http import HTTPStatus

import flask
import torch
from meta2d.common.config_2d import Config2D
from meta2d.services.image_service import get_image_url
from meta2d.services.image_service import text_to_image, get_client, get_image_key
from meta3d.common.config_3d import Config3D
from meta3d.services import Meta3dService, PointCloudSampler
from meta3d.services.s3_service import upload_file, get_url
from waitress import serve

app = flask.Flask(__name__)


# write a home to show the home page
@app.route('/', methods=['GET'])
def index():
    return flask.jsonify({'url': "my home"}), HTTPStatus.CREATED


@app.route('/meta3d', methods=['POST'])
def meta3d():
    """
    Given the following request body
    {
        "message": "Hello world"
    }
    get the message and return it
    """
    # get the request body
    req = flask.request.get_json()
    # get the message
    message = req['message']

    prompt = message
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    service = Meta3dService(config=config_3d)
    model_path = "./"
    ply_path = './'

    service.check_model(model_path=model_path)
    base_model, upsampler_model = service.load_model(device=device, model_path=model_path)
    base_diffusion, upsampler_diffusion = service.create_diffusion()

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 0.0],
        model_kwargs_key_filter=('texts', ''),
    )
    pc = service.generate_3d_result(sampler, prompt=prompt)

    file_name = service.save_model2ply(pc, ply_path)

    # upload the file to s3
    object_name = upload_file(file_name=file_name, bucketname=config_3d.BUCKET_NAME)
    # get the url
    url = get_url(object_name=object_name, bucketname=config_3d.BUCKET_NAME, region=config_3d.REGION)

    return flask.jsonify({'url': url}), HTTPStatus.CREATED


# I want to run my server on http://localhost:5000
@app.route('/meta2d', methods=['POST'])
def meta2d():
    """
    Given the following request body
    {
        "message": "Hello world"
    }
    get the message and return it
    """
    # get the request body
    req = flask.request.get_json()
    # get the message
    message = req['message']
    # return the message
    image_content = text_to_image(img_url=config_2d.IMAGE_URL, token=config_2d.TOKEN, prompt=message)
    # create s3 client
    client = get_client(region=config_2d.REGION)
    # use uuid as filename
    key = get_image_key()
    # get url
    url = get_image_url(client=client, key=key, imageContent=image_content, bucketname=config_2d.BUCKET_NAME,
                        region=config_2d.REGION)
    return flask.jsonify({'url': url}), HTTPStatus.CREATED


# I want to run my server on http://localhost:5000
if __name__ == '__main__':
    config_2d = Config2D(BUCKET_NAME=os.getenv("BUCKET_NAME", "bucket_1"),
                         TOKEN=os.getenv("TOKEN", "123"),
                         IMAGE_URL=os.getenv("IMAGE_URL", "http://localhost:5000"),
                         REGION=os.getenv("REGION", "as-1"))

    config_3d = Config3D(BUCKET_NAME=os.getenv("BUCKET_NAME", "bucket_1"),
                         REGION=os.getenv("REGION", "as-1"),
                         BUCKET_model_folder=os.getenv("BUCKET_model_folder", "model_folder"))

    serve(app, host='0.0.0.0', port=5000, threads=2)