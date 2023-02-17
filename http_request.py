from lib2to3.pgen2 import token
import os

import uuid
from http import HTTPStatus
from dotenv import load_dotenv

import requests
import json
import flask
import torch
from meta2d.common.config_2d import Config2D
from meta2d.services.image_service import get_image_url
from meta2d.services.image_service import text_to_image, get_client, get_image_key
from meta3d.common.config_3d import Config3D
from meta3d.services import Meta3dService, PointCloudSampler
from meta3d.services.s3_service import upload_file, get_url, get_model_image
from waitress import serve
import hashlib


load_dotenv()
flag = True
flag2 = True
# create a flask app
app = flask.Flask(__name__)


# test image to hash and post
@app.route('/image_test', methods=['POST'])
def image_test():
    with open("test.jpg", 'rb') as f:
        image_content = f.read()

    image_content_hash = hashlib.sha256(image_content).hexdigest()
    url = "https://test-nft-cyberport.s3.ap-east-1.amazonaws.com/b/" + "11156" + ".png"
    # return flask.jsonify({'HOME PAGE': image_content_hash}), HTTPStatus.CREATED

    user_data = {"name": "user_name", "NFTurl": url}
    token_data = {"hash": image_content_hash, "NFTurl": url}

    user_response = requests.post(NFT_USER_URL, json=user_data)
    if user_response.status_code == 201:
        response = requests.post(NFT_TOKEN_URL, json=token_data)
        if response.status_code == 201:
            response_dict = json.loads(response.text)

            token = response_dict['tokenId']

            return flask.jsonify({'url': url}, {'tokenId': token}), HTTPStatus.CREATED
        else:
            response_dict = json.loads(response.text)
            return flask.jsonify({'statusCode': response_dict['statusCode']}, {'message': response_dict['message']}, {'error': response_dict['error']}), HTTPStatus.CREATED
    else:
        user_response_dict = json.loads(user_response.text)
        return flask.jsonify({'statusCode': user_response_dict['statusCode']}, {'message': user_response_dict['message']}, {'error': user_response_dict['error']}), HTTPStatus.CREATED


# write a home to show the home page
@app.route('/', methods=['GET'])
def index():
    return flask.jsonify({'HOME PAGE': "my home"}), HTTPStatus.CREATED


@app.route('/env_test', methods=['GET'])
def env_test():
    return flask.jsonify({'THIS IS ENV TEST': NFT_TOKEN_URL}), HTTPStatus.CREATED


@app.route('/message_test', methods=['POST'])
def message_test():
    # get the request body
    req = flask.request.get_json()
    # get the message
    message = req['message']
    return flask.jsonify({'THIS IS YOUR MESSAGE': message}), HTTPStatus.CREATED



@app.route('/meta3d', methods=['POST'])
# set a flag to make sure meta3d only run once
def check_flag():
    global flag
    if flag:
        flag = False
        return meta3d()
    else:
        return flask.jsonify({'THIS IS YOUR MESSAGE': "meta3d is running"}), HTTPStatus.CREATED


def meta3d():
    global flag
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
    base_model, upsampler_model = service.load_model(
        device=device, model_path=model_path)
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

    image_content = service.generate_model_image(model=pc, grid_size=3)
    # change image_content<class 'matplotlib.figure.Figure'> into image_content_bytes
    image_content_bytes = service.convert_image_to_bytes(image_content)

    # get the hash of ply_result
    file_name = service.save_model2ply(pc, ply_path)
    with open(file_name, 'rb') as f:
        ply_result = f.read()
    sha256 = hashlib.sha256()
    sha256.update(ply_result)

    object_name = str(uuid.uuid4()) + ".ply"
    # upload the file to s3
    upload_file(file_name=file_name, object_name=object_name, bucketname=config_3d.BUCKET_NAME, region=config_3d.REGION,
                aws_access_key_id=config_3d.AWS_ACCESS_KEY_ID, aws_secret_access_key=config_3d.AWS_SECRET_ACCESS_KEY,
                endpoint_url=config_3d.ENDPOINT_URL)
    # get the url
    url = get_url(object_name=object_name,
                  bucketname=config_3d.BUCKET_NAME, download_endpoint=config_3d.DOWNLOAD_ENDPOINT)
    # upload the image to s3
    image_url = get_model_image(imageContent=image_content_bytes, object_name=object_name, region=config_3d.REGION,
                                bucketname=config_3d.BUCKET_NAME,
                                aws_access_key_id=config_3d.AWS_ACCESS_KEY_ID,
                                aws_secret_access_key=config_3d.AWS_SECRET_ACCESS_KEY,
                                endpoint_url=config_3d.ENDPOINT_URL, download_endpoint=config_3d.DOWNLOAD_ENDPOINT)

    user_data = {'name': 'user_name', 'NFTurl': url}
    token_data = {'hash': sha256.hexdigest(), 'NFT': url}

    user_response = requests.post(NFT_USER_URL, json=user_data)
    if user_response.status_code == 201:
        response = requests.post(NFT_TOKEN_URL, json=token_data)
        if response.status_code == 201:
            response_dict = json.loads(response.text)
            flag = True
            return flask.jsonify({'image_url': image_url}, {'ply_url': url}, {'tokenId': response_dict['tokenId']}), HTTPStatus.CREATED
        else:
            response_dict = json.loads(response.text)
            flag = True
            return flask.jsonify({'statusCode': response_dict['statusCode']}, {'message': response_dict['message']}, {'error': response_dict['error']}), HTTPStatus.CREATED
    else:
        user_response_dict = json.loads(user_response.text)
        flag = True
        return flask.jsonify({'statusCode': user_response_dict['statusCode']}, {'message': user_response_dict['message']}, {'error': user_response_dict['error']}), HTTPStatus.CREATED


@app.route('/meta2d', methods=['POST'])
def check_flag2():
    global flag2
    if flag2:
        flag2 = False
        return meta2d()
    else:
        return flask.jsonify({'THIS IS YOUR MESSAGE': "meta2d is running"}), HTTPStatus.CREATED


def meta2d():
    global flag2
    # get the request body
    req = flask.request.get_json()
    # get the message
    message = req['message']
    # message genetate image
    image_content = text_to_image(
        img_url=config_2d.IMAGE_URL, token=config_2d.TOKEN, prompt=message)
    # create s3 client
    client = get_client(region=config_2d.REGION, aws_access_key_id=config_2d.AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=config_2d.AWS_SECRET_ACCESS_KEY, endpoint_url=config_2d.ENDPOINT_URL)
    # use uuid as filename
    key = get_image_key()
    # get url
    url = get_image_url(client=client, key=key, imageContent=image_content, bucketname=config_2d.BUCKET_NAME,
                        download_endpoint=config_2d.DOWNLOAD_ENDPOINT)
    # get hsah
    sha256 = hashlib.sha256()
    sha256.update(image_content)

    # post to smart contract to get the tokenid
    user_data = {'name': 'user_name', 'NFTurl': url}
    token_data = {'hash': sha256.hexdigest(), 'NFT': url}

    user_response = requests.post(NFT_USER_URL, json=user_data)
    if user_response.status_code == 201:
        response = requests.post(NFT_TOKEN_URL, json=token_data)
        if response.status_code == 201:
            response_dict = json.loads(response.text)
            flag2 = True
            return flask.jsonify({'image_url': url}, {'ply_url': url}, {'tokenId': response_dict['tokenId']}), HTTPStatus.CREATED
        else:
            response_dict = json.loads(response.text)
            flag2 = True
            return flask.jsonify({'statusCode': response_dict['statusCode']}, {'message': response_dict['message']}, {'error': response_dict['error']}), HTTPStatus.CREATED
    else:
        user_response_dict = json.loads(user_response.text)
        flag2 = True
        return flask.jsonify({'statusCode': user_response_dict['statusCode']}, {'message': user_response_dict['message']}, {'error': user_response_dict['error']}), HTTPStatus.CREATED


if __name__ == '__main__':
    config_2d = Config2D(BUCKET_NAME=os.getenv("BUCKET_NAME", "bucket_1"),
                         TOKEN=os.getenv("TOKEN", "123"),
                         IMAGE_URL=os.getenv(
                             "IMAGE_URL", "http://localhost:5000"),
                         REGION=os.getenv("REGION", "as-1"),
                         AWS_ACCESS_KEY_ID=os.getenv(
                             "AWS_ACCESS_KEY_ID", "123"),
                         AWS_SECRET_ACCESS_KEY=os.getenv(
                             "AWS_SECRET_ACCESS_KEY", "123"),
                         ENDPOINT_URL=os.getenv(
                             "ENDPOINT_URL", "http://localhost:9000"),
                         DOWNLOAD_ENDPOINT=os.getenv("DOWNLOAD_ENDPOINT", "http://localhost:9000"))

    config_3d = Config3D(BUCKET_NAME=os.getenv("BUCKET_NAME", "bucket_1"),
                         REGION=os.getenv("REGION", "as-1"),
                         BUCKET_model_folder=os.getenv(
                             "BUCKET_model_folder", "model_folder"),
                         AWS_ACCESS_KEY_ID=os.getenv(
                             "AWS_ACCESS_KEY_ID", "123"),
                         AWS_SECRET_ACCESS_KEY=os.getenv(
                             "AWS_SECRET_ACCESS_KEY", "123"),
                         ENDPOINT_URL=os.getenv(
                             "ENDPOINT_URL", "http://localhost:9000"),
                         DOWNLOAD_ENDPOINT=os.getenv("DOWNLOAD_ENDPOINT", "http://localhost:9000"))

    NFT_USER_URL = os.getenv("NFT_USER_URL", "http://localhost:5000")
    NFT_TOKEN_URL = os.getenv("NFT_TOKEN_URL", "http://localhost:5000")

    serve(app, host='0.0.0.0', port=5000, threads=2)
    # app.run(host='0.0.0.0', port=5000, debug=True)
