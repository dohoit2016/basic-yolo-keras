from flask_cors import CORS
from flask import Flask
from flask import request
from flask import Response
from flask import jsonify

from six.moves.urllib.request import urlretrieve
# from urllib.parse import urlsplit
import json

from frontend import YOLO
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

app = Flask(__name__)
cors = CORS(app)

data_dir = './tmp'
config_path  = "config.json"
weights_path = "full_yolo_hand.h5"
config = None
yolo = None

def create_graph():
	global yolo, config
	with open(config_path) as config_buffer:
		config = json.load(config_buffer)

	yolo = YOLO(architecture=config['model']['architecture'],
				input_size=config['model']['input_size'],
				labels=config['model']['labels'],
				max_box_per_image=config['model']['max_box_per_image'],
				anchors=config['model']['anchors'])
	print weights_path
	yolo.load_weights(weights_path)


@app.route('/api/handdetection', methods=['POST'])
def hand_detect():
	global yolo, config
	body = request.get_data().decode('utf-8').replace("'",'"')

	if body == "":
		return jsonify(error="Image url not found")
	else:
		url = json.loads(body, encoding='utf-8')

	url = str(url)
	file_name = url.split('/')[-1]
	dest_file = os.path.join(data_dir, file_name)
	print dest_file
	dic = {
		'url' : url
	}
	try:
		# parsed_link = urlsplit(url)
		# parsed_link = list(parsed_link)
		# parsed_link[2] = urllib.parse.quote(parsed_link[2])
		# encoded_link = urllib.parse.urlunsplit(parsed_link)
		image_name, _ = urlretrieve(url, dest_file)

		image = cv2.imread(dest_file)
		boxes = yolo.predict(image)
		print "starting draw boxes..."
		image = draw_boxes(image, boxes, config['model']['labels'])
		print len(boxes), 'boxes are found'

		cv2.imwrite(data_dir + "/" + file_name[:-4] + '_detected' + file_name[-4:], image)
		# os.remove(dest_file)
		dic['label'] = 'hand' if len(boxes) > 0 else 'nonhand'
	except Exception as e:
		print e
		pass
	return jsonify(dic)


if __name__ == '__main__':
	create_graph()
	app.run(host='0.0.0.0')
