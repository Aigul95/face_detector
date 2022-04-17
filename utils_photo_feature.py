import sys

import cv2
import numpy as np
import pandas as pd
import PIL
import torch
from PIL import Image
from skimage.transform import resize
from torch import nn
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms as transforms_torchvision

FACE_PROB_THRESH = 0.5
FACE_SQUARE_THRESH = 0.008
IMG_Y = 300


device = torch.device("cuda:0")

NUM_CLASSES = 55
GRAYSCALE = False

TEMPLATE = pd.read_csv(
    "../everyday_vectorization/scripts/TEMPLATE.csv", names=[0, 1], sep=","
).values
TEMPLATE = np.float32(TEMPLATE)

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)
INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
OUTER_EYES_AND_NOSE = [36, 45, 33]

np.set_printoptions(precision=3, suppress=True)

SHAPE_PREDICTOR = "../everyday_vectorization/scripts/shape_predictor_68_face_landmarks.dat"
MODEL_PARAMETERS = "../everyday_vectorization/scripts/human_vs_subhuman_baseline_parameters.h5"
MODEL_WEIGHTS = "../everyday_vectorization/scripts/human_vs_subhuman_baseline_weights.h5"
IMAGE_DIM = 160


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    :param mat: image to rotate
    :param angle: angle to rotate
    :return: rotated image
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
        width / 2,
        height / 2,
    )  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def nwh(img):
    """ 
    Calculate part of white pixels
    :param img
    :return: part of white pixels
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_gray is not None:
        Nw = np.sum(img_gray == 255)
        return Nw / (img_gray.shape[0] * img_gray.shape[1])


def get_hist(img):
    """ 
    Calculate hist of image with Hue, Saturation channels
    :param img: image BGR
    :return: hist
    """
    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]
    # hue varies from 0 to 179, saturation from 0 to 255
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges  # concat lists
    # Use the 0-th and 1-st channels
    channels = [0, 1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def get_hist_bg(img, mask):
    """ 
    Calculate hist of person background (MaskRCNN) with Hue, Saturation channels
    :param img: image BGR
    :param mask: inverted output of MaskRCNN with max area
    :return: hist
    """
    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]
    # hue varies from 0 to 179, saturation from 0 to 255
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges  # concat lists
    # Use the 0-th and 1-st channels
    channels = [0, 1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img], channels, mask, histSize, ranges, accumulate=False)
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def get_large_bbox_face_detector(output):
    """ 
    Calculate bbox with max area by output of SSD face detector
    :param output: output of SSD
    :return: bbox: bbox with max area
    """
    bbox_large = output["detection_boxes"][0]
    if len(output) != 0:
        indices = np.where(output["detection_scores"] > 0.5)[0]
        if len(indices) > 1:
            s = []
            for box in output["detection_boxes"][indices]:
                ymin, xmin, ymax, xmax = box
                s.append((ymax - ymin) * (xmax - xmin))
            i_max = np.argmax(np.array(s))
            bbox_large = output["detection_boxes"][i_max]
    return bbox_large


def get_bboxes_face_detector(output):
    """
    Calculate bbox with max area by output of SSD face detector
    :param output: output of SSD
    :return: bbox: bbox with score more then 0.5
    """
    bboxes = []
    if len(output) != 0:
        indices = np.where(output["detection_scores"] > 0.5)[0]
        if len(indices) > 0:
            bboxes = output["detection_boxes"][indices]
    return bboxes


######
# CLASSES FOR AGE REGRESSOR
######


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, grayscale):
        self.num_classes = num_classes
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1, padding=2)
        self.fc = nn.Linear(2048 * block.expansion, 1, bias=False)
        self.linear_1_bias = nn.Parameter(torch.zeros(self.num_classes - 1).float())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2.0 / n) ** 0.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        logits = logits + self.linear_1_bias
        probas = torch.sigmoid(logits)
        return logits, probas


def resnet34(num_classes, grayscale):
    """Constructs a ResNet-34 model."""
    model = ResNet(
        block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes, grayscale=grayscale
    )
    return model


def init_age_model(path_to_model, map_loc):
    model_age = resnet34(NUM_CLASSES, GRAYSCALE)
    # model_age.to(device)
    checkpoint = torch.load(
        path_to_model, map_location=torch.device("cpu")
    )  # , map_location=map_loc)
    model_age.load_state_dict(checkpoint["model_state_dict"])
    return model_age


def model_forward_age(image, model):
    """ 
    inference of age regressor
    :param image: cropped face image BGR
    :return: probas, predicted_age
    """
    tfs = transforms_torchvision.Compose(
        [transforms_torchvision.Resize((120, 120)), transforms_torchvision.ToTensor()]
    )
    image = Image.fromarray(image)
    image = tfs(image)
    model.eval()
    image = image.unsqueeze(0)
    # image = image.to(device)
    with torch.set_grad_enabled(False):
        logits, probas = model(image)
        predict_levels = probas > 0.5
        predicted_label = torch.sum(predict_levels, dim=1)
    return probas, predicted_label.item() + 16


######
# CLASSES FOR GENDER CLASSIFIER
######


def init_gender_model(path_to_model, map_loc):

    model_gender = models.vgg16(pretrained=True)
    num_features = model_gender.classifier[6].in_features
    features = list(model_gender.classifier.children())[:-1]
    features.extend([nn.Linear(num_features, 2)])
    model_gender.classifier = nn.Sequential(*features)
    # model_gender.to(device)
    checkpoint = torch.load(
        path_to_model, map_location=torch.device("cpu")
    )  # , map_location=map_loc)
    model_gender.load_state_dict(checkpoint["model_state_dict"])
    return model_gender


def model_forward_gender(image, model):
    """ 
    inference of gender classifier
    :param image: cropped face image BGR
    :return: label classes=['Male', 'Female']
    """
    tfs = transforms_torchvision.Compose(
        [
            transforms_torchvision.ToTensor(),
            transforms_torchvision.Normalize(mean=[0.43, 0.44, 0.47], std=[0.20, 0.20, 0.20]),
        ]
    )
    image = cv2.resize(image, (150, 200))
    image = PIL.Image.fromarray(image)
    input = tfs(image)
    input = Variable(input.view(-1, 3, 200, 150))
    # input = input.to(device)
    model.eval()
    _, out = torch.max(model(input), 1)
    return int(out)


######
# CLASSES FOR FACIAL EXPRESSIONS
######


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def init_facial_expression_model(path_to_model, map_loc):
    model = VGG("VGG19")
    checkpoint = torch.load(path_to_model, map_location=map_loc)
    model.load_state_dict(checkpoint["net"])
    model.cuda()
    model.eval()
    return model


def model_forward_facial_expression(image, model):
    """ 
    inference of facial expression classifier
    :param image: cropped face image BGR
    :return: label class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    """
    cut_size = 44

    transform_test = transforms.Compose(
        [
            transforms.TenCrop(cut_size),
            transforms.Lambda(
                lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])
            ),
        ]
    )
    gray = rgb2gray(image)
    gray = resize(gray, (48, 48), mode="symmetric").astype(np.uint8)

    img = gray[:, :, np.newaxis]

    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = transform_test(img)

    ncrops, c, h, w = np.shape(inputs)

    inputs = inputs.view(-1, c, h, w)
    inputs = inputs.cuda()
    inputs = Variable(inputs, volatile=True)
    outputs = model(inputs)

    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
    score = F.softmax(outputs_avg)
    _, predicted = torch.max(outputs_avg.data, 0)
    return int(predicted)


######
# CLASSES FOR FACENET EMBEDDINGS
######


def get_aligner(shape_predictor=SHAPE_PREDICTOR):
    """
    Create AlignDlib object we are to use to crop images.
    :param shape_predictor: path to .dat file with landmarks
    :return: aligner
    """
    aligner = openface.align_dlib.AlignDlib(shape_predictor)
    return aligner


def read_image_facenet(image, bbox, aligner):
    """
    Reads image and converts it to RGB format.
    Crop by output of SSD and align by openface
    :param img_filename: full path to image
    :return: image, np.array
    """
    #     image = cv2.imread(img_filename, cv2.IMREAD_UNCHANGED)
    image = image[:, :, :3]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ymin, xmin, ymax, xmax = bbox
    dlib_bbox = dlib.rectangle(
        int(xmin * image.shape[1]),
        int(ymin * image.shape[0]),
        int(xmax * image.shape[1]),
        int(ymax * image.shape[0]),
    )
    face_align = aligner.align(
        imgDim=IMAGE_DIM, rgbImg=image, bb=dlib_bbox, landmarkIndices=INNER_EYES_AND_BOTTOM_LIP
    )
    return face_align


def get_embedder(path_to_model, cuda):
    """
    Make embedder (torch net).
    :param path_to_model: path to saved model
    :param cuda: whether to use cuda or not
    :return: TorchNeuralNet
    """
    return openface.TorchNeuralNet(path_to_model, cuda=cuda)


def embed_image(image, embedder, image_size=(96, 96)):
    """
    Calculates facenet vector of image.
    :param image: image to calculate facenet vector of
    :param embedder: model that has .forward method to calculate facenet vector
    :return: facenet vector of image, np.array
    """
    # transformation to make image digestible by embedder
    resized = cv2.resize(image, dsize=image_size, interpolation=cv2.INTER_LANCZOS4)
    facenet_vector = embedder.forward(resized)

    return facenet_vector


def detect(net, img):
    img = img - np.array([104, 117, 123])
    img = img.transpose(2, 0, 1)
    img = img.reshape((1,) + img.shape)

    img = Variable(torch.from_numpy(img).float(), volatile=True).cuda()
    BB, CC, HH, WW = img.size()
    olist = net(img)

    bboxlist = []
    # softmax loss for classification (even indices)
    for i in range(len(olist) // 2):
        olist[i * 2] = F.softmax(olist[i * 2])
    olist = [oelem.data.cpu() for oelem in olist]
    for i in range(len(olist) // 2):
        # even indices - classififcation layers, odd indices - regression layers (localization loss)
        ocls, oreg = olist[i * 2], olist[i * 2 + 1]
        FB, FC, FH, FW = ocls.size()  # feature map size
        # 4 page of article
        stride = 2 ** (i + 2)  # 4,8,16,32,64,128
        anchor = stride * 4
        # before applying nms filter out most boxes by a conf thresh of 0.05 and keep the top 400
        poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
        for Iindex, hindex, windex in poss:
            axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
            score = ocls[0, 1, hindex, windex]
            loc = oreg[0, :, hindex, windex].contiguous().view(1, 4)
            priors = torch.Tensor([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
            variances = [0.1, 0.2]
            box = decode(loc, priors, variances)
            x1, y1, x2, y2 = box[0] * 1.0
            #             bboxlist.append([x1 / img.shape[3], y1 / img.shape[2], x2 / img.shape[3], y2 / img.shape[2], score])
            bboxlist.append([x1, y1, x2, y2, score])
    bboxlist = np.array(bboxlist)
    if 0 == len(bboxlist):
        bboxlist = np.zeros((1, 5))
    return bboxlist


def run_inference_s3fd(img, net):
    if img.shape[0] != IMG_Y:
        img = cv2.resize(
            img, (int(img.shape[1] * IMG_Y / img.shape[0]), IMG_Y), interpolation=cv2.INTER_AREA
        )
    bboxlist = detect(net, img)
    if bboxlist.size > 0:
        keep = nms(bboxlist, 0.3)
        if len(keep) != 0:
            bboxlist = bboxlist[keep, :]
            output = {}
            rel_boxes = np.zeros(np.array(bboxlist[:, :-1]).shape)
            rel_boxes[:, 0] = np.array(bboxlist[:, 0]) / img.shape[1]
            rel_boxes[:, 1] = np.array(bboxlist[:, 1]) / img.shape[0]
            rel_boxes[:, 2] = np.array(bboxlist[:, 2]) / img.shape[1]
            rel_boxes[:, 3] = np.array(bboxlist[:, 3]) / img.shape[0]
            #             output['detection_boxes'] = np.array(bboxlist[:, :-1])
            output["detection_boxes"] = rel_boxes
            output["detection_scores"] = np.array(bboxlist[:, 4])
            return output
        else:
            print("Empty detector output after NMS")
            raise
    else:
        print("Empty detector output before NMS")
        raise


def get_large_bbox_face_detector_s3fd(output):
    """
    Calculate bbox with max area by output of S3FD face detector
    :param output: output of S3FD
    :return: bbox: bbox with max area
    """
    bbox_large = output["detection_boxes"][0]
    i_max = 0
    if len(output) != 0:
        indices = np.where(output["detection_scores"] > FACE_PROB_THRESH)[0]
        if len(indices) > 1:
            s = []
            for box in output["detection_boxes"][indices]:
                xmin, ymin, xmax, ymax = box
                s.append((ymax - ymin) * (xmax - xmin))
            i_max = np.argmax(np.array(s))
            bbox_large = output["detection_boxes"][i_max]
    return bbox_large, i_max


def output_processing_thresh(output):
    """
    Compare relative square of top detected face with threshold
    :param output: output of S3FD
    :return: bbox: bbox with max area and confidence > threshold,
    index of bbox
    """
    conf = output["detection_scores"][0]
    xmin, ymin, xmax, ymax = output["detection_boxes"][0]
    square = (ymax - ymin) * (xmax - xmin)
    if square < FACE_SQUARE_THRESH:
        bbox_large, i_max = get_large_bbox_face_detector_s3fd(output)
        xmin_l, ymin_l, xmax_l, ymax_l = bbox_large
        square = (ymax_l - ymin_l) * (xmax_l - xmin_l)
        conf = output["detection_scores"][i_max]
    return conf, square
