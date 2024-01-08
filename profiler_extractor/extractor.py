# usage: python ./extractor.py <model_name> <core size> <batch size>
# the batch size should be 1 for extracting input data
import os
import sys
sys.path.append('.')
import time
import torch
import json

from PIL import Image

import transformers
from transformers import BertConfig, BertForQuestionAnswering, AutoTokenizer, BertTokenizer
from transformers import ViTConfig, ViTForImageClassification

from torchvision import models, transforms

from ultralytics import YOLO

import pytorchtool

from classes import class_names
from COCO_classes import class_names_coco

def process_img(path_img, size=None):
    # hard code
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    if size is None:
        inference_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])
    else:
        inference_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

    # tensor
    img_tensor = inference_transform(Image.open(path_img).convert('RGB'))
    img_tensor.unsqueeze_(0)        # chw --> bchw
    
    return img_tensor

def process_img_detection(img_path, cropsize=(300, 300)):
    inference_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor(),
    ])

    # tensor
    img_tensor = inference_transform(Image.open(img_path).convert('RGB'))
    img_tensor.unsqueeze_(0)        # chw --> bchw
    
    return img_tensor

class Model:
    # @profile (Uncomment to use memory_profiler tool for line-by-line memory analysis e.g. python -m memory_profiler ./extractor.py resnet50 4 1)
    def __init__(self, model_name, use_gpu=False):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.depth = -1

        if self.model_name in 'inception':
            self.model_name = 'inception'
            self.path = "../model_repository/inception_v3_google-0cc3c7bd.pth"

            model = models.Inception3(aux_logits=None, transform_input=None, 
                                    init_weights=None)
            model.eval()
            self.model = model
            self.depth = 2
        
        elif self.model_name == 'ssd300':
            self.path = '../model_repository/ssd300_vgg16_coco-b556d3b4.pth'
            self.backbonePath = "../model_repository/ssd300_vgg16_features-amdegroot-88682ab5.pth"
            model = models.detection.ssd300_vgg16(None, weights_backbone=None)
            model.eval()
            self.model = model
        
        elif self.model_name == 'vgg16':
            self.path = '../model_repository/vgg16-397923af.pth'
            model = models.vgg16(None)
            model.eval()
            self.model = model
        
        elif self.model_name == 'resnet50':
            self.path = '../model_repository/resnet50-0676ba61.pth'
            model = models.resnet50(None)
            model.eval()
            self.model = model
        
        elif self.model_name == 'efficientnet_b5':
            self.path = '../model_repository/efficientnet_b5_lukemelas-b6417697.pth'
            model = models.efficientnet_b5(None)
            model.eval()
            self.model = model
            self.depth = 4

        elif self.model_name == "yolov8":
            self.path = "../model_repository/yolov8x.pt"
            if use_gpu:
                model = YOLO("yolov8x.yaml")
                model.to(torch.device('cuda:0'))
            else:
                model = YOLO("yolov8x.yaml")
                model.to(torch.device('cpu'))
            
            model = model.model
            model.eval()
            self.model = model
            self.depth = 4

        elif self.model_name == "ViT_base":
            self.path = "../model_repository/ViT_base/pytorch_model.bin"
            with open("../model_repository/ViT_base/ViT_base_config.json") as f:
                config_json = json.load(f)
            
            config = ViTConfig()
            config.id2label = config_json["id2label"]
            config.label2id = config_json["label2id"]
            
            if use_gpu:
                model = ViTForImageClassification(config).to(torch.device('cuda:0'))
            else:
                model = ViTForImageClassification(config).to(torch.device('cpu'))
            model.eval()
            self.model = model

        elif self.model_name == "bert":
            self.path = "../model_repository/bert/model.pytorch"
            with open("../model_repository/bert/bert_config.json") as f:
                config_json = json.load(f)

            config = BertConfig(
                attention_probs_dropout_prob=config_json["attention_probs_dropout_prob"],
                hidden_act=config_json["hidden_act"],
                hidden_dropout_prob=config_json["hidden_dropout_prob"],
                hidden_size=config_json["hidden_size"],
                initializer_range=config_json["initializer_range"],
                intermediate_size=config_json["intermediate_size"],
                max_position_embeddings=config_json["max_position_embeddings"],
                num_attention_heads=config_json["num_attention_heads"],
                num_hidden_layers=config_json["num_hidden_layers"],
                type_vocab_size=config_json["type_vocab_size"],
                vocab_size=config_json["vocab_size"])
            
            if use_gpu:
                model = BertForQuestionAnswering(config).to(torch.device('cuda:0'))
            else:
                model = BertForQuestionAnswering(config).to(torch.device('cpu'))
            model.eval()
            self.model = model
            self.depth = 5
        else:
            raise Exception("Wrong model name")

        if self.use_gpu:
            self.model = self.model.cuda()

    # @profile
    def load_weight(self):
        state_dict_read = torch.load(self.path)
        self.model.load_state_dict(state_dict_read, strict=None)
        
        if self.model_name=="ssd300":
            state_dict_read = torch.load(self.backbonePath)
            self.model.backbone.load_state_dict(state_dict_read, None)

    def print_weight(self):
        state_dict_read = self.model.state_dict()
        for k, v in state_dict_read.items():
            print(k, v)

    def get_model(self):
        return self.model
    
    def get_input(self):
        return self.x
    
    def expand_input(self, input, batchsize):
        return input.expand((batchsize, ) + input.shape[1:]).clone()

    def set_input(self, inputPath='./blackswan.jpg', batchSize=1):
        print("batchsize:", batchSize)
        if self.model_name == "ssd300":
            # self.x = torch.rand(2, 3, 300, 300)
            self.x = process_img_detection(inputPath, (300, 300))
        elif self.model_name == "yolov8":
            self.x = process_img(inputPath, 640)
        elif self.model_name == "bert":
            self.tokenizer = BertTokenizer("./model_weight/vocab.txt")
            # question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
            question = "Where do most teachers get their credentials from?"
            text = "The role of teacher is often formal and ongoing, carried out at a school or other place of formal education. In many countries, a person who wishes to become a teacher must first obtain specified professional qualifications or credentials from a university or college. These professional qualifications may include the study of pedagogy, the science of teaching. Teachers, like other professionals, may have to continue their education after they qualify, a process known as continuing professional development. Teachers may use a lesson plan to facilitate student learning, providing a course of study which is called the curriculum."
            # Batch
            # inputs = self.tokenizer([question, question], [text, text], padding=True, truncation=True, return_tensors="pt")
            inputs = self.tokenizer(question, text, return_tensors="pt")
            self.x = inputs
        elif self.model_name == "efficientnet_b5":
            self.x = process_img(inputPath, size=456)
        else:
            self.x = process_img(inputPath)
        
        # Expand data based on batchsize
        if (isinstance(self.x, torch.Tensor)):
            self.x = self.x.expand((batchSize, ) + self.x.shape[1:]).clone()
        elif (isinstance(self.x, transformers.tokenization_utils_base.BatchEncoding)):
            for key in self.x.keys():
                self.x[key] = self.expand_input(self.x[key], batchSize)
        else:
            raise Exception("Unsupported input type: ", type(self.x))
        
        if self.use_gpu:
            self.x = self.x.cuda()

    # @profile
    def inference(self, outputVerbose = False):
        if (self.model_name == "bert"):
            with torch.no_grad():
                outputs = self.model(**self.x)
        else:
            with torch.no_grad():
                outputs = self.model(self.x)
        # print(outputs)
        if (outputVerbose == True):
            if self.model_name == "ssd300":
                if self.use_gpu:
                    labels = outputs[0]['labels'].cpu().numpy()     # label
                    scores = outputs[0]['scores'].cpu().numpy()     # score
                    bboxes = outputs[0]['boxes'].cpu().numpy()      # bbox
                else:
                    labels = outputs[0]['labels'].numpy()     # label
                    scores = outputs[0]['scores'].numpy()     # score
                    bboxes = outputs[0]['boxes'].numpy()      # bbox
                print("result:", class_names_coco[labels[0]-1])
            elif self.model_name == "yolov8":
                # print(outputs)
                pass
            elif self.model_name == "bert":
                answer_start_index = outputs.start_logits.argmax(-1)
                answer_end_index = outputs.end_logits.argmax(-1)
                for id in range (outputs.start_logits.shape[0]):
                    predict_answer_tokens = self.x.input_ids[id, answer_start_index[id] : answer_end_index[id] + 1]
                    print(self.tokenizer.decode(predict_answer_tokens, skip_special_tokens=True))
            
            elif self.model_name == "ViT_base":
                answer = outputs.logits.argmax(-1)
                for id in range (outputs.logits.shape[0]):
                    print(self.model.config.id2label[str(answer[id].item())])
            else:
                print("result: " + class_names[torch.argmax(outputs, 1)[0]])

    def extract(self, depth=-1):
        if (self.model_name == "bert"):
            with pytorchtool.Profile(self.model, self.model_name, use_cuda=self.use_gpu, 
                    depth=depth) as prof:
                with torch.no_grad():
                    outputs = self.model(**self.x)
        else:
            with pytorchtool.Profile(self.model, self.model_name, use_cuda=self.use_gpu, 
                    depth=depth) as prof:
                with torch.no_grad():
                    outputs = self.model(self.x)

        prof.saveInput("./model_weight/" + self.model_name + "_layers/" + "input.pickle")


def inferenceTest(m:Model):
    REPEATTIME = 10
    start_total = time.time()
    for _ in range(REPEATTIME):
        m.inference(outputVerbose=True)
    
    print("Average inference time: ", (time.time() - start_total) / REPEATTIME)

if __name__ == "__main__":
    torch.randn(4)
    name = sys.argv[1]
    coreNumber = int(sys.argv[2])
    batchSize = int(sys.argv[3])

    # Set the number of threads for torch
    torch.set_num_interop_threads(coreNumber)
    torch.set_num_threads(coreNumber)
    print(torch.get_num_interop_threads())
    print(torch.get_num_threads())

    # init model
    m = Model(name, use_gpu=False)
    m.load_weight()
    m.set_input(batchSize=batchSize)

    # extract weight of all layers 
    pytorchtool.save_model(m.get_model(), m.model_name, depth=m.depth)
    # extract inputdata of all layers
    m.extract(depth=m.depth)

    # inference test
    inferenceTest(m)