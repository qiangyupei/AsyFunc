""" Measure the layer type, layer configuration, disk usage, execute time, input data size, and output data size of model layers 
under different batch size and CPU core size """
# Usage: python ./profiler.py ./model_weight/<model_name>_layers/ <core size> <batch size>

import sys
sys.path.append(".")
import os
import time
import torch
import pickle
import pandas as pd
from collections import defaultdict

from torchvision.models.detection.image_list import ImageList

REPEATTIME = 500

def saveFile(layerMetrics:defaultdict(list), filePath:str):
    df = pd.DataFrame.from_dict(layerMetrics, orient='index', 
        columns=['Layer Type', 'Layer Configuration', 'Disk Size(MB)', 'Batch Size', 'Core Number', 'Execute Time(ms)', 'Input Size(MB)', 'Output Size(MB)'])
    df.to_csv(filePath)

def createBatchInput(input: torch.Tensor, batch_size):
    '''Construct input with specified batch size based on the original input'''
    if (isinstance(input, torch.Tensor)):
        return input.expand((batch_size, ) + input.shape[1:]).clone()
    elif (isinstance(input, list)):    # YOLOv8
        for i in range(len(input)):
            input[i] = createBatchInput(input[i], batch_size)
        return input
    # elif (isinstance(input, ImageList)):
    #     input.tensors = input.tensors.expand((batch_size, ) + input.tensors.shape[1:]).clone()
    #     for i in range(len(input.image_sizes), batch_size):
    #         input.image_sizes.append(input.image_sizes[0])
    #     return input
    else:
        raise Exception("Not support type for create batch input: ", type(input))

def getDataSize(data):
    '''Calculate the size of the input data'''
    binary_data = pickle.dumps(data)
    return sys.getsizeof(binary_data) / 1024. / 1024.

if __name__ == "__main__":
    topDir = sys.argv[1]
    coreSize = int(sys.argv[2])
    batchSize = int(sys.argv[3])
    print(time.strftime("%Y-%m-%d %H:%M:%S") + f"————{topDir} core:{coreSize} batch:{batchSize}")

    # x = torch.randn(4)

    torch.set_num_interop_threads(coreSize)
    torch.set_num_threads(coreSize)
    
    # Read input data for all layers
    with open(os.path.join(topDir, "input.pickle"), "rb") as f:
        inputDict = pickle.load(f)

    layerMetrics = defaultdict(list)
    moduleList = list()

    # measure metrics for each layer
    for layerName, inputData in inputDict.items():
        filePath = os.path.join(topDir, layerName + ".pkl")

        # disk usage
        fileSize = os.path.getsize(filePath) / 1024. / 1024.

        # load layer
        module = torch.load(filePath)
        moduleList.append(module)

        layerMetrics[layerName].append(module.__class__.__name__)
        layerMetrics[layerName].append(str(module))
        layerMetrics[layerName].append(fileSize)

        # read input data
        inputData = inputDict[layerName]
        if (isinstance(inputData, tuple)):
            if (len(inputData) == 1 or (len(inputData) == 2 and inputData[1] is None)):
                inputData = inputData[0]
            else:
                print("skip layer——", layerName)
                continue

        print("running layer: ", layerName)
        # construct input data with specified batch size
        inputData = createBatchInput(inputData, batchSize)

        layerMetrics[layerName].append(batchSize)
        layerMetrics[layerName].append(coreSize)

        module.eval()

        # inference
        startTime = time.time()

        for i in range(REPEATTIME):
            with torch.no_grad():
                output = module(inputData)
        
        exec_time = (time.time() - startTime) * 1000 / REPEATTIME
        layerMetrics[layerName].append(exec_time)

        layerMetrics[layerName].append(getDataSize(inputData))
        layerMetrics[layerName].append(getDataSize(output))
        time.sleep(0.1)

    dirPath = os.path.join("./layerMetrics/", topDir.strip('/').split('/')[-1].rsplit('_',1)[0])
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    saveFile(layerMetrics, os.path.join(dirPath, topDir.strip('/').split('/')[-1].rsplit('_',1)[0]+ "_bs"+str(batchSize) + "_cn"+str(coreSize) + ".csv"))