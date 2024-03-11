import torch
import torch.nn as nn



import torch
import torch.nn as nn
import math
from meta.instruction_classifier.next_steps import NextSteps
class LinearStepClassifier(nn.Module):
    def __init__(self, num_next_steps, num_step_classes,encoder_output_shape,batch_index=0):
        super(LinearStepClassifier, self).__init__()
        
       
        # print("encode2",self.encoder)
        # print(self.encoder(dummy_input))
        # print("encoder",self.encoder)
        # print("dummy_input",dummy_input.shape)
        self.in_features_size=math.prod(encoder_output_shape)
        self.batch_index=batch_index
        
        self.net=nn.ModuleDict()
        #nn.Linear(self.in_features_size, num_step_classes * num_next_steps)
        self.num_next_steps = num_next_steps
        self.num_step_classes = num_step_classes
        #print("init num step classes",num_step_classes)


    def forward(self, features, previous:NextSteps, task):
        if task not in self.net:
            net=self.net[task]=nn.Linear(self.in_features_size, self.num_step_classes * self.num_next_steps)
        else:
            net=self.net[task]
           
        if self.batch_index!=0:
            encode=encode.transpose(0,self.batch_index)
        batch=encode.size(0)
        encode=encode.view(batch,-1)
        result=net(features).view(batch,self.num_step_classes,self.num_next_steps)

        return NextSteps(result)
