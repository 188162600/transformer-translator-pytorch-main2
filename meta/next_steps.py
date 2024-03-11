
import math
import torch
import torch.nn as nn


class NextSteps:
    def __init__(self, tensor: torch.Tensor):
        #print("tensor1",tensor.shape)
        tensor=torch.sum(tensor, dim=0).unsqueeze(0)
        self.tensor = tensor
        #print("tensor2",tensor.shape)
      
        self.indices = torch.argmax(tensor,dim=1)
        # print(self.indices)
        # print("self.indices.shape,self.tensor.shape",self.indices.shape,self.tensor.shape)
        expanded_indices = self.indices.unsqueeze(-1)

        self.softmax = torch.softmax(self.tensor, dim=1)
       
        self.probability = torch.gather(self.softmax, 1,expanded_indices)
        self.confidence=torch.sum(self.probability)
class LinearStepClassifier(nn.Module):
    def __init__(self, num_next_steps, num_step_classes,encoder_output_shape,batch_index=0,device=None):
        super(LinearStepClassifier, self).__init__()
        
       
        # print("encode2",self.encoder)
        # print(self.encoder(dummy_input))
        # print("encoder",self.encoder)
        # print("dummy_input",dummy_input.shape)
        self.in_features_size=math.prod(encoder_output_shape)
        self.batch_index=batch_index
        
        self.net=dict()
        #nn.Linear(self.in_features_size, num_step_classes * num_next_steps)
        self.num_next_steps = num_next_steps
        self.num_step_classes = num_step_classes
        #self.detach=detach
        self.device=device
        #print("init num step classes",num_step_classes)


    def forward(self, features, previous:NextSteps, task):
        if task not in self.net:
            net=nn.Linear(self.in_features_size, self.num_step_classes * self.num_next_steps,device=self.device)
            self.net[task]=net
            self.add_module(str(len(self.net)),net)
        else:
            net=self.net[task]
        #print(features.shape)
        #encode=self.detach(features)
        encode=features
        if self.batch_index!=0:
            #print("trans")
            encode=encode.transpose(0,self.batch_index)
        #print(encode.shape)
        batch=encode.size(0)
        encode=encode.view(batch,-1)
        #print(encode.shape,batch)
        result=net(features).view(batch,self.num_step_classes,self.num_next_steps)
        return result
        #return NextSteps(result)

class LSTMNextStepClassifier(nn.Module):
    def __init__(self, num_next_steps, num_step_classes,encoder_output_shape,batch_index=0,device=None):
        super().__init__()
        


        # print("encode2",self.encoder)
        # print(self.encoder(dummy_input))
        # print("encoder",self.encoder)
        # print("dummy_input",dummy_input.shape)
        self.in_features_size=math.prod(encoder_output_shape)

        self.nets = nn.ModuleList(
            [nn.LSTMCell(input_size= self.in_features_size, hidden_size=num_step_classes * num_next_steps,device=device) for _ in
             range(num_next_steps)])
        self.num_next_steps = num_next_steps
        self.num_step_classes = num_step_classes
        #print("init num step classes",num_step_classes)


    def forward(self, features, previous:NextSteps, task):
        # print("classifier vars",vars(self))
        # print("")
        # print("forward", self.nets, features)
        #print("Next Steps Features",features.shape)
        batch = features.size(0)
        #print("encoder",self.encoder)
        #print("features device",features.device)
        #features=self.encoder(features)
        #print("Next Steps Features encode",features.shape)
        features=features.view(batch,-1)
        features_size=features[0].numel()
        
        if features_size!=self.in_features_size:
            features=nn.functional.pad(features,(0,0,0,self.in_features_size-features_size))
        
        #print("feature shape after",features.shape)
        hidden_long_term = task.hidden_long_term[self] if task.hidden_long_term.get(self) is not None else torch.zeros(
            batch, self.num_step_classes * self.num_next_steps, device=features.device)

        if previous is None:
            hidden_short_term = torch.zeros(batch, self.num_step_classes , self.num_next_steps,
                                            device=features.device)
        else:
            hidden_short_term = previous.tensor.detach()
     
        hx = hidden_short_term
      
        if hx.size(1)!=self.num_step_classes or hx.size(2)!= self.num_next_steps:
            #print("self.num_step_classes",self.num_step_classes)
            #hx=hx.view(batch,self.num_step_classes,-1)
            #print(hx.shape,(batch,self.num_step_classes,))
            #print("interpolating",hx.shape)
            #print("hx.shape,(self.num_step_classes,self.num_next_steps ,)",hx.shape,(self.num_step_classes,self.num_next_steps ,))
            hx=hx.unsqueeze(0)
            #print("hx.shape,(self.num_step_classes,self.num_next_steps ,)",hx.shape,(self.num_step_classes,self.num_next_steps ,))
            hx=torch.nn.functional.interpolate(hx,(self.num_step_classes,self.num_next_steps ,))
            #print("interpolated",hx.shape)
        hx=hx.view(batch,-1)
        #
        # previous_steps=hx.size(1)
        # hx=hx[:,:,:]
        cx = hidden_long_term
        #print("hx cx",hx,cx,features)
        #print("hidden", batch, self.num_step_classes, self.num_next_steps, self.num_previous_steps)
        for net in self.nets:
            #print("forward2", net, features.shape, hx.shape, cx.shape)
            #print("forward2",features.shape,hx.shape,cx.shape)
            hx, cx = net(features, (hx, cx))
            #print("forward2 result", net, features.shape, hx.shape, cx.shape)
        hx=hx.view(batch, self.num_step_classes, self.num_next_steps)
        task.hidden_long_term[self] = cx.detach()
        return hx
        #return NextSteps(hx)
