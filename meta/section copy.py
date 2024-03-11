import math
import typing
import torch
import torch.nn as nn
import copy
from meta.next_steps import NextSteps

# from meta.instruction_classifier.lstm_steps_classifier import NextStepClassifier

import inspect
class SectionLayers(nn.Module):
    def __init__(self,name,num_options_each_layer,layers:typing.List[nn.Module],is_encoder) -> None:
        super().__init__()
        self.base_layers=[]
        self.num_shared_layers=[]
        self.shared_index=[]
        self.is_setup=False
        self.num_options_each_layer=num_options_each_layer
        self.name=name
        
        self.last_sections_dummy_features=[]
        self.last_sections_steps=[]
        self.layers=nn.ModuleList()
        self.extend_layers(layers)
        self.tasks=dict()

        self.is_encoder=is_encoder
        
        self.features=None
    def save_network(self,path):
        state=vars(self).copy()
        state.pop("tasks")
        state.pop("is_encoder")
        #state.pop("detach")
        torch.save(state,path)
    def load_network(self,path):
        state=torch.load(path)
        self.__dict__.update(state)
        
    def append_layer(self,layer):
        self.base_layers.append(layer)
        self.num_shared_layers.append(0)
        self.shared_index.append(None)
    def append_shared_layers(self,index,num_shared_layers):
        self.base_layers.append(self.base_layers[index])
        self.num_shared_layers.append(num_shared_layers)
        self.shared_index.append(index)
        
    def extend_layers(self,layers):
        for layer in layers:
            self.append_layer(layer)
    def extend_shared_layers(self,indices,num_shared_layers):
        for i in indices:
            self.append_shared_layers(i,num_shared_layers)
    
    
    def setup(self):

        if self.is_setup:
            return 

     
        self.is_layer_with_params=[]
        self.num_layers_with_params=0
        self.num_total_layers=0
        self.last_feature_index=None
        for i,layer in enumerate(self.base_layers):
           
            shared_index=self.shared_index[i]
            num_shared_layers=self.num_shared_layers[i]
            new_layer=nn.ModuleList()
            if len(list(layer.parameters()))>0:
                self.num_layers_with_params+=1
                self.is_layer_with_params.append(True)
                for j in range(num_shared_layers):
                    new_layer.append(self.layers[shared_index][j])
                for j  in range(self.num_options_each_layer-num_shared_layers):
                
                   
                    new_layer.append(copy.deepcopy(layer))
                
            else:
                new_layer.append(copy.deepcopy(layer))  
                self.is_layer_with_params.append(False) 
            self.layers.append(new_layer)
           
           
            if self.is_encoder(layer):
                self.last_feature_index=self.num_total_layers
           
            self.num_total_layers+=1    
      
        self.is_setup=True
    def dummy_forward(self,data):

        last_features=None
        for i,layer in enumerate(self.base_layers):
           
            data=layer(data)
            if self.is_encoder(layer):
                with torch.no_grad():
                    last_features=data

            
        return data
        
    def forward_with_update(self,next_steps:NextSteps,update, *args,**kwargs):

        assert self.is_setup
        #print(next_steps.indices)
     
        self.next_steps=next_steps
        indices=next_steps.indices
        batch=next_steps.tensor.size(0)
        assert batch==1
        indices=indices[0]
        index_with_params=0
        last_features=None
        for i in range(self.num_total_layers):
            
            if self.is_layer_with_params[i]:
                index=indices[index_with_params]
                layer=self.layers[i][index]
                output=layer(*args,**kwargs)
                index_with_params+=1
            else:
                output=self.layers[i](*args,**kwargs)
               
          
            

            if i==self.last_feature_index:
                # with torch.no_grad():
                last_features=output
                
            args,kwargs=update(output,args,kwargs)
            
        return args,last_features
    def forward(self,next_steps,*args,**kwargs):
        return self.forward_with_update(next_steps,lambda __output,__args,__kwargs:__output,*args,**kwargs)

class CandidateSteps:
    def __init__(self,num_candidate,num_tracking,num_next_steps,num_step_options) -> None:
        self.indecies=...
        self.softmax=... 
        self.losses=... 
    def index_most_simular(self,next_step)->int:
        pass
    def merge(self,other):
        pass
    def clear(self):
        pass
    def update(self,index,loss):
        pass
    def get(self,index):
        pass
    
    
    
    
class Section:
    def __init__(self,layers:SectionLayers,encoder,classifier,adjust_encode,detach) -> None:
        self.layers=layers
        self.encoder=encoder
        self.classifier=classifier
        self.layers_optimizer=None
        self.steps_classifier_optimizer=None
        self.adjust_encode=adjust_encode
        #self. adjust_ebcoder_output=adjust_ebcoder_output
        self.detach=detach
        self.canidate_steps=...
        self.new_candidate_steps=...
        
    def forward(self,previous_steps,task,*args, **kwargs):
        if self.encoder is None:

            detached_args, detached_kwargs = self.detach(args, kwargs)
            encode=self.adjust_encode(*detached_args,**detached_kwargs)
        else:
            detached_args,detached_kwargs=self.detach(args, kwargs)
            encode= self.adjust_encode( self.encoder(*detached_args,**detached_kwargs))
        batch = encode.size(0)
        if batch != 1:
            encode=encode[0:1]
        next_steps=self.classifier(encode,previous_steps,task)


        next_steps=NextSteps(next_steps)
        output,last_features= self.layers.forward(next_steps,*args,**kwargs)
        return output,last_features,next_steps
    # def dummy_forward(self,features):
    #     output,last_features=self.layers.dummy_forward(features)
    #     return output,last_features,None
    # def __call__(self, *args: typing.Any, **kwds: typing.Any) -> typing.Any:
    #     return self.forward(*args, **kwds)
    
    def optimize_layers(self,loss):
        self.layers_optimizer.zero_grad()
        loss.backward()
        self.layers_optimizer.step()
        
    def optimize_steps_classifier(self,loss,previous_steps):
        self.steps_classifier_optimizer.zero_grad()
        classifier_loss=self.get_steps_classifier_loss(loss.detach(),previous_steps)
        classifier_loss.backward()
        self.steps_classifier_optimizer.step()
    def get_steps_classifier_loss(self,loss,next_steps:NextSteps):
        confidence=torch.sum(next_steps.probability)
        return confidence*(loss.detach())
    def save_network(self,saved:set,path:str,formate_args:dict,module_name_arg="model_name"):
        if self.layers not in saved:
            self.layers.save_network(path.format_map(formate_args|{module_name_arg:self.layers.name}))
            saved.add(self.layers)
        if self.encoder not in saved:
            if self.encoder is not None:
                torch.save(self.encoder.state_dict(),path.format_map(formate_args|{module_name_arg:f"{self.layers.name}_encoder"}))
                saved.add(self.encoder)
        if self.classifier not in saved:
            if self.classifier is not  None:
                torch.save(self.classifier.state_dict(),path.format_map(formate_args|{module_name_arg:f"{self.layers.name}_classifier"}))
                saved.add(self.classifier)
        # torch.save(self.encoder,path.format_map(formate_args|{module_name_arg:"encoder"}))
        # torch.save(self.classifier,path.format_map(formate_args|{module_name_arg:"classifier"}))
    def load_network(self,loaded:set,path:str,formate_args:dict,module_name_arg="model_name"):
        if self.layers not in loaded:
            self.layers.load_network(path.format_map(formate_args|{module_name_arg:self.layers.name}))
           
            loaded.add(self.layers)
        if self.encoder not in loaded:
            if self.encoder is not  None:
                encoder_state=torch.load(path.format_map(formate_args|{module_name_arg:f"{self.layers.name}_encoder"}))
                #print(path.format_map(formate_args|{module_name_arg:"encoder"}))
                self.encoder.load_state_dict(encoder_state)
                loaded.add(self.encoder)
        if self.classifier not in loaded:
            if self.classifier is not None:
                classifier_state=torch.load(path.format_map(formate_args|{module_name_arg:f"{self.layers.name}_classifier"}))
                self.classifier.load_state_dict(classifier_state)
                loaded.add(self.classifier)
       
    def setup(self):
        self.layers.setup()
        #self.classifier=torch.load(path.format_map(formate_args|{module_name_arg:"classifier"}))
        
        
        
    
    
    