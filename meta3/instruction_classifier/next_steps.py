import torch
import torch.nn as nn



import torch
import torch.nn as nn

# class NextSteps:
#     def __init__(self, tensor: torch.Tensor, num_step_classes, num_next_steps):
#         self.tensor = tensor
#         batch = tensor.size(0)
        
#         # Reshape tensor correctly assuming 'tensor' is [batch, num_step_classes*num_next_steps]
#         # Assuming tensor should be reshaped to [num_step_classes, batch, num_next_steps]
#         self.reshaped_tensor = tensor.permute(2, 0, 1)
#         #print(batch,num_step_classes,num_next_steps,self.reshaped_tensor.shape)
#         self.indices = torch.argmax(self.reshaped_tensor, dim=-1)
#         #print("self.indices.shape",self.indices.shape)
#         self.softmax = torch.softmax(self.reshaped_tensor, dim=-1)
#         #print("self.softmax.shape", self.softmax.shape)
#         # Correcting the expanded indices to match dimensions for gathering
#         expanded_indices = self.indices.unsqueeze(-1)
#             #.expand(-1, -1, num_next_steps)
        
#         # Gathering probabilities
#         self.probability = torch.gather(self.softmax, 2, expanded_indices)
#         #print("self.probability.shape", self.probability.shape)
#         #print(self.indices.shape, self.softmax.shape, self.probability.shape)
        
        
class NextSteps:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        batch = tensor.size(0)
        #print(num_picking_classes,"num_picking_classes")
        # Reshape tensor correctly assuming 'tensor' is [batch, num_step_classes*num_next_steps]
        # Assuming tensor should be reshaped to [num_step_classes, batch, num_next_steps]
        #self.reshaped_tensor = tensor.permute(2, 0, 1)
        #print("self.reshaped_tensor.shape",self.reshaped_tensor.shape)
        #print(batch,num_step_classes,num_next_steps,self.reshaped_tensor.shape)
        #hx=hx.view(batch, self.num_step_classes, self.num_next_steps)
        self.indices = torch.argmax(tensor,dim=1,descending=True)
        expanded_indices = self.indices.unsqueeze(-1)
        #print("self.indices.shape",self.indices.shape)
        self.softmax = torch.softmax(self.tensor, dim=1)
       
        self.probability = torch.gather(self.softmax, 1,expanded_indices)
        self.confidence=torch.sum(self.probability)
        #self.hx=hx
        #print(self.indices)
        #print("num_picking_classes",num_picking_classes,self.probability.shape,self.indices.shape,self.softmax.shape)
        #print("self.probability.shape", self.probability.shape)
        #print(self.indices.shape, self.softmax.shape, self.probability.shape)

