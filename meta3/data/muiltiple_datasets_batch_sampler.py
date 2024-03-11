import random
class MuiltipleDatasetsBatchSampler:
    def enable(self,index,enable):
        if self.enables[index]!=enable:
            self.enables[index]=enable
            self.batches = self._generate_batches()
            
            self.total_items = sum(len(batch) for batch in self.batches)
        #     print("enables",self.enables)
        # print("enables2",self.enables)
        # print("len",self.total_items)
    def __init__(self, datasets, batch_size, shuffle):
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.enables=[False]*len(datasets)
        # Initialize starting indices here if needed, or pass dynamically when calling _generate_batches

        self.batches = self._generate_batches()
        self.total_items = sum(len(batch) for batch in self.batches)

    def _generate_batches(self):
        print("generate batches")
        batches = []
        start_index = 0
        for i,dataset in enumerate( self.datasets):
            #print("generate batches",i,self.enables[i])
            if self.enables[i]:
                # Adjust the range to start from the given starting index, ensuring it does not exceed dataset length
                #start_index = max(0, min(start_index, len(dataset)))
                dataset_indices = list(range(start_index,start_index+ len(dataset)))
                if len(dataset_indices) % self.batch_size != 0:
                    # Adjust dataset indices to ensure the last batch meets the batch size requirement
                    dataset_indices = dataset_indices[:-(len(dataset_indices) % self.batch_size)]
                #print(dataset_indices)
                dataset_batches = [dataset_indices[i:i + self.batch_size] for i in
                                range(0, len(dataset_indices), self.batch_size)]
                batches.extend(dataset_batches)
                start_index+=len(dataset)
            else:
                start_index+=len(dataset)
            #print(start_index)
        return batches

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)  # Shuffle the batches
        yield from self.batches

    def __len__(self):
        # Return the total number of items across all batches
        return self.total_items
