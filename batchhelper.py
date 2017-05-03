import numpy as np

class BatchGenerator:
    
    def __init__(self, batch_size, filenames, input_length, num_splits):
        """Creates a batch generator with a given batch size from files.
        
        Args:
            batch_size: int64 batch size [what is the limit]
            filenames: list of string filenames (must be absolute or CORRECT relative to this directory).
            input_length: string of the input length
        """
        self.batch_size = batch_size
        self.filenames = filenames
        self.current_file_pointer = iter(self.filenames)
        self.input_length = input_length
        self.num_splits = num_splits
        self.load_file(self.current_file_pointer.next())
    
        self.mapper = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1],
          'N':[0,0,0,0],'R':[0,1,0,1],'Y':[1,0,1,0],'M':[1,1,0,0],
          'K':[0,0,1,1],'S':[0,1,1,0],'W':[1,0,0,1],'B':[0,1,1,1],
          'V':[1,1,1,0],'H':[1,1,0,1],'D':[1,0,1,1]}
        for key, val in self.mapper.items():
            self.mapper[key] = np.array(val)
        #self.unused_indices = list(xrange(0, self.filtered_input.shape[0]))
        
    def load_file(self, current_file):
        inp = np.loadtxt(current_file, delimiter=',', skiprows=1, dtype=str)
        lens = np.expand_dims(map(lambda x: int(len(x)), inp[:,1]), axis=1)
        in2 = np.concatenate([inp[:], lens], axis=1)
        filtered = in2[in2[:,3] == str(self.input_length)] ## weirdly needs to be a string
        self.filtered_input = filtered
        self.unused_indices = list(xrange(0, self.filtered_input.shape[0]))
        
    def next_batch(self):
        """Get a one-hot encoded batch of batch_size.
        
        Returns:
            one_hot_x : encoded one hot X input size (batch_size x [X dims])
            batch_y : labels batch  size (batch_size, num_classes)
        """
        if len(self.unused_indices) >= self.batch_size:
            rand = np.random.choice(self.unused_indices, self.batch_size, replace=False)
            self.unused_indices = list(set(self.unused_indices) - set(rand))
            batch = self.filtered_input[rand,1]
            labels = self.filtered_input[rand,2]

        else:
            # if there aren't enough examples left in the batch, take the unused examples plus some from the next file
            batch_top = self.filtered_input[self.unused_indices, 1]
            labels_top = self.filtered_input[self.unused_indices, 2]
            
            try:
                next_file = self.current_file_pointer.next() 
                self.load_file(next_file)
                
                rand = np.random.choice(self.unused_indices, self.batch_size - len(batch_top), replace=False)
                self.unused_indices = list(set(self.unused_indices) - set(rand))
                
                batch_bottom = self.filtered_input[rand,1]
                labels_bottom = self.filtered_input[rand,2]
                
                batch = np.concatenate([batch_top, batch_bottom], axis=0)
                labels = np.concatenate([labels_top, labels_bottom], axis=0)
                
            except StopIteration:
                raise
            
        split_length = self.input_length/self.num_splits

        assert self.num_splits * split_length == len(batch[0])

        batch_y = np.concatenate(np.tile(map(lambda x: np.expand_dims(np.array([abs(float(x)-1.0/self.num_splits), abs(float(abs(1-int(x)))- 1.0/self.num_splits)]), axis=0), labels), (self.num_splits, 1)), axis=0)
        #batch_y = np.concatenate(np.tile(map(lambda x: np.expand_dims(np.array([float(x), float(abs(1-int(x)))]), axis=0), labels), (self.num_splits, 1)), axis=0)

        getseq = lambda seq: np.expand_dims(np.concatenate([np.expand_dims(self.mapper[i], axis=0) for i in seq], axis=0), axis=0)

        sliced_batch = [[seq[0+i:split_length+i] for i in range(0, len(seq), split_length)] for seq in batch]

        one_hot_x = np.concatenate([np.expand_dims(np.concatenate([getseq(x) for x in section], axis=0), axis=2) for section in sliced_batch], axis=0)

        rand2 = np.random.choice(range(batch_y.shape[0]), batch_y.shape[0], replace=False)

        return one_hot_x[rand2], batch_y[rand2]
                
    
    def reset(self):
        """Reset the generator to start from the beginning of the list of files."""
        
        self.current_file_pointer = iter(self.filenames)
        self.load_file(self.current_file_pointer.next())
