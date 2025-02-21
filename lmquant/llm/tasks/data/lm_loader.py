import os
import random
import math
import numpy as np
import json

from infinibatch import iterators, CheckpointableIterator
from .utils import FixedBlockwiseShuffleIterator, NativeCheckpointableIterator, WeightNoRandomStateIterator
from .basic_loader import BaseBatchGen
import logging
import torch

logger = logging.getLogger("LMLoader logger")

class LMLoader(BaseBatchGen):
    def __init__(
            self,
            args,
            dataset,
            tokenizer,
            max_tokens=None,
            max_sentences=None,
            max_positions=None,
            ignore_invalid_inputs=False,
            required_batch_size_multiple=1,
            seed=1,
            epoch=1,
            num_shards=1,
            shard_id=0,
            reject_sampling=1,
    ):
        super().__init__()
        self.args = args
        self.data = dataset.data
        self.data_dir = dataset.data_dir
        self.shuffle = dataset.shuffle
        self.tokenizer = tokenizer

        self.max_tokens = max_tokens
        # self.max_sentences = max_sentences
        self.max_sentences = 1
        self.max_positions = max_positions
        # self.tokens_per_sample = args.tokens_per_sample
        self.tokens_per_sample = 2048
        self.mlm_cut_length = getattr(args, "mlm_cut_length", 0)
        self.mlm_tokens_proportion = getattr(args, "mlm_tokens_proportion", 0)
        self.pad_to_max_len = getattr(args, "pad_to_max_len", False)
        self.ignore_invalid_inputs = ignore_invalid_inputs
        self.required_batch_size_multiple = required_batch_size_multiple
        self.seed = str(seed)
        self.epoch = epoch
        self.num_shards = num_shards
        self.shard_id = shard_id

        self.batch_read_ahead = args.batch_read_ahead
        self.sharded_checkpoint = True

        self._build_iter()
    
    def _build_iter(self):
        tokenized_lines = self._tokenize()
        self.padded_batches = self._batchify(tokenized_lines)
        
        
        # prefetch_batches = iterators.PrefetchIterator(
        #     self.padded_batches, 
        #     buffer_size=10, 
        #     buffer_in_main_process=True, 
        #     log_empty_buffer_warning=True and self.shard_id == 0,
        # )

        

        # prefetch_batches = iterators.MapIterator(
        #     prefetch_batches, self._move_to_tensor
        # )

        prefetch_batches = iterators.MapIterator(
            self.padded_batches, self._move_to_tensor
        )

        self._iter = prefetch_batches

    def _tokenize(self):
        '''
        data:
        {
            'source': list[Path],
        }
        '''
        dataset = list(zip(self.data['source']))

        if self.shuffle:
            chunk_files = \
                iterators.InfinitePermutationSourceIterator(
                    dataset,
                    seed=self.seed, 
                    shuffle=self.shuffle, 
                    num_instances=self.num_shards, 
                    instance_rank=self.shard_id,
                )
        else:
            chunk_files = \
                iterators.ChunkedSourceIterator(
                    dataset,
                    num_instances=self.num_shards, 
                    instance_rank=self.shard_id,
                )
        
        tokenized_lines = iterators.SelectManyIterator(chunk_files, lambda files: self._read_from_files(*files))
        tokenized_lines = iterators.SamplingRandomMapIterator(tokenized_lines, self._prepare, self.seed)
        
        return tokenized_lines

    def getstate(self):
        state = super().getstate()
        state["epoch"] = self.epoch
        state["iterations_in_epoch"] = None
        return state

    def _batchify(self, lines):
        
        if self.max_sentences is not None:
            if self.batch_read_ahead > 0:
                lines = FixedBlockwiseShuffleIterator(lines, self.batch_read_ahead, self.seed)
            batches = iterators.FixedBatchIterator(lines, self.max_sentences)
        else:
            # -
            def dynamic_batch_size(sample):
                lengths = [len(x) for x in sample]
                batch_size = self.max_tokens // max(lengths) // self.required_batch_size_multiple * self.required_batch_size_multiple
                return max(1, batch_size)
            
            batches = iterators.BucketedReadaheadBatchIterator(
                    lines,
                    read_ahead=self.batch_read_ahead, 
                    key=(lambda x: max(len(x[0]), len(x[1]))) if self.shuffle else None, 
                    batch_size=dynamic_batch_size, 
                    shuffle=self.shuffle,
                    seed=self.seed,
            )

        def collate(batch):
            batch_size = len(batch)
            gpt_max_length = max([len(x[0]) for x in batch])
            if self.pad_to_max_len:
                gpt_max_length = self.tokens_per_sample + 1

            gpt_source_ids = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32,
                                 fill_value=self.tokenizer.eos_token_id)
            gpt_target_ids = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32,
                                     fill_value=self.tokenizer.eos_token_id)
            gpt_input_mask_all = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=0)
            gpt_loss_mask_all = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=1)

            for i, (gpt_ids, gpt_input_mask,  gpt_loss_mask) in enumerate(batch):
                gpt_source_ids[i, :len(gpt_ids)-1] = gpt_ids[:-1]
                gpt_target_ids[i, :len(gpt_ids)-1] = gpt_ids[1:]
                gpt_input_mask_all[i, :len(gpt_ids)-1] = gpt_input_mask[:-1]
                gpt_loss_mask_all[i, :len(gpt_ids)-1] = gpt_loss_mask[1:]
            
            ret_batch = {
                'net_input': gpt_source_ids.astype(np.int64),
                'target': gpt_target_ids.astype(np.int64),
                'nsentences': batch_size,
                'ntokens': sum([len(x[0]) for x in batch]),
            }

            return ret_batch

        padded_batches = iterators.MapIterator(
            batches, collate
        )

        return padded_batches

    def _prepare(self, doc):
        gpt_input_mask = [0] * len(doc)
        gpt_loss_mask = [1] * len(doc)
        full_tokens = doc
        return full_tokens, gpt_input_mask, gpt_loss_mask

    def _tokenize(self):
        multilingual_iters = []
        weights = []

        for data in self.data:
            multilingual_iters.append(
                self._tokenize_foreach_lang(data)
            )
            if 'weight' in data:
                weights.append(float(data['weight']))
            else:
                weights.append(int(data['count']))
        
        if len(multilingual_iters) == 1:
            return multilingual_iters[0]

        sampling_iterator = WeightNoRandomStateIterator(weights, self.seed)
        control_iterator = NativeCheckpointableIterator(sampling_iterator)
        tokenized_lines = iterators.MultiplexIterator(control_iterator, multilingual_iters)
        
        return tokenized_lines

    def _tokenize_foreach_lang(self, data):
        # if 'epoch' in data:
        if 'source' not in data or len(data['source']) == 0:
            # load source from single file, format: self.data_dir/json/{name}.json
            file_path = os.path.join(self.data_dir, 'json', f"{data['name']}.json")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"file {file_path} not exists")
            with open(file_path, 'r', encoding='utf8') as f:
                data_source = json.load(f)
                data['source'] = data_source
        data_source = data['source']
        dataset = list(zip(data_source))
        # print('data name: ', data['name'], 'len(dataset): ', len(dataset))
        chunk_files = iterators.ChunkedSourceIterator(
            dataset,
            num_instances=self.num_shards, 
            instance_rank=self.shard_id,)
        
        tokenized_lines = iterators.SelectManyIterator(chunk_files, lambda files: self._read_from_files(*files))
        tokenized_lines = iterators.MapIterator(tokenized_lines, self._prepare)
        
        return tokenized_lines

    @staticmethod
    def _doc_to_ids(text, tokenizer=None):
        tokenized_ids = [] # list of list of ids
        lines = text.split('\n\n')
        for line_idx, line in enumerate(lines):
            suffix = '\n\n' if line_idx != len(lines) - 1 else ''
            if len(line) == 0:
                continue

            sublines = line.split('\n')
            for idx, subline in enumerate(sublines):
                # if len(subline) > 200000:
                #     continue
                if len(subline) == 0:
                    continue
                if idx == len(sublines) - 1:
                    tokenized_ids.append(tokenizer.encode(subline + suffix))
                else:
                    tokenized_ids.append(tokenizer.encode(subline + '\n'))

        tokenized_ids[-1].append(tokenizer.eos_token_id)
        return tokenized_ids

    def _read_lines(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf8') as f:
                lines = f.read().strip().split('\n')
        except:
            return iter([]) # skip bad file
        return lines
    
    def _read_from_files(self, source_file):
        self.tokenizer.add_bos_token = False

        data = []
        if self.args.absolute_path:
            file_path = source_file
        else:
            file_path = os.path.join(self.data_dir, source_file)
        
        if not os.path.exists(file_path):
            print('| file {} not exists'.format(file_path), flush=True)
            return iter([]) # skip bad file

        lines = self._read_lines(file_path)

        tokenized_ids = []
        for doc_jsonstr in lines:
            try:
                json_obj = json.loads(doc_jsonstr)
                
                if 'text' in json_obj:
                    text = json_obj['text']
                elif 'content' in json_obj:
                    text = json_obj['content']
                elif 'raw_content_lines' in json_obj:
                    text = "\n".join(json_obj['raw_content_lines'])
                else:
                    print('no text in json_obj')

                if len(text) == 0:
                    continue
                ret = LMLoader._doc_to_ids(text, self.tokenizer)
                tokenized_ids.extend(ret)
            except Exception as e:
                print(source_file, flush=True)
                print(e, flush=True)
            
        # ###################################################

        doc = [self.tokenizer.bos_token_id] 
        for ids in tokenized_ids:
            if len(doc) + len(ids) > self.tokens_per_sample + 1:
                doc.extend(ids)
                doc = doc[:self.tokens_per_sample + 1]
                data.append(doc)
                doc = [self.tokenizer.bos_token_id] 
            else:
                doc.extend(ids)

        # if len(doc) > 1 and len(doc) <= self.tokens_per_sample + 1:
        #     data.append(doc)
        self.tokenizer.add_bos_token = True
        return data
    



class LMLoader_TextInput(BaseBatchGen):
    def __init__(
            self,
            args,
            dataset,
            tokenizer,
            max_tokens=None,
            max_sentences=None,
            max_positions=None,
            ignore_invalid_inputs=False,
            required_batch_size_multiple=1,
            seed=1,
            epoch=1,
            num_shards=1,
            shard_id=0,
            reject_sampling=1,
    ):
        super().__init__()
        self.args = args
        self.data = dataset.data
        self.data_dir = dataset.data_dir
        self.shuffle = dataset.shuffle
        self.tokenizer = tokenizer

        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.max_positions = max_positions
        self.tokens_per_sample = args.tokens_per_sample
        self.mlm_cut_length = getattr(args, "mlm_cut_length", 0)
        self.mlm_tokens_proportion = getattr(args, "mlm_tokens_proportion", 0)
        self.pad_to_max_len = getattr(args, "pad_to_max_len", False)
        self.ignore_invalid_inputs = ignore_invalid_inputs
        self.required_batch_size_multiple = required_batch_size_multiple
        self.seed = str(seed)
        self.epoch = epoch
        self.num_shards = num_shards
        self.shard_id = shard_id

        self.batch_read_ahead = args.batch_read_ahead
        self.sharded_checkpoint = True

        self.current_row_num = 0

        self._build_iter()
    
    def _build_iter(self):
        tokenized_lines = self._tokenize()
        self.padded_batches = self._batchify(tokenized_lines)
        
        
        # prefetch_batches = iterators.PrefetchIterator(
        #     self.padded_batches, 
        #     buffer_size=10, 
        #     buffer_in_main_process=True, 
        #     log_empty_buffer_warning=True and self.shard_id == 0,
        # )

        

        # prefetch_batches = iterators.MapIterator(
        #     prefetch_batches, self._move_to_tensor
        # )

        prefetch_batches = iterators.MapIterator(
            self.padded_batches, self._move_to_tensor
        )

        self._iter = prefetch_batches

    def getstate(self):
        state = super().getstate()
        state["epoch"] = self.epoch
        state["iterations_in_epoch"] = None
        return state

    def _batchify(self, lines_and_name):
        lines = lines_and_name['tokenized_lines']
        data_name = lines_and_name["data_name"]

        if self.max_sentences is not None:
            batches = iterators.FixedBatchIterator(lines, self.max_sentences)
        else:
            # -
            def dynamic_batch_size(sample):
                lengths = [len(x) for x in sample]
                batch_size = self.max_tokens // max(lengths) // self.required_batch_size_multiple * self.required_batch_size_multiple
                return max(1, batch_size)
            
            batches = iterators.BucketedReadaheadBatchIterator(
                    lines,
                    read_ahead=self.batch_read_ahead, 
                    key=(lambda x: max(len(x[0]), len(x[1]))) if self.shuffle else None, 
                    batch_size=dynamic_batch_size, 
                    shuffle=self.shuffle,
                    seed=self.seed,
            )

        def collate(batch):
            logging.info(f"current row num = {self.current_row_num}")
            batch_size = len(batch)

            self.current_row_num += batch_size

            gpt_max_length = max([len(x) for x in batch])
            if self.pad_to_max_len:
                gpt_max_length = self.tokens_per_sample

            for x in batch:
                assert len(x) == self.tokens_per_sample, NotImplementedError

            gpt_source_ids = np.full(shape=(batch_size, gpt_max_length), dtype=np.int32,
                                 fill_value=0)
            
            for i, gpt_ids in enumerate(batch):
                gpt_source_ids[i, :len(gpt_ids)] = gpt_ids
            
            ret_batch = {
                'net_input': gpt_source_ids.astype(np.int64),
                'nsentences': batch_size,
                'ntokens': sum([len(x) for x in batch]),
                'data_name': data_name
            }

            return ret_batch

        padded_batches = iterators.MapIterator(
            batches, collate
        )

        return padded_batches

    def _tokenize(self):
        multilingual_iters = []
        weights = []

        for data in self.data:
            multilingual_iters.append(
                self._tokenize_foreach_lang(data)
            )
            if 'weight' in data:
                weights.append(float(data['weight']))
            else:
                weights.append(int(data['count']))
        
        if len(multilingual_iters) == 1:
            return multilingual_iters[0]

        sampling_iterator = WeightNoRandomStateIterator(weights, self.seed)
        control_iterator = NativeCheckpointableIterator(sampling_iterator)
        tokenized_lines = iterators.MultiplexIterator(control_iterator, multilingual_iters)
        
        return tokenized_lines

    def _tokenize_foreach_lang(self, data):
        # if 'epoch' in data:
        if 'source' not in data or len(data['source']) == 0:
            # load source from single file, format: self.data_dir/json/{name}.json
            file_path = os.path.join(self.data_dir, 'json', f"{data['name']}.json")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"file {file_path} not exists")
            with open(file_path, 'r', encoding='utf8') as f:
                data_source = json.load(f)
                data['source'] = data_source

        data_source = data['source']
        temp_list = data_source
        dataset = list(zip(temp_list))

        # print('data name: ', data['name'], 'len(dataset): ', len(dataset))
        chunk_files = iterators.ChunkedSourceIterator(
            dataset,
            num_instances=self.num_shards, 
            instance_rank=self.shard_id,)
        
        tokenized_lines = iterators.SelectManyIterator(chunk_files, lambda files: self._read_from_files(*files))
        
        return {"tokenized_lines": tokenized_lines, "data_name": data['name']}

    @staticmethod
    def _doc_to_ids(text, tokenizer=None):
        tokenized_ids = [] # list of list of ids
        lines = text.split('\n\n')
        for line_idx, line in enumerate(lines):
            suffix = '\n\n' if line_idx != len(lines) - 1 else ''
            if len(line) == 0:
                continue

            sublines = line.split('\n')
            for idx, subline in enumerate(sublines):
                if len(subline) > 200000:
                    continue
                if len(subline) == 0:
                    continue
                if idx == len(sublines) - 1:
                    tokenized_ids.append(tokenizer.encode(subline + suffix))
                else:
                    tokenized_ids.append(tokenizer.encode(subline + '\n'))

        tokenized_ids[-1].append(tokenizer.eos_token_id)
        return tokenized_ids

    def _read_lines(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf8') as f:
                lines = f.read().strip().split('\n')
        except:
            return iter([]) # skip bad file
        return lines
    
    def _read_from_files(self, source_file):
        self.tokenizer.add_bos_token = False

        data = []
        if self.args.absolute_path:
            file_path = source_file
        else:
            file_path = os.path.join(self.data_dir, source_file)
        
        if not os.path.exists(file_path):
            print('| file {} not exists'.format(file_path), flush=True)
            return iter([]) # skip bad file

        lines = self._read_lines(file_path)

        tokenized_ids = []
        for doc_jsonstr in lines:
            try:
                json_obj = json.loads(doc_jsonstr)
                
                if 'text' in json_obj:
                    text = json_obj['text']
                elif 'content' in json_obj:
                    text = json_obj['content']
                elif 'raw_content_lines' in json_obj:
                    text = "\n".join(json_obj['raw_content_lines'])
                else:
                    print('no text in json_obj')

                if len(text) == 0:
                    continue

                ret = LMLoader_TextInput._doc_to_ids(text, self.tokenizer)
                tokenized_ids.extend(ret)
            except Exception as e:
                print(source_file, flush=True)
                print(e, flush=True)
            
        # ###################################################

        doc = [self.tokenizer.bos_token_id]
        for ids in tokenized_ids:
            if len(doc) + len(ids) > self.tokens_per_sample:
                doc.extend(ids)
                doc = doc[:self.tokens_per_sample]
                data.append(doc)
                doc = [self.tokenizer.bos_token_id]
            else:
                doc.extend(ids)

        logging.info(f"*** \n now in {source_file} \n ***")
        logging.info(f"*** \n total_rows = {len(data)} \n ***")
        self.current_row_num = 0

        self.tokenizer.add_bos_token = True
        return data
    




class LMLoader_Args_and_Outputs(BaseBatchGen):
    def __init__(
            self,
            args,
            dataset,
            layer_idx,
            max_tokens=None,
            max_sentences=None,
            max_sentences_training=None,
            max_positions=None,
            ignore_invalid_inputs=False,
            required_batch_size_multiple=1,
            seed=1,
            epoch=1,
            num_shards=1,
            shard_id=0,
            reject_sampling=1,
    ):
        super().__init__()
        self.args = args
        self.data = dataset.data
        self.data_dir = dataset.data_dir
        self.shuffle = dataset.shuffle
        self.layer_idx = layer_idx

        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.max_sentences_training = max_sentences_training
        self.max_positions = max_positions
        self.tokens_per_sample = args.tokens_per_sample
        self.mlm_cut_length = getattr(args, "mlm_cut_length", 0)
        self.mlm_tokens_proportion = getattr(args, "mlm_tokens_proportion", 0)
        self.pad_to_max_len = getattr(args, "pad_to_max_len", False)
        self.ignore_invalid_inputs = ignore_invalid_inputs
        self.required_batch_size_multiple = required_batch_size_multiple
        self.seed = str(seed)
        self.epoch = epoch
        self.num_shards = num_shards
        self.shard_id = shard_id

        self.batch_read_ahead = args.batch_read_ahead
        self.sharded_checkpoint = True

        self.current_row_num = 0

        self._build_iter()
    
    def _build_iter(self):
        self.padded_batches = self._get_args()
        
        
        # prefetch_batches = iterators.PrefetchIterator(
        #     self.padded_batches, 
        #     buffer_size=10, 
        #     buffer_in_main_process=True, 
        #     log_empty_buffer_warning=True and self.shard_id == 0,
        # )

        

        # prefetch_batches = iterators.MapIterator(
        #     prefetch_batches, self._move_to_tensor
        # )

        prefetch_batches = self.padded_batches
        self._iter = prefetch_batches

    def getstate(self):
        state = super().getstate()
        state["epoch"] = self.epoch
        state["iterations_in_epoch"] = None
        return state

    def _get_args(self):
        multilingual_iters = []
        weights = []

        for data in self.data:
            multilingual_iters.append(
                self._args_foreach_lang(data)
            )
            if 'weight' in data:
                weights.append(float(data['weight']))
            else:
                weights.append(int(data['count']))
        
        if len(multilingual_iters) == 1:
            return multilingual_iters[0]

        sampling_iterator = WeightNoRandomStateIterator(weights, self.seed)
        control_iterator = NativeCheckpointableIterator(sampling_iterator)
        args_and_outputs = iterators.MultiplexIterator(control_iterator, multilingual_iters)
        
        return args_and_outputs

    def _args_foreach_lang(self, data):
        # if 'epoch' in data:
        if 'source' not in data or len(data['source']) == 0:
            # load source from single file, format: self.data_dir/json/{name}.json
            file_path_input = os.path.join(self.data_dir, 'json', f"{data['name']}_input_args_layer_{self.layer_idx}.json")
            file_path_output = os.path.join(self.data_dir, 'json', f"{data['name']}_teacher_output_layer_{self.layer_idx}.json")
            if not os.path.exists(file_path_input):
                raise FileNotFoundError(f"file {file_path_input} not exists")
            if not os.path.exists(file_path_output):
                raise FileNotFoundError(f"file {file_path_output} not exists")
            with open(file_path_input, 'r', encoding='utf8') as f:
                data_source_input = json.load(f)
                data['source_input'] = data_source_input
            with open(file_path_output, 'r', encoding='utf8') as f:
                data_source_output = json.load(f)
                data['source_output'] = data_source_output

        data_source_input = data['source_input']
        data_source_output = data['source_output']
        assert len(data_source_input) == len(data_source_output)
        dataset = [[data_source_input_item, data_source_output_item] for data_source_input_item, data_source_output_item in zip(data_source_input, data_source_output)]

        # print('data name: ', data['name'], 'len(dataset): ', len(dataset))
        chunk_files = iterators.ChunkedSourceIterator(
            dataset,
            num_instances=self.num_shards, 
            instance_rank=self.shard_id,)
        
        args_and_outputs = iterators.SelectManyIterator(chunk_files, lambda files: self._read_from_files(*files))
        
        return args_and_outputs

    
    def _read_from_files(self, file_path_input, file_path_output):
        if not self.args.absolute_path:
            file_path_input = os.path.join(self.data_dir, file_path_input)
            file_path_output = os.path.join(self.data_dir, file_path_output)
        
        if not os.path.exists(file_path_input):
            print('| file {} not exists'.format(file_path_input), flush=True)
            return iter([]) # skip bad file
        if not os.path.exists(file_path_output):
            print('| file {} not exists'.format(file_path_output), flush=True)
            return iter([]) # skip bad file

        # args: tuple, len=1, tuple[0] is a tensor, shape (batch_size, token_num_per_line, dim_size)
        args = torch.load(file_path_input)
        outputs = torch.load(file_path_output)

        ret_args_and_outputs = []
        if self.max_sentences is not None and \
        self.max_sentences_training is not None and \
        self.max_sentences_training < self.max_sentences:
            split_num = self.max_sentences // self.max_sentences_training
            split_args = (torch.split(tensor, self.max_sentences_training) for tensor in args)
            split_outputs = (torch.split(tensor, self.max_sentences_training) for tensor in outputs)
            for i in range(split_num):
                ret_args_and_outputs = ret_args_and_outputs + [
                    {"args": split_arg[i], "teacher_outputs": split_output[i]}
                    for split_arg, split_output in zip(split_args, split_outputs)
                ]
        else:
            ret_args_and_outputs = [{"args": args, "teacher_outputs": outputs, 
                                     "input_file": file_path_input, "output_file": file_path_output}]

        return ret_args_and_outputs
    
