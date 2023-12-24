from sentence_transformers import CrossEncoder
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import tqdm 
import logging
from torch.optim import Adam
import os 
from gpl_improved.loss import MarginDistillationLoss
from gpl_improved.hard_negative_dataset import HardNegativeDataset, hard_negative_collate_fn
from gpl_improved.utils import load_sbert, batch_to_device
from sentence_transformers.readers.InputExample import InputExample
import torch
from gpl_improved.RetriverWriter import EvaluateGPL, RetriverWriter
from beir.datasets.data_loader import GenericDataLoader


class PseudoLabeler(object):
    def __init__(
        self,
        generated_path,
        gen_queries,
        corpus,
        total_steps,
        batch_size,
        cross_encoder,
        max_seq_length,
        gpl_score_function,
        base_model
    ):
        assert "hard-negatives.jsonl" in os.listdir(generated_path)
        fpath_hard_negatives = os.path.join(generated_path, "hard-negatives.jsonl")
        self.cross_encoder = CrossEncoder(cross_encoder)
        hard_negative_dataset = HardNegativeDataset(
            fpath_hard_negatives, gen_queries, corpus
        )
        self.hard_negative_dataloader = DataLoader(
            hard_negative_dataset, shuffle=True, batch_size=batch_size, drop_last=True
        )
        self.hard_negative_dataloader.collate_fn = hard_negative_collate_fn
        self.output_path = os.path.join(generated_path, "gpl-training-data.tsv")
        self.total_steps = total_steps

        #### retokenization
        self.retokenizer = AutoTokenizer.from_pretrained(cross_encoder)
        self.max_seq_length = max_seq_length
        self.logger = logging.getLogger(__name__ + ".PseudoLabeler")

        self.bi_retriver = load_sbert(base_model, pooling = None, max_seq_length = 350).cuda()
        self.loss = MarginDistillationLoss(model=self.bi_retriver, similarity_fct=gpl_score_function.value).cuda()
        self.optimizer_bi = Adam(self.bi_retriver.parameters(), lr=0.00001)
        self.device = torch.device("cuda")
        self.path = generated_path
        self.base_model = base_model
        # self.optimizer_cross = Adam(self.cross_encoder.parameters())
    def retokenize(self, texts):
        ## We did this retokenization for two reasons:
        ### (1) Setting the max_seq_length;
        ### (2) We cannot simply use CrossEncoder(cross_encoder, max_length=max_seq_length),
        ##### since the max_seq_length will then be reflected on the concatenated sequence,
        ##### rather than the two sequences independently
        texts = list(map(lambda text: text.strip(), texts))
        features = self.retokenizer(
            texts,
            padding=True,
            truncation="longest_first",
            return_tensors="pt",
            max_length=self.max_seq_length,
        )
        decoded = self.retokenizer.batch_decode(
            features["input_ids"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return decoded
    def tokenize(self, texts):
        """
        Tokenizes the texts
        """
        return self.bi_retriver._first_module().tokenize(texts)

    def smart_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of InputExample instances: [InputExample(...), ...]

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        texts = [example.texts for example in batch]
        sentence_features = [self.tokenize(sentence) for sentence in zip(*texts)]
        labels = torch.tensor([example.label for example in batch])
        return sentence_features, labels
      
    def train(self):
        number_of_data_points = len(self.hard_negative_dataloader.dataset)
        batch_size = self.hard_negative_dataloader.batch_size

        if number_of_data_points < batch_size:
            raise ValueError(
                "Batch size larger than number of data points / generated queries "
                f"(batch size: {batch_size}, "
                f"number of data points / generated queries: {number_of_data_points})"
            )


        hard_negative_iterator = iter(self.hard_negative_dataloader)
        self.logger.info("Begin training")
        for _ in tqdm.trange(self.total_steps):
            try:
                batch = next(hard_negative_iterator)
            except StopIteration:
                hard_negative_iterator = iter(self.hard_negative_dataloader)
                batch = next(hard_negative_iterator)

            (query_id, pos_id, neg_id), (query, pos, neg) = batch
            query, pos, neg = [self.retokenize(texts) for texts in [query, pos, neg]]
            scores = self.cross_encoder.predict(
                list(zip(query, pos)) + list(zip(query, neg)), show_progress_bar=False
            )
            labels = scores[: len(query)] - scores[len(query) :]
            labels = (
                labels.tolist()
            )  # Using `tolist` will keep more precision digits!!!

            train_examples = [InputExample(texts=[q, p, n], label=label) for q,p,n,label in zip(query,pos,neg,labels)]
            train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=len(train_examples))
            train_dataloader.collate_fn = self.smart_batching_collate
            for features, labels in train_dataloader:
              labels = labels.to(self.device)
              features = list(map(lambda batch: batch_to_device(batch, self.device), features))
              loss = self.loss(features,labels)
              loss.backward()
              self.optimizer_bi.step()
              self.optimizer_bi.zero_grad()
        return self.bi_retriver
      
    def eval(self):
      self.logger.info("Doing evaluation for GPL")
      corpus, queries, qrels = GenericDataLoader(self.path).load(split="test")
      retriver = EvaluateGPL(self.bi_retriver, queries, corpus)
      writer = RetriverWriter(retriver = retriver, output_dir = os.path.join("/content/results", "fiqa", self.base_model + "_GPL", "test"))
      writer.evaluate_query_based(qrels)
      writer.evaluate_beir_format(qrels)



