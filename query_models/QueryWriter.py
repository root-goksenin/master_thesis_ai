from beir.generation.models import QGenModel
from beir.datasets.data_loader import GenericDataLoader
import random
from gpl_improved.query_models import QueryAugmentMod, QueryGenerator, QAugmentModel
import logging, os


def qgen(
    data_path,
    output_dir,
    generator_name_or_path ="BeIR/query-gen-msmarco-t5-base-v1",
    forward_model_path = "Helsinki-NLP/opus-mt-en-fr", 
    back_model_path = "Helsinki-NLP/opus-mt-fr-en",
    ques_per_passage=3,
    bsz=4,
    top_p = 0.95,
    top_k = 25,
    max_length = 64,
    gpl_data_prefix="gpl",
    augmented : QueryAugmentMod = QueryAugmentMod.UseNew,
    augment_probability : float = 1.0,
    augment_per_query: int = 2,
    augment_temperature: float = 0.0,
):
    #### Provide the data_path where nfcorpus has been downloaded and unzipped
    corpus = GenericDataLoader(data_path).load_corpus()
    corpus = dict(random.sample(corpus.items(), 100))

    #### question-generation model loading
    if type(augmented) != QueryAugmentMod.None_:
      generator = QueryGenerator(model=QGenModel(generator_name_or_path), 
                                 augment_model = QAugmentModel(forward_model_path = forward_model_path, back_model_path = back_model_path),
      )
    else:
      generator = QueryGenerator(model=QGenModel(generator_name_or_path))
 
    #### Query-Generation using Nucleus Sampling (top_k=25, top_p=0.95) ####
    #### https://huggingface.co/blog/how-to-generate
    #### Prefix is required to seperate out synthetic queries and qrels from original
    prefix = gpl_data_prefix

    #### Generating 3 questions per passage.
    #### Reminder the higher value might produce lots of duplicates
    generator.generate(
        corpus,
        output_dir=output_dir,
        ques_per_passage=ques_per_passage,
        prefix=prefix,
        batch_size=bsz,
        top_p = top_p,
        top_k = top_k,
        max_length = max_length,
        augment_probability = augment_probability,
        augment_per_query = augment_per_query,
        augment_temperature = augment_temperature
    )


class QueryWriter:

  def __init__(self, queries_per_passage:int,  path_to_data: str, gpl_data_prefix: str= "gpl"):
    self.queries_per_passage = queries_per_passage
    self.path_to_data = path_to_data
    self.gpl_data_prefix = gpl_data_prefix
    self.generated_queries = None
    self.generated_qrels = None
    self.corpus = None
    self.logger = logging.getLogger(__name__ + ".QueryWriter")

  def generate(self, 
               use_train_qrels : bool = False, 
               generator_name: str = "BeIR/query-gen-msmarco-t5-base-v1",
               batch_size: int= 4,
               top_p = 0.95,
               top_k = 25,
               max_length = 64,
               augmented : QueryAugmentMod = QueryAugmentMod.UseNew,
               augment_probability : float = 1.0,
               forward_model_path = "Helsinki-NLP/opus-mt-en-fr", 
               back_model_path = "Helsinki-NLP/opus-mt-fr-en",
               augment_per_query : int = 2,
               augment_temperature : float = 2.0):
    assert "corpus.jsonl" in os.listdir(self.path_to_data), "At least corpus should exist!"

    if use_train_qrels == True:
      assert "qrels" in os.listdir(self.path_to_data) and "queries.jsonl" in os.listdir(self.path_to_data), "No queries found"
      self.logger.info("Loading from existing labeled data")
      corpus, gen_queries, gen_qrels = GenericDataLoader(
          self.path_to_data
      ).load(split="train")

    elif f"{self.gpl_data_prefix}-qrels" in os.listdir(
      self.path_to_data
      ) and f"{self.gpl_data_prefix}-queries.jsonl" in os.listdir(self.path_to_data):
      self.logger.info("Loading from existing generated data")
      corpus, gen_queries, gen_qrels = GenericDataLoader(
        self.path_to_data, prefix=self.gpl_data_prefix
      ).load(split="train")
    else:
      self.logger.info("No generated queries found. Now generating it")
      qgen(
          self.path_to_data,
          self.path_to_data,
          generator_name_or_path=generator_name,
          ques_per_passage=self.queries_per_passage,
          bsz=batch_size,
          gpl_data_prefix=self.gpl_data_prefix,
          top_p = top_p,
          top_k = top_k,
          max_length = max_length,
          augmented = QueryAugmentMod.UseNew,
          augment_probability  = augment_probability,
          forward_model_path = "Helsinki-NLP/opus-mt-en-fr", 
          back_model_path = "Helsinki-NLP/opus-mt-fr-en",
          augment_per_query = augment_per_query,
          augment_temperature = augment_temperature

      )
      # Returns the generated queries...
      corpus, gen_queries, gen_qrels = GenericDataLoader(
          self.path_to_data, prefix=self.gpl_data_prefix
      ).load(split="train")

    self.corpus = corpus
    self.gen_queries = gen_queries
    self.gen_qrels = gen_qrels
