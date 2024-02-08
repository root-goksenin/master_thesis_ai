from utils import CorpusComparator, QRelsComparator
import os
    
if __name__ == "__main__":
    data_names = list(map(lambda x: os.path.join("./beir_data", x), os.listdir("./beir_data")))
    qrel_comp = QRelsComparator(*data_names)
    qrel_comp.check_query_overlap("query_similarities.json")
