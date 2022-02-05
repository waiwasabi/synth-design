import numpy as np
import time
from tqdm import tqdm
from rdkit.DataStructs.cDataStructs import CreateFromBitString, TanimotoSimilarity
from rdkit.Chem.Fingerprints.SimilarityScreener import TopNScreener
import pandas as pd


def tanimoto(probe, target):
    target = np.fromstring(target,'u1') - ord('0')
    return np.count_nonzero(np.bitwise_and(probe, target)) / np.count_nonzero(np.bitwise_or(probe, target))


def top_n_chunked_tanimoto(n: int, probe: str, target_path: str, chunksize=25000) -> list:
    """
    :param n: number of similar fingerprints to return
    :param probe: (str) a bitstring (string of 0s and 1s) representation of the probe fingerprint
    :param target_path: (str) a path to a csv file of bitstring fingerprints for searching
    :param chunksize: number of samples to load into memory at a time
    :return: a list of n fingerprints with the top n similarity scores compared to the probe.

    """
    df = pd.read_csv(target_path, chunksize=chunksize)
    out = []
    for chunk in df:
        fps = chunk['BitString']
        out.extend([x for x in TopNScreener(n, probe=CreateFromBitString(probe),
                                            metric=TanimotoSimilarity, dataSource=fps,
                                            fingerprinter=CreateFromBitString)])
    print([n[0] for n in out])
    return sorted(out, key=lambda x: x[0])[-n:]


if __name__ == '__main__':
    pass
