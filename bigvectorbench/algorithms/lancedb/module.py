""" LanceDB module for BigVectorBench framework.

LanceDB is an embedded vector database, so unlike server-based databases
(Qdrant, Milvus, ...) it runs in-process inside the algorithm container and
needs no docker-compose service. The store is written to a local directory.

This adapter currently supports the `mm-ann` (multi-modal) and plain `ann`
workloads: vector-only search with no labels/filters.
"""

import shutil
from time import time

import numpy as np
import pyarrow as pa

import lancedb

from bigvectorbench.algorithms.base.module import BaseANN


def metric_mapping(_metric: str):
    """
    Mapping metric type to LanceDB distance type

    Args:
        _metric (str): metric type

    Returns:
        str: LanceDB distance type
    """
    _metric = _metric.lower()
    _metric_type = {
        "dot": "dot",
        "angular": "cosine",
        "euclidean": "l2",
    }.get(_metric, None)
    if _metric_type is None:
        raise ValueError(f"[LanceDB] Not support metric type: {_metric}!!!")
    return _metric_type


class LanceDB(BaseANN):
    """LanceDB base implementation (vector-only search)."""

    def __init__(self, metric: str, index_param: dict):
        self._metric = metric
        self._distance_type = metric_mapping(metric)
        self._db_path = "/tmp/lancedb_bvb"
        self._table_name = "bvb"
        self._dim = None
        self.table = None
        self.load_batch_size = 10000
        # query-time params
        self._nprobes = None
        self._ef = None
        self._refine_factor = None
        # cleanup any previous store
        shutil.rmtree(self._db_path, ignore_errors=True)
        self.db = lancedb.connect(self._db_path)
        print("[LanceDB] client connected successfully!!!")
        self.name = f"LanceDB metric:{self._metric}"
        super().__init__()

    def _build_index(self):
        """Create the vector index. Implemented by subclasses."""
        raise NotImplementedError

    def load_data(
        self,
        embeddings: np.array,
        labels: np.ndarray | None = None,
        label_names: list[str] | None = None,
        label_types: list[str] | None = None,
    ) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        num, dim = embeddings.shape
        self._dim = dim
        print(f"[LanceDB] load data: {num} vectors of dim {dim}")

        if self._table_name in self.db.table_names():
            self.db.drop_table(self._table_name)

        vector_type = pa.list_(pa.float32(), dim)
        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("vector", vector_type),
            ]
        )

        for i in range(0, num, self.load_batch_size):
            end = min(i + self.load_batch_size, num)
            ids = pa.array(np.arange(i, end, dtype=np.int64))
            vecs = pa.array(
                list(embeddings[i:end]), type=vector_type
            )
            batch = pa.table({"id": ids, "vector": vecs}, schema=schema)
            if self.table is None:
                self.table = self.db.create_table(self._table_name, data=batch)
            else:
                self.table.add(batch)
        print(f"[LanceDB] loaded {num} vectors successfully!!!")
        self.num_entities = num

    def create_index(self):
        # Synchronous create_index blocks until the index is fully built.
        t0 = time()
        self._build_index()
        print(f"[LanceDB] index built in {time() - t0:.2f}s")

    def _search(self, v, n):
        q = self.table.search(v).distance_type(self._distance_type).limit(n)
        if self._nprobes is not None:
            q = q.nprobes(self._nprobes)
        if self._ef is not None:
            q = q.ef(self._ef)
        if self._refine_factor is not None and self._refine_factor > 1:
            q = q.refine_factor(self._refine_factor)
        return q.select(["id"]).to_list()

    def query(self, v, n, filter_expr=None):
        rows = self._search(np.asarray(v, dtype=np.float32), n)
        return [row["id"] for row in rows]

    def done(self):
        try:
            self.db.drop_table(self._table_name, ignore_missing=True)
        except Exception as e:
            print(f"[LanceDB] drop_table skipped: {e}")
        shutil.rmtree(self._db_path, ignore_errors=True)


class LanceDBIVFPQ(LanceDB):
    """LanceDB IVF_PQ index."""

    def __init__(self, metric: str, index_param: dict):
        super().__init__(metric, index_param)
        self._num_partitions = index_param.get("num_partitions", 256)
        self._num_sub_vectors = index_param.get("num_sub_vectors", 64)
        self.name = (
            f"LanceDB-IVF_PQ metric:{self._metric} "
            f"num_partitions:{self._num_partitions} "
            f"num_sub_vectors:{self._num_sub_vectors}"
        )

    def _build_index(self):
        self.table.create_index(
            metric=self._distance_type,
            num_partitions=self._num_partitions,
            num_sub_vectors=self._num_sub_vectors,
            index_type="IVF_PQ",
            vector_column_name="vector",
            replace=True,
        )

    def set_query_arguments(self, nprobes, refine_factor):
        self._nprobes = nprobes
        self._ef = None
        self._refine_factor = refine_factor
        self.name = (
            f"LanceDB-IVF_PQ metric:{self._metric} "
            f"num_partitions:{self._num_partitions} "
            f"num_sub_vectors:{self._num_sub_vectors} "
            f"nprobes:{nprobes} refine_factor:{refine_factor}"
        )


class LanceDBIVFHNSW(LanceDB):
    """LanceDB IVF_HNSW (HnswSq) index with num_partitions=1."""

    def __init__(self, metric: str, index_param: dict):
        super().__init__(metric, index_param)
        self._num_partitions = index_param.get("num_partitions", 1)
        self._m = index_param.get("m", 16)
        self._ef_construction = index_param.get("ef_construction", 200)
        self.name = (
            f"LanceDB-IVF_HNSW metric:{self._metric} "
            f"num_partitions:{self._num_partitions} "
            f"m:{self._m} ef_construction:{self._ef_construction}"
        )

    def _build_index(self):
        self.table.create_index(
            metric=self._distance_type,
            num_partitions=self._num_partitions,
            index_type="IVF_HNSW_SQ",
            m=self._m,
            ef_construction=self._ef_construction,
            vector_column_name="vector",
            replace=True,
        )

    def set_query_arguments(self, ef, refine_factor):
        self._ef = ef
        self._nprobes = self._num_partitions
        self._refine_factor = refine_factor
        self.name = (
            f"LanceDB-IVF_HNSW metric:{self._metric} "
            f"num_partitions:{self._num_partitions} "
            f"m:{self._m} ef_construction:{self._ef_construction} "
            f"ef:{ef} refine_factor:{refine_factor}"
        )
