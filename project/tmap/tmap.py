"""
Pure Python implementation of the TMAP algorithm.

Reference: https://github.com/reymond-group/tmap
Paper: Probst & Reymond, JCIM 2020

Pipeline: Fingerprints -> MinHash -> LSH Forest -> kNN Graph -> MST -> Layout
"""

from __future__ import annotations

import bisect
import struct
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Minhash
# ---------------------------------------------------------------------------

class Minhash:
    """Permutation-based MinHash for binary fingerprint vectors."""

    PRIME = 2_305_843_009_213_693_951  # Mersenne prime 2^61 - 1
    MAX_HASH = np.uint32(0xFFFFFFFF)

    def __init__(self, d: int = 128, seed: int = 42):
        self.d = d
        rng = np.random.RandomState(seed)
        self.perms_a = rng.randint(1, self.MAX_HASH, size=d, dtype=np.uint64)
        self.perms_b = rng.randint(0, self.MAX_HASH, size=d, dtype=np.uint64)

    def from_binary_array(self, vec: np.ndarray) -> np.ndarray:
        """MinHash a single binary vector. *vec* can be dense (0/1) or sparse
        (array of set-bit indices)."""
        indices = np.where(vec)[0].astype(np.uint64) if vec.dtype != np.uint64 else vec
        if len(indices) == 0:
            return np.full(self.d, self.MAX_HASH, dtype=np.uint32)

        # (d, len(indices))  — broadcast: h_j(i) = ((a_j * i + b_j) % prime) & max_hash
        hashes = (
            (self.perms_a[:, None] * indices[None, :] + self.perms_b[:, None])
            % self.PRIME
        ) & int(self.MAX_HASH)
        return hashes.min(axis=1).astype(np.uint32)

    def batch_from_binary_array(self, vecs: np.ndarray) -> np.ndarray:
        """MinHash a batch of binary vectors (n x bits) -> (n x d)."""
        return np.array([self.from_binary_array(v) for v in vecs], dtype=np.uint32)


# ---------------------------------------------------------------------------
# LSHForest
# ---------------------------------------------------------------------------

def _byte_key(arr: np.ndarray) -> bytes:
    """Convert a uint32 slice to a sortable bytes key (big-endian)."""
    return b"".join(struct.pack(">I", int(v)) for v in arr)


class LSHForest:
    """Locality-Sensitive Hashing Forest for approximate nearest-neighbour
    search over MinHash vectors."""

    def __init__(self, d: int = 128, l: int = 8, store: bool = True):
        if d < l:
            raise ValueError("d must be >= l")
        self.d = d
        self.l = l
        self.k = d // l
        self.store = store

        self._hashtables: list[dict[bytes, list[int]]] = [
            defaultdict(list) for _ in range(l)
        ]
        self._hashranges: list[tuple[int, int]] = [
            (i * self.k, (i + 1) * self.k) for i in range(l)
        ]
        self._sorted_keys: list[list[bytes]] = [[] for _ in range(l)]
        self._data: list[np.ndarray] = []
        self._size = 0
        self._clean = False

    # -- insertion ----------------------------------------------------------

    def add(self, vec: np.ndarray) -> None:
        if self.store:
            self._data.append(vec)
        for i, (lo, hi) in enumerate(self._hashranges):
            key = _byte_key(_swap_vec(vec[lo:hi]))
            self._hashtables[i][key].append(self._size)
        self._size += 1
        self._clean = False

    def batch_add(self, vecs: np.ndarray) -> None:
        for v in vecs:
            self.add(v)

    # -- indexing -----------------------------------------------------------

    def index(self) -> None:
        for i in range(self.l):
            self._sorted_keys[i] = sorted(self._hashtables[i].keys())
        self._clean = True

    # -- querying -----------------------------------------------------------

    def query(self, vec: np.ndarray, k: int) -> list[int]:
        results: set[int] = set()
        for r in range(self.k, 0, -1):
            self._query_internal(vec, r, results, k)
            if len(results) >= k:
                return list(results)
        return list(results)

    def _query_internal(
        self, vec: np.ndarray, r: int, results: set[int], k: int
    ) -> None:
        for i, (lo, _hi) in enumerate(self._hashranges):
            prefix = _byte_key(_swap_vec(vec[lo : lo + r]))
            prefix_len = len(prefix)

            sorted_keys = self._sorted_keys[i]
            j = bisect.bisect_left(sorted_keys, prefix)

            for idx in range(j, len(sorted_keys)):
                sk = sorted_keys[idx]
                if sk[:prefix_len] != prefix:
                    break
                for item in self._hashtables[i][sk]:
                    results.add(item)
                    if len(results) >= k:
                        return

    def query_linear_scan(
        self, vec: np.ndarray, k: int, kc: int = 10
    ) -> list[tuple[float, int]]:
        if not self.store:
            raise RuntimeError("LSHForest was not instantiated with store=True")
        if not self._clean:
            raise RuntimeError("LSHForest has not been indexed — call .index()")

        candidates = self.query(vec, k * kc)
        return self._linear_scan(vec, candidates, k)

    def query_linear_scan_by_id(
        self, idx: int, k: int, kc: int = 10
    ) -> list[tuple[float, int]]:
        return self.query_linear_scan(self._data[idx], k, kc)

    def _linear_scan(
        self, vec: np.ndarray, indices: list[int], k: int
    ) -> list[tuple[float, int]]:
        k = min(k, len(indices)) if k else len(indices)
        scored = [(self.get_distance(vec, self._data[i]), i) for i in indices]
        scored.sort()
        return scored[:k]

    # -- kNN graph ----------------------------------------------------------

    def get_knn_graph(
        self, k: int, kc: int = 10
    ) -> tuple[list[int], list[int], list[float]]:
        """Build the kNN graph over all stored entries.

        Returns (from_ids, to_ids, weights).
        """
        if not self.store or not self._clean:
            raise RuntimeError("LSHForest must be stored and indexed first")

        froms, tos, weights = [], [], []
        for i in range(self._size):
            neighbours = self.query_linear_scan(self._data[i], k, kc)
            for dist, j in neighbours:
                if j == i:
                    continue
                froms.append(i)
                tos.append(j)
                weights.append(dist)
        return froms, tos, weights

    # -- distance -----------------------------------------------------------

    @staticmethod
    def get_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return 1.0 - float(np.sum(vec_a == vec_b)) / len(vec_a)

    # -- helpers ------------------------------------------------------------

    @property
    def size(self) -> int:
        return self._size


def _swap32(v: int) -> int:
    """Byte-swap a 32-bit integer (matches the C++ Swap)."""
    return (
        ((v >> 24) & 0xFF)
        | ((v << 8) & 0xFF0000)
        | ((v >> 8) & 0xFF00)
        | ((v << 24) & 0xFF000000)
    )


def _swap_vec(arr: np.ndarray) -> np.ndarray:
    return np.array([_swap32(int(x)) for x in arr], dtype=np.uint32)


# ---------------------------------------------------------------------------
# MST — Kruskal's with union-find
# ---------------------------------------------------------------------------

class _UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def _kruskal_mst(
    n: int,
    edges: list[tuple[int, int, float]],
) -> list[tuple[int, int, float]]:
    """Kruskal's MST. Returns list of (u, v, weight) edges."""
    edges_sorted = sorted(edges, key=lambda e: e[2])
    uf = _UnionFind(n)
    mst: list[tuple[int, int, float]] = []
    for u, v, w in edges_sorted:
        if uf.union(u, v):
            mst.append((u, v, w))
            if len(mst) == n - 1:
                break
    return mst


# ---------------------------------------------------------------------------
# Spring layout — Fruchterman-Reingold (numpy-vectorised)
# ---------------------------------------------------------------------------

def _fruchterman_reingold(
    n: int,
    edges: list[tuple[int, int, float]],
    iterations: int = 500,
    k: float = 1 / 65,
    seed: int = 42,
) -> np.ndarray:
    """Return (n, 2) positions using Fruchterman-Reingold force-directed layout."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(-0.5, 0.5, size=(n, 2))

    if n <= 1 or not edges:
        return pos

    edge_src = np.array([e[0] for e in edges], dtype=np.intp)
    edge_dst = np.array([e[1] for e in edges], dtype=np.intp)

    # optimal distance
    area = 1.0
    k_opt = k if k else np.sqrt(area / n)
    temp = 0.1 * np.sqrt(area)
    dt = temp / (iterations + 1)

    for _ in range(iterations):
        # pairwise repulsion
        delta = pos[:, None, :] - pos[None, :, :]  # (n, n, 2)
        dist = np.sqrt((delta ** 2).sum(axis=2))
        np.fill_diagonal(dist, 1.0)  # avoid division by zero
        repulsion = (delta * (k_opt ** 2 / dist ** 2)[:, :, None]).sum(axis=1)

        # edge attraction
        edge_delta = pos[edge_src] - pos[edge_dst]  # (E, 2)
        edge_dist = np.sqrt((edge_delta ** 2).sum(axis=1, keepdims=True))
        edge_dist = np.maximum(edge_dist, 1e-6)
        attract_force = edge_delta * edge_dist / k_opt

        attraction = np.zeros_like(pos)
        np.add.at(attraction, edge_src, -attract_force)
        np.add.at(attraction, edge_dst, attract_force)

        # displacement
        disp = repulsion + attraction
        disp_len = np.sqrt((disp ** 2).sum(axis=1, keepdims=True))
        disp_len = np.maximum(disp_len, 1e-6)
        pos += disp / disp_len * np.minimum(disp_len, temp)

        temp -= dt

    return pos


# ---------------------------------------------------------------------------
# Radial tree layout — O(n) for trees/MSTs
# ---------------------------------------------------------------------------

def _radial_tree_layout(
    n: int,
    edges: list[tuple[int, int, float]],
    seed: int = 42,
) -> np.ndarray:
    """O(n) radial tree layout. Returns (n, 2) positions."""
    pos = np.zeros((n, 2))
    if n <= 1 or not edges:
        return pos

    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    # Root at approximate tree centroid (midpoint of diameter path)
    def _bfs(start: int) -> tuple[int, list[int]]:
        dist = [-1] * n
        dist[start] = 0
        prev = [-1] * n
        q = [start]
        h = 0
        far = start
        while h < len(q):
            u = q[h]; h += 1
            for v, _ in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    prev[v] = u
                    q.append(v)
                    if dist[v] > dist[far]:
                        far = v
        return far, prev

    e1, _ = _bfs(0)
    e2, prev2 = _bfs(e1)
    path: list[int] = []
    nd = e2
    while nd >= 0:
        path.append(nd)
        nd = prev2[nd]
    root = path[len(path) // 2]

    # BFS from root to build ordered tree
    children: list[list[int]] = [[] for _ in range(n)]
    ew = np.zeros(n)
    order: list[int] = []
    vis = bytearray(n)
    vis[root] = 1
    q2 = [root]
    h2 = 0
    while h2 < len(q2):
        u = q2[h2]; h2 += 1
        order.append(u)
        for v, w in adj[u]:
            if not vis[v]:
                vis[v] = 1
                children[u].append(v)
                ew[v] = w
                q2.append(v)

    # Post-order: leaf counts
    lc = np.zeros(n, dtype=np.int64)
    for u in reversed(order):
        if not children[u]:
            lc[u] = 1
        else:
            for c in children[u]:
                lc[u] += lc[c]

    # Pre-order: assign angular wedges proportional to subtree leaf count
    TWO_PI = 2.0 * np.pi
    a_lo = np.zeros(n)
    a_hi = np.zeros(n)
    a_hi[root] = TWO_PI
    r = np.zeros(n)

    base_step = 1.0 / (n ** 0.35)

    for u in order:
        wedge = a_hi[u] - a_lo[u]
        tot = float(max(lc[u], 1))
        cur = a_lo[u]
        for c in children[u]:
            frac = float(lc[c]) / tot
            cw = wedge * frac
            a_lo[c] = cur
            a_hi[c] = cur + cw
            mid = cur + cw * 0.5
            r[c] = r[u] + base_step * (0.5 + ew[c])
            pos[c, 0] = r[c] * np.cos(mid)
            pos[c, 1] = r[c] * np.sin(mid)
            cur += cw

    return pos


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

@dataclass
class LayoutResult:
    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    t: np.ndarray
    mst_weight: float = 0.0
    adjacency_list: list[list[tuple[int, float]]] = field(default_factory=list)


def _build_layout(
    n: int,
    mst_edges: list[tuple[int, int, float]],
    node_size: float,
    fme_iterations: int,
) -> LayoutResult:
    """Shared layout logic: MST edges -> spring layout -> normalised result."""
    mst_weight = sum(w for _, _, w in mst_edges)

    s_list, t_list = [], []
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for u, v, w in mst_edges:
        s_list.append(u)
        t_list.append(v)
        adjacency[u].append((v, w))
        adjacency[v].append((u, w))

    pos = _radial_tree_layout(n, mst_edges)
    xs, ys = pos[:, 0].copy(), pos[:, 1].copy()

    for arr in (xs, ys):
        mn, mx = arr.min(), arr.max()
        diff = mx - mn
        if diff > 0:
            arr[:] = (arr - mn) / diff - 0.5

    return LayoutResult(
        x=xs,
        y=ys,
        s=np.array(s_list, dtype=np.uint32),
        t=np.array(t_list, dtype=np.uint32),
        mst_weight=mst_weight,
        adjacency_list=adjacency,
    )


def layout_from_lsh_forest(
    lsh_forest: LSHForest,
    k: int = 10,
    kc: int = 10,
    node_size: float = 1 / 65,
    fme_iterations: int = 500,
) -> LayoutResult:
    """Full TMAP pipeline: kNN graph -> MST -> force-directed layout."""
    n = lsh_forest.size
    froms, tos, weights = lsh_forest.get_knn_graph(k, kc)

    # deduplicate edges, keeping minimum weight
    edge_map: dict[tuple[int, int], float] = {}
    for u, v, w in zip(froms, tos, weights):
        w = max(w, 1e-10)
        key = (min(u, v), max(u, v))
        if key not in edge_map or w < edge_map[key]:
            edge_map[key] = w
    edge_list = [(u, v, w) for (u, v), w in edge_map.items()]

    mst_edges = _kruskal_mst(n, edge_list)
    return _build_layout(n, mst_edges, node_size, fme_iterations)


def layout_from_edge_list(
    vertex_count: int,
    edges: list[tuple[int, int, float]],
    node_size: float = 1 / 65,
    fme_iterations: int = 500,
    create_mst: bool = True,
) -> LayoutResult:
    """Layout from an explicit edge list [(from, to, weight), ...]."""
    edge_map: dict[tuple[int, int], float] = {}
    for u, v, w in edges:
        w = max(w, 1e-10)
        key = (min(u, v), max(u, v))
        if key not in edge_map or w < edge_map[key]:
            edge_map[key] = w
    edge_list = [(u, v, w) for (u, v), w in edge_map.items()]

    if create_mst:
        edge_list = _kruskal_mst(vertex_count, edge_list)

    return _build_layout(vertex_count, edge_list, node_size, fme_iterations)
