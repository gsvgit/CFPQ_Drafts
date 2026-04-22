"""
lacfpq_blocked.py
=================
Single-Source CFPQ via RSM — block-matrix formulation in python-graphblas.

Every Python loop over RSM states and transitions is replaced by a small
number of sparse matrix multiplications that operate on the whole state
space at once.

Core idea — block matrix stacking
----------------------------------
Instead of keeping one |V|x|V| matrix per RSM state:

    P[q], M[q]  in {0,1}^{VxV}    (one pair per state q)

we stack all of them into a single  QV x V  matrix:

    P_block[ q_idx * V + u,  v ] = P[q][u, v]
    M_block[ q_idx * V + u,  v ] = M[q][u, v]

where  q_idx  is the canonical 0-based index of state q.

Routing matrix  K_l  (precomputed once per RSM)
------------------------------------------------
    K_l = (N_l)^T otimes I_V   in {0,1}^{QV x QV}

where N_l is the QxQ RSM transition matrix for label l.
K_l[q'*V+i, q*V+i] = 1  whenever q -l-> q',  for all i in [0,V).

Key identity (replaces per-state Phase-2 loop):

    block-row q' of  K_l @ (M_block @ G^l)
      =  OR_{q : N_l[q,q']=1} ( M[q] @ G^l )

Selector matrices  (precomputed once per RSM)
----------------------------------------------
  Final_sel[A]  in {0,1}^{VxQV} -- I_V block at every final state of B_A.
      ( Final_sel[A] @ M_block )[u,v]
          = OR_{q_f in F_A}  M[q_f][u,v]
      Replaces the per-final-state loop in Phase 1.

  Caller_sel[A] in {0,1}^{VxQV} -- same structure for caller states
      (states q_c with a q_c -A-> q_r transition).
      Caller_sel[A] @ M_block   then   reduce_columnwise()
      gives the call-mask for Phase 3 with no per-transition loop.

Phase summary
-------------
Phase | Original loop                     | Replacement
------+-----------------------------------+----------------------------------
  1   | for q in finals(A)                | Final_sel[A]  @ M_block  (VxQV @ QVxV)
  2   | for q in all_states x labels      | K_l @ (M_block @ G^l)    one per label
  3   | for t in nt_transitions           | Caller_sel[A] @ M_block  then col-reduce
  4   | for t in nt_transitions           | K_A @ (P_block @ dG[A])  one per NT
  5   | for q in all_states               | single nvals check on New_block

Dependencies
------------
    pip install python-graphblas
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from graphblas import Matrix, Vector, binary, monoid, semiring


# ===============================================================================
# 1.  RSM DATA STRUCTURES
# ===============================================================================

@dataclass
class RSMBox:
    """One DFA component of an RSM, corresponding to a single non-terminal."""
    non_terminal: str
    all_states:   List[int]       # globally unique state IDs
    start_state:  int             # unique entry point
    final_states: FrozenSet[int]  # accepting / exit states
    transitions:  List[Tuple[int, str, int]]  # (src, label, dst)


@dataclass
class RSM:
    """
    Recursive State Machine.

    Non-terminal labels on transitions represent recursive calls to the
    corresponding box; terminal labels consume one graph edge.
    """
    non_terminals:  FrozenSet[str]
    terminals:      FrozenSet[str]
    start_nt:       str
    boxes:          Dict[str, RSMBox]

    # Derived fields populated by __post_init__ -- do not set manually.
    all_states:     List[int]                        = field(default_factory=list, init=False, repr=False)
    state_to_nt:    Dict[int, str]                   = field(default_factory=dict, init=False, repr=False)
    q_outgoing:     Dict[int, List[Tuple[str, int]]] = field(default_factory=dict, init=False, repr=False)
    nt_transitions: List[Tuple[int, str, int]]       = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        for nt, box in self.boxes.items():
            for s in box.all_states:
                self.all_states.append(s)
                self.state_to_nt[s] = nt
                self.q_outgoing[s]  = []
            for src, lbl, dst in box.transitions:
                self.q_outgoing[src].append((lbl, dst))
                if lbl in self.non_terminals:
                    self.nt_transitions.append((src, lbl, dst))


def build_rsm(
    productions: Dict[str, List[List[str]]],
    start_nt:    str,
    terminals:   Set[str],
) -> RSM:
    """
    Build an RSM from a plain CFG.

    Each production  A -> X1 X2 ... Xn  becomes a chain of states in B_A
    with the last state added to F_A.  An empty RHS  []  makes s_A final
    (epsilon-production).  Multiple productions for the same NT share s_A
    (NFA-style branching; no determinisation required).

    Example
    -------
        build_rsm({"S": [["a", "S", "b"], ["a", "b"]]}, "S", {"a", "b"})
    """
    non_terminals: Set[str] = set(productions.keys())
    counter: List[int] = [0]

    def _new() -> int:
        s = counter[0]; counter[0] += 1; return s

    boxes: Dict[str, RSMBox] = {}
    for nt, prods in productions.items():
        start  = _new()
        all_st: List[int]                  = [start]
        finals: Set[int]                   = set()
        trans:  List[Tuple[int, str, int]] = []
        for rhs in prods:
            if not rhs:
                finals.add(start)
            else:
                cur = start
                for sym in rhs:
                    nxt = _new()
                    all_st.append(nxt)
                    trans.append((cur, sym, nxt))
                    cur = nxt
                finals.add(cur)
        boxes[nt] = RSMBox(nt, all_st, start, frozenset(finals), trans)

    return RSM(
        non_terminals = frozenset(non_terminals),
        terminals     = frozenset(terminals),
        start_nt      = start_nt,
        boxes         = boxes,
    )


def build_rsm_raw(
    raw_boxes: Dict[str, Tuple[int, List[int], List[Tuple[int,str,int]]]],
    start_nt:    str,
    terminals:   Set[str],
) -> RSM:
    """
    Build an RSM from a set of transitions.

!!!!
    Each production  A -> X1 X2 ... Xn  becomes a chain of states in B_A
    with the last state added to F_A.  An empty RHS  []  makes s_A final
    (epsilon-production).  Multiple productions for the same NT share s_A
    (NFA-style branching; no determinisation required).

    Example
    -------
        build_rsm({"S": [["a", "S", "b"], ["a", "b"]]}, "S", {"a", "b"})
    """
    non_terminals: Set[str] = set(raw_boxes.keys())

    boxes: Dict[str, RSMBox] = {}
    for nt, (start, finals, transitions) in raw_boxes.items():
        all_st: List[int]                  = [start]
        for (state_from,smb,state_to) in transitions:
            all_st.append(state_from)
            all_st.append(state_to)
        boxes[nt] = RSMBox(nt, list(frozenset(all_st)), start, frozenset(finals), transitions)

    return RSM(
        non_terminals = frozenset(non_terminals),
        terminals     = frozenset(terminals),
        start_nt      = start_nt,
        boxes         = boxes,
    )


# ===============================================================================
# 2.  PRECOMPUTED RSM MATRICES
# ===============================================================================

@dataclass
class RSMPrecomputed:
    """
    Fixed sparse matrices derived from the RSM and the graph vertex count.
    Computed once by precompute(); reused across all single-source queries.

    Attributes
    ----------
    Q            : number of RSM states.
    V            : number of graph vertices.
    QV           : Q * V  (block-matrix row count).
    state_order  : canonical list of RSM state IDs (length Q).
    idx          : RSM state -> 0-based position in state_order.
    K            : label -> Kronecker routing matrix K_l  (QV x QV).
    Final_sel    : NT -> final-state selector matrix       (V  x QV).
    Caller_sel   : NT -> caller-state selector matrix      (V  x QV).
    nt_start_idx : NT -> 0-based index of the box start state.
    all_labels   : every label (terminal or NT) with at least one RSM transition.
    """
    Q:            int
    V:            int
    QV:           int
    state_order:  List[int]
    idx:          Dict[int, int]
    K:            Dict[str, Matrix]
    Final_sel:    Dict[str, Matrix]
    Caller_sel:   Dict[str, Matrix]
    nt_start_idx: Dict[str, int]
    all_labels:   List[str]


def precompute(rsm: RSM, n_vertices: int) -> RSMPrecomputed:
    """
    Build all fixed routing and selector matrices for an RSM and graph size.

    Kronecker routing matrix  K_l = (N_l)^T otimes I_V
    ---------------------------------------------------
    For each RSM transition  q -l-> q'  the Kronecker product places V
    diagonal entries at positions  (q'*V+i, q*V+i) for i=0..V-1.
    Built with vectorised numpy index arithmetic, then loaded into GraphBLAS
    via Matrix.from_coo -- no Python loop over vertices.

    Selector matrices
    -----------------
    For a list of state indices S, the selector has
        sel[i,  s*V + i] = 1   for s in S,  i in [0, V).
    One numpy broadcast expression constructs all (row, col) pairs at once.
    """
    Q  = len(rsm.all_states)
    V  = n_vertices
    QV = Q * V

    state_order: List[int]      = list(rsm.all_states)
    idx:         Dict[int, int] = {q: i for i, q in enumerate(state_order)}

    # Collect (src_idx, dst_idx) pairs per label
    trans_by_label: Dict[str, Tuple[List[int], List[int]]] = defaultdict(lambda: ([], []))
    for q in rsm.all_states:
        for lbl, dst in rsm.q_outgoing[q]:
            trans_by_label[lbl][0].append(idx[q])
            trans_by_label[lbl][1].append(idx[dst])

    # Kronecker routing matrices K_l (QV x QV)
    K:          Dict[str, Matrix] = {}
    all_labels: List[str]         = []
    v_range = np.arange(V, dtype=np.int64)

    for lbl, (src_list, dst_list) in trans_by_label.items():
        src_arr = np.asarray(src_list, dtype=np.int64)  # q  index
        dst_arr = np.asarray(dst_list, dtype=np.int64)  # q' index
        # Outer broadcast: one row per transition, V entries per row
        row_k = (dst_arr[:, None] * V + v_range[None, :]).ravel()  # q'*V + i
        col_k = (src_arr[:, None] * V + v_range[None, :]).ravel()  # q *V + i
        K[lbl] = Matrix.from_coo(
            row_k, col_k,
            np.ones(len(row_k), dtype=bool),
            nrows=QV, ncols=QV, dtype=bool,
        )
        all_labels.append(lbl)

    # Selector builder shared by Final_sel and Caller_sel
    def _make_selector(state_indices: List[int]) -> Matrix:
        """V x QV Boolean matrix with an I_V block at each listed state index."""
        if not state_indices:
            return Matrix(bool, nrows=V, ncols=QV)
        s_arr = np.asarray(state_indices, dtype=np.int64)
        row_s = np.tile(v_range, len(s_arr))                        # i
        col_s = (s_arr[:, None] * V + v_range[None, :]).ravel()     # s*V + i
        return Matrix.from_coo(
            row_s, col_s,
            np.ones(len(row_s), dtype=bool),
            nrows=V, ncols=QV, dtype=bool,
        )

    # Final-state selector per NT
    Final_sel: Dict[str, Matrix] = {
        nt: _make_selector([idx[qf] for qf in box.final_states])
        for nt, box in rsm.boxes.items()
    }

    # Caller-state selector per NT
    # A caller of NT A is any state q_c with a q_c -A-> q_r transition.
    callers_of: Dict[str, Set[int]] = {nt: set() for nt in rsm.non_terminals}
    for q_c, lbl, _ in rsm.nt_transitions:
        callers_of[lbl].add(idx[q_c])
    Caller_sel: Dict[str, Matrix] = {
        nt: _make_selector(sorted(callers_of[nt]))
        for nt in rsm.non_terminals
    }

    nt_start_idx: Dict[str, int] = {
        nt: idx[box.start_state] for nt, box in rsm.boxes.items()
    }

    return RSMPrecomputed(
        Q            = Q,
        V            = V,
        QV           = QV,
        state_order  = state_order,
        idx          = idx,
        K            = K,
        Final_sel    = Final_sel,
        Caller_sel   = Caller_sel,
        nt_start_idx = nt_start_idx,
        all_labels   = all_labels,
    )


# ===============================================================================
# 3.  GRAPHBLAS HELPERS
# ===============================================================================

_LOR = semiring.lor_land[bool]
_OR  = binary.lor[bool]


def _empty(nrows: int, ncols: int) -> Matrix:
    """Allocate an empty Boolean sparse matrix."""
    return Matrix(bool, nrows=nrows, ncols=ncols)


def _or_into(dst: Matrix, src: Matrix) -> None:
    """In-place Boolean OR:  dst |= src."""
    if src.nvals > 0:
        dst << dst.ewise_add(src, op=_OR)


def _new_only(product: Matrix, visited: Matrix) -> Matrix:
    """
    Boolean set-difference via complement structural mask:
        return { (i,j) in product  |  (i,j) not in visited }
    """
    if product.nvals == 0:
        return product.dup()
    if visited.nvals == 0:
        return product.dup()
    result = _empty(product.nrows, product.ncols)
    result(~visited.S) << product
    return result


def _inject_diagonal(call_mask: Vector, s_A_idx: int, QV: int) -> Matrix:
    """
    Build the QV x V seed matrix for a fresh box-entry at RSM state s_A:

        result[s_A * V + v,  v] = True   for each v present in call_mask.

    Vectorised numpy index arithmetic -- no Python loop over vertices.
    """
    V = call_mask.size
    if call_mask.nvals == 0:
        return _empty(QV, V)
    v_idxs, _ = call_mask.to_coo()
    row_idxs   = s_A_idx * V + v_idxs
    return Matrix.from_coo(
        row_idxs, v_idxs,
        np.ones(len(v_idxs), dtype=bool),
        nrows=QV, ncols=V, dtype=bool,
    )


# ===============================================================================
# 4.  LACFPQ -- BLOCK-MATRIX ALGORITHM
# ===============================================================================

def lacfpq(
    pre:               RSMPrecomputed,
    rsm:               RSM,
    terminal_matrices: Dict[str, Matrix],
    start_vertex:      int,
) -> Set[int]:
    """
    Single-source CFPQ reachability using block-matrix sparse linear algebra.

    Parameters
    ----------
    pre              : precomputed RSM matrices -- call precompute() once
                       and reuse across queries on the same graph.
    rsm              : the RSM object.
    terminal_matrices: terminal symbol -> Boolean VxV adjacency matrix.
    start_vertex     : source vertex v_s (0-indexed).

    Returns
    -------
    Set of vertex indices w such that there is a path from start_vertex
    to w whose edge-label word is accepted by rsm.start_nt.

    Block-matrix layout
    -------------------
    P_block, M_block  in {0,1}^{QV x V}

        P_block[q_idx*V + u,  v] = 1
            iff  box(q) was entered at graph vertex u,  AND
                 RSM state q / graph vertex v are simultaneously reachable.

        M_block  = frontier (entries added in the latest BFS wave only).

    Phase 1 -- summary extraction
    ------------------------------
    For NT A, OR all M_block row-blocks belonging to final states of B_A:

        dG[A]  =  Final_sel[A] @ M_block   (VxQV @ QVxV = VxV)
               restricted to entries absent from G_nt[A].

    Phase 2 -- frontier propagation
    ---------------------------------
    For each label l, advance every state-block through graph matrix G^l,
    then route to destination state blocks via K_l:

        step  = M_block  @ G^l       (QVxV @ VxV  = QVxV)
        new_l = K_l      @ step      (QVxQV @ QVxV = QVxV)
        New_block |= new_l minus P_block

    Phase 3 -- call injection
    --------------------------
    For NT A, find current-graph-vertices v active in any caller state and
    seed fresh B_A exploration with context (s_A, v, v):

        caller_OR = Caller_sel[A] @ M_block        (VxQV @ QVxV = VxV)
        call_mask = caller_OR.reduce_columnwise()   (V-vector)
        New_block |= Diag(call_mask) at s_A block   minus P_block

    Phase 4 -- retroactive propagation
    ------------------------------------
    Stale call-site contexts in P_block continue through newly found
    summary edges dG[A]:

        step  = P_block  @ dG[A]     (QVxV  @ VxV  = QVxV)
        new_A = K_A      @ step      (QVxQV @ QVxV = QVxV)
        New_block |= new_A minus P_block

    Phase 5 -- commit
    ------------------
        P_block |= New_block;   M_block = New_block
    A single nvals check replaces the per-state changed flag.
    """
    Q  = pre.Q
    V  = pre.V
    QV = pre.QV

    # Initialise
    P_block = _empty(QV, V)
    M_block = _empty(QV, V)
    G_nt: Dict[str, Matrix] = {nt: _empty(V, V) for nt in rsm.non_terminals}

    # Seed: B_{start_nt} entered at v_s; RSM at s_S, graph at v_s.
    s_S_idx  = pre.nt_start_idx[rsm.start_nt]
    seed_row = s_S_idx * V + start_vertex
    P_block[seed_row, start_vertex] = True
    M_block[seed_row, start_vertex] = True

    while True:
        New_block = _empty(QV, V)

        # Phase 1: Summary edge extraction
        # dG[A][u,w] = 1  iff  box B_A was entered at u and just exited at w.
        delta_G: Dict[str, Matrix] = {}
        for nt_A, fs in pre.Final_sel.items():
            if fs.nvals == 0:
                delta_G[nt_A] = _empty(V, V)
                continue
            extracted     = fs.mxm(M_block, _LOR).new()   # VxV
            dG            = _new_only(extracted, G_nt[nt_A])
            delta_G[nt_A] = dG
            if dG.nvals > 0:
                _or_into(G_nt[nt_A], dG)

        # Phase 2: Frontier propagation
        # K_l @ (M_block @ G^l) routes each state-block through label-l edges
        # and delivers results to the correct destination state blocks.
        for lbl in pre.all_labels:
            G_lbl: Optional[Matrix] = (
                terminal_matrices.get(lbl) if lbl in rsm.terminals
                else G_nt.get(lbl)
            )
            if G_lbl is None or G_lbl.nvals == 0:
                continue
            K_lbl = pre.K.get(lbl)
            if K_lbl is None:
                continue
            step = M_block.mxm(G_lbl, _LOR).new()                      # QVxV
            if step.nvals == 0:
                continue
            fresh = _new_only(K_lbl.mxm(step, _LOR).new(), P_block)
            if fresh.nvals > 0:
                _or_into(New_block, fresh)

        # Phase 3: Call injection
        # Any current-graph-vertex v active in a caller-state frontier spawns
        # a fresh exploration of B_A seeded at context (s_A, v, v).
        for nt_A in rsm.non_terminals:
            cs = pre.Caller_sel[nt_A]
            if cs.nvals == 0:
                continue
            caller_OR = cs.mxm(M_block, _LOR).new()                    # VxV
            if caller_OR.nvals == 0:
                continue
            call_mask = caller_OR.reduce_columnwise(monoid.lor[bool]).new()
            if call_mask.nvals == 0:
                continue
            inject = _inject_diagonal(call_mask, pre.nt_start_idx[nt_A], QV)
            fresh  = _new_only(inject, P_block)
            if fresh.nvals > 0:
                _or_into(New_block, fresh)

        # Phase 4: Retroactive propagation
        # Old call-site contexts x newly discovered summary edges.
        # Structurally identical to Phase 2 with P_block and dG[A].
        for nt_A in rsm.non_terminals:
            dG = delta_G.get(nt_A)
            if dG is None or dG.nvals == 0:
                continue
            K_A = pre.K.get(nt_A)
            if K_A is None:
                continue
            step = P_block.mxm(dG, _LOR).new()                         # QVxV
            if step.nvals == 0:
                continue
            fresh = _new_only(K_A.mxm(step, _LOR).new(), P_block)
            if fresh.nvals > 0:
                _or_into(New_block, fresh)

        # Phase 5: Commit
        if New_block.nvals == 0:
            break
        _or_into(P_block, New_block)
        M_block = New_block

    # Extract result row from the start-NT summary matrix
    G_S = G_nt[rsm.start_nt]
    if G_S.nvals == 0:
        return set()
    row = G_S[start_vertex, :].new()
    if row.nvals == 0:
        return set()
    idxs, _ = row.to_coo()
    return {int(i) for i in idxs}


# ===============================================================================
# 5.  TEST INFRASTRUCTURE
# ===============================================================================

def _build_terminal_matrices(
    n:         int,
    edges:     List[Tuple[int, str, int]],
    terminals: Set[str],
) -> Dict[str, Matrix]:
    """Build per-terminal Boolean VxV adjacency matrices from an edge list."""
    mats: Dict[str, Matrix] = {
        t: Matrix(bool, nrows=n, ncols=n) for t in terminals
    }
    for u, lbl, v in edges:
        if lbl in mats:
            mats[lbl][u, v] = True
    return mats


_pass_n: List[int] = [0]
_fail_n: List[int] = [0]


def _run_test(
    name:     str,
    rsm:      RSM,
    n:        int,
    edges:    List[Tuple[int, str, int]],
    start:    int,
    expected: Set[int],
) -> None:
    mats = _build_terminal_matrices(n, edges, set(rsm.terminals))
    pre  = precompute(rsm, n)
    got  = lacfpq(pre, rsm, mats, start)
    ok   = got == expected
    tag  = "PASS +" if ok else "FAIL X"
    if ok:
        _pass_n[0] += 1
    else:
        _fail_n[0] += 1
    print(f"  [{tag}]  {name}")
    if not ok:
        print(f"          expected : {sorted(expected)}")
        print(f"          got      : {sorted(got)}")


# ===============================================================================
# 6.  TEST CASES
# ===============================================================================

_CYCLIC = [(0,"a",1),(1,"a",2),(2,"a",0),(1,"b",3),(3,"b",1)]


def test_cyclic_anbn_start2() -> None:
    """Cyclic graph. S->aSb|ab. L={a^n b^n | n>=1}. start=2. Expected {1,3}."""
    _run_test("cyclic a^n b^n          start=2",
              build_rsm({"S": [["a","S","b"], ["a","b"]]}, "S", {"a","b"}),
              4, _CYCLIC, 2, {1, 3})


def test_cyclic_anbn_start0() -> None:
    """Same cyclic graph, start=0. Expected {1,3}."""
    _run_test("cyclic a^n b^n          start=0",
              build_rsm({"S": [["a","S","b"], ["a","b"]]}, "S", {"a","b"}),
              4, _CYCLIC, 0, {1, 3})


def test_right_recursive() -> None:
    """Chain 0-a->1-a->2-a->3. S->aS|a. L={a^n | n>=1}. Expected {1,2,3}."""
    _run_test("right-recursive S->aS|a start=0",
              build_rsm({"S": [["a","S"], ["a"]]}, "S", {"a"}),
              4, [(0,"a",1),(1,"a",2),(2,"a",3)], 0, {1, 2, 3})


def test_left_recursive() -> None:
    """Chain 0-a->1-a->2. S->Sa|e. L={a^n | n>=0}. Expected {0,1,2}."""
    _run_test("left-recursive  S->Sa|e start=0",
              build_rsm({"S": [["S","a"], []]}, "S", {"a"}),
              3, [(0,"a",1),(1,"a",2)], 0, {0, 1, 2})


def test_dyck() -> None:
    """Linear 0-a->1-a->2-b->3-b->4. S->aSb|e. Expected {0,4}."""
    _run_test("Dyck   S->aSb|e         start=0",
              build_rsm({"S": [["a","S","b"], []]}, "S", {"a","b"}),
              5, [(0,"a",1),(1,"a",2),(2,"b",3),(3,"b",4)], 0, {0, 4})


def test_dyck_self_loop() -> None:
    """Single vertex with a/b self-loops. S->aSb|e. Expected {0}."""
    _run_test("Dyck self-loop  S->aSb|e start=0",
              build_rsm({"S": [["a","S","b"], []]}, "S", {"a","b"}),
              1, [(0,"a",0),(0,"b",0)], 0, {0})


def test_mutual_recursion() -> None:
    """S->aA, A->Sb|b. Same language as S->aSb|ab. Cyclic graph, start=2. Expected {1,3}."""
    _run_test("mutual recursion S->aA,A->Sb|b start=2",
              build_rsm({"S": [["a","A"]], "A": [["S","b"],["b"]]}, "S", {"a","b"}),
              4, _CYCLIC, 2, {1, 3})


def test_epsilon_only() -> None:
    """S->e. Only the source vertex is reachable. Expected {0}."""
    _run_test("epsilon-only    S->e     start=0",
              build_rsm({"S": [[]]}, "S", {"a"}),
              2, [(0,"a",1)], 0, {0})


def test_start_equals_final_with_cycle() -> None:
    """0-a->1-a->0 cycle plus 0-a->2. S->aSa|e. L={a^(2n) | n>=0}. Expected {0}."""
    _run_test("even-palindrome S->aSa|e start=0",
              build_rsm({"S": [["a","S","a"], []]}, "S", {"a"}),
              3, [(0,"a",1),(1,"a",0),(0,"a",2)], 0, {0})


def test_no_match() -> None:
    """0-a->1-a->0 cycle, 0-b->2 dead-end. S->aSb|ab. Expected {}."""
    _run_test("no match (dead-end b)    start=0",
              build_rsm({"S": [["a","S","b"], ["a","b"]]}, "S", {"a","b"}),
              3, [(0,"a",1),(1,"a",0),(0,"b",2)], 0, set())


def test_ambiguous_grammar() -> None:
    """S->SS|a|e. L={a^n | n>=0}. Chain 0-a->1-a->2-a->3. Expected {0,1,2,3}."""
    _run_test("ambiguous S->SS|a|e      start=0",
              build_rsm({"S": [["S","S"], ["a"], []]}, "S", {"a"}),
              4, [(0,"a",1),(1,"a",2),(2,"a",3)], 0, {0, 1, 2, 3})


def test_three_nonterminals() -> None:
    """S->AB, A->a, B->b. L={ab}. 0-a->1-b->2, 0-b->3. Expected {2}."""
    _run_test("three NTs S->AB,A->a,B->b start=0",
              build_rsm({"S": [["A","B"]], "A": [["a"]], "B": [["b"]]}, "S", {"a","b"}),
              4, [(0,"a",1),(1,"b",2),(0,"b",3)], 0, {2})


def test_cyclic_with_epsilon_base() -> None:
    """Cyclic graph. S->aSb|e. n=0 includes vertex 2 itself. Expected {1,2,3}."""
    _run_test("cyclic Dyck S->aSb|e     start=2",
              build_rsm({"S": [["a","S","b"], []]}, "S", {"a","b"}),
              4, _CYCLIC, 2, {1, 2, 3})


def test_cyclic_3_5_anbn_start1() -> None:
    """!!!
    !!! Same cyclic graph, but start=0.
    After k a-steps:  k≡0→0, k≡1→1, k≡2→2  (mod 3).
    Need to land on vertex 1 (k≡1 mod 3):  n=1→dest 3 (ab);  n=4→dest 1 (a⁴b⁴); …
    Expected: {1, 3}  (same reachable set, different witness paths)
    """
    rsm   = build_rsm({"S": [["a","S","b"], ["a","b"]]}, "S", {"a","b"})
    edges = [(0,"a",1),(1,"a",2),(2,"a",0),(1,"b",3),(3,"b",4),(4,"b",5),(5,"b",6),(6,"b",1)]
    _run_test("cyclic a^n b^n  3(a)-5(b)-loops start=0", rsm, 7, edges, 0, {1, 3, 4, 5, 6})    


def test_cyclic_3_5_anbn_CNF_start1() -> None:
    """!!!
    !!! Same cyclic graph, but start=0.
    After k a-steps:  k≡0→0, k≡1→1, k≡2→2  (mod 3).
    Need to land on vertex 1 (k≡1 mod 3):  n=1→dest 3 (ab);  n=4→dest 1 (a⁴b⁴); …
    Expected: {1, 3}  (same reachable set, different witness paths)
    """
    rsm   = build_rsm({"S": [["M","b"], ["a","b"]],"M": [["a","S"]]}, "S", {"a","b"})
    edges = [(0,"a",1),(1,"a",2),(2,"a",0),(1,"b",3),(3,"b",4),(4,"b",5),(5,"b",6),(6,"b",1)]
    _run_test("cyclic a^n b^n CNF 3(a)-5(b)-loops start=0", rsm, 7, edges, 0, {1, 3, 4, 5, 6}) 


def test_cyclic_reg_anbn_CNF_start0() -> None:
    """!!!
    !!! Same cyclic graph, but start=0.
    After k a-steps:  k≡0→0, k≡1→1, k≡2→2  (mod 3).
    Need to land on vertex 1 (k≡1 mod 3):  n=1→dest 3 (ab);  n=4→dest 1 (a⁴b⁴); …
    Expected: {1, 3}  (same reachable set, different witness paths)
    """
    rsm   = build_rsm({"S": [["M","b"], ["a","b"]],"M": [["a","S"]]}, "S", {"a","b"})
    edges = [(0,"a",0),(0,"b",1),(1,"b",1)]
    _run_test("cyclic reg a^n b^n CNF start=0", rsm, 2, edges, 0, {1}) 

def test_left_recursive_cycle() -> None:
    """Chain 0-a->1-a->2-a->0. S->Sa|e. L={a^n | n>=0}. Expected {0,1,2}."""
    _run_test("left-recursive  S->Sa|e start=0",
              build_rsm({"S": [["S","a"], []]}, "S", {"a"}),
              3, [(0,"a",1),(1,"a",2),(2,"a",0)], 0, {0, 1, 2})

def test_cyclic_2_3_anbncndn_start1() -> None:
    """!!!
    !!! Same cyclic graph, but start=0.
    After k a-steps:  k≡0→0, k≡1→1, k≡2→2  (mod 3).
    Need to land on vertex 1 (k≡1 mod 3):  n=1→dest 3 (ab);  n=4→dest 1 (a⁴b⁴); …
    Expected: {1, 3}  (same reachable set, different witness paths)
    """
    rsm   = build_rsm({"S": [["a","S","b"], ["a","b"],["c","S","d"],["c","d"]]}, "S", {"a","b","c","d"})
    edges = [(0,"a",1),(1,"a",2),(2,"a",0),(1,"d",3),(3,"d",4),(4,"d",0),(0,"b",5),(5,"b",0),(6,"c",0),(0,"c",6)]
    _run_test("cyclic a^n b^n c^n d^n  3(a)-5(b)-loops start=1", rsm, 7, edges, 1, {0, 5})

def test_cyclic_2_3_anbncndn_start6() -> None:
    """!!!
    !!! Same cyclic graph, but start=0.
    After k a-steps:  k≡0→0, k≡1→1, k≡2→2  (mod 3).
    Need to land on vertex 1 (k≡1 mod 3):  n=1→dest 3 (ab);  n=4→dest 1 (a⁴b⁴); …
    Expected: {1, 3}  (same reachable set, different witness paths)
    """
    rsm   = build_rsm({"S": [["a","S","b"], ["a","b"],["c","S","d"],["c","d"]]}, "S", {"a","b","c","d"})
    edges = [(0,"a",1),(1,"a",2),(2,"a",0),(0,"d",3),(3,"d",4),(4,"d",0),(0,"b",5),(5,"b",0),(6,"c",0),(0,"c",6)]
    _run_test("cyclic a^n b^n c^n d^n  3(a)-5(b)-loops start=6", rsm, 7, edges, 6, {0, 3, 4})

def test_cyclic_2_3_anbncndn_concat_start1() -> None:
    """!!!
    !!! Same cyclic graph, but start=0.
    After k a-steps:  k≡0→0, k≡1→1, k≡2→2  (mod 3).
    Need to land on vertex 1 (k≡1 mod 3):  n=1→dest 3 (ab);  n=4→dest 1 (a⁴b⁴); …
    Expected: {1, 3}  (same reachable set, different witness paths)
    """
    rsm   = build_rsm({"S": [["a","S","b"], ["a","b"],["c","S","d"],["c","d"],["S","S"]]}, "S", {"a","b","c","d"})
    edges = [(0,"a",1),(1,"a",2),(2,"a",0),(0,"d",3),(3,"d",4),(4,"d",0),(0,"b",5),(5,"b",0),(6,"c",0),(0,"c",6)]
    _run_test("cyclic a^n b^n c^n d^n concat  3(a)-5(b)-loops start=1", rsm, 7, edges, 1, {0,3,4, 5})

def test_cyclic_3_5_anbn_raw_RSM_start1() -> None:
    """!!!
    !!! Same cyclic graph, but start=0.
    After k a-steps:  k≡0→0, k≡1→1, k≡2→2  (mod 3).
    Need to land on vertex 1 (k≡1 mod 3):  n=1→dest 3 (ab);  n=4→dest 1 (a⁴b⁴); …
    Expected: {1, 3}  (same reachable set, different witness paths)
    """
    rsm   = build_rsm_raw({"S": (0,[0],[[0,"a",1], [1,"S",2],[2,"b",0]])}, "S", {"a","b"})
    edges = [(0,"a",1),(1,"a",2),(2,"a",0),(1,"b",3),(3,"b",4),(4,"b",5),(5,"b",6),(6,"b",1)]
    _run_test("cyclic a^n b^n  3(a)-5(b)-loops start=0", rsm, 7, edges, 0, {0, 1, 3, 4, 5, 6})    

def test_cyclic_2_3_anbncndn_concat_raw_rsm_start1() -> None:
    """!!!
    !!! Same cyclic graph, but start=0.
    After k a-steps:  k≡0→0, k≡1→1, k≡2→2  (mod 3).
    Need to land on vertex 1 (k≡1 mod 3):  n=1→dest 3 (ab);  n=4→dest 1 (a⁴b⁴); …
    Expected: {1, 3}  (same reachable set, different witness paths)
    """
    rsm   = build_rsm_raw({"S": (0,[0],[[0,"a",1], [1,"S",2],[2,"b",0],[0,"c",3], [3,"S",4],[4,"d",0]])}, "S", {"a","b","c","d"})
    edges = [(0,"a",1),(1,"a",2),(2,"a",0),(0,"d",3),(3,"d",4),(4,"d",0),(0,"b",5),(5,"b",0),(6,"c",0),(0,"c",6)]
    _run_test("cyclic a^n b^n c^n d^n concat  raw RSM 3(a)-5(b)-loops start=1", rsm, 7, edges, 1, {1,0,3,4, 5})


#  boxes: Dict[str, Tuple[int, List[int], List[Tuple[int,str,int]]]],
#      start_nt:    str,
#      terminals:   Set[str],

# ===============================================================================
# 7.  MAIN RUNNER
# ===============================================================================

if __name__ == "__main__":
    print()
    print("=" * 62)
    print("  LACFPQ blocked -- python-graphblas test suite")
    print("=" * 62)

    test_cyclic_anbn_start2()
    test_cyclic_anbn_start0()
    test_right_recursive()
    test_left_recursive()
    test_dyck()
    test_dyck_self_loop()
    test_mutual_recursion()
    test_epsilon_only()
    test_start_equals_final_with_cycle()
    test_no_match()
    test_ambiguous_grammar()
    test_three_nonterminals()
    test_cyclic_with_epsilon_base()
    test_cyclic_3_5_anbn_start1()
    test_cyclic_3_5_anbn_CNF_start1()
    test_cyclic_reg_anbn_CNF_start0()
    test_left_recursive_cycle()
    test_cyclic_2_3_anbncndn_start1()
    test_cyclic_2_3_anbncndn_start6()
    test_cyclic_2_3_anbncndn_concat_start1()
    test_cyclic_3_5_anbn_raw_RSM_start1()
    test_cyclic_2_3_anbncndn_concat_raw_rsm_start1()

    print("=" * 62)
    total = _pass_n[0] + _fail_n[0]
    print(f"  {_pass_n[0]}/{total} passed"
          + ("" if _fail_n[0] == 0 else f"  ({_fail_n[0]} FAILED)"))
    print("=" * 62)
    print()
