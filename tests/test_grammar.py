"""Unit tests for layer2.grammar — type system, tree ops, and genetic operators."""
import random

import pytest

from layer2.grammar import (
    ALL_FUNCTIONS,
    ADD,
    AND,
    CROSS_ABOVE,
    DIV,
    EMB_OPERATORS,
    EMB_TYPES,
    GT,
    IF_REAL,
    IF_SIDE,
    IN_REGIME,
    LAG,
    LT,
    MUL,
    NOT,
    OR,
    SQRT,
    SUB,
    FuncDef,
    FuncNode,
    Grammar,
    GType,
    Node,
    Regime,
    Side,
    TermDef,
    TermNode,
    _read_preregistration_grammar_counts,
    assert_grammar_matches_preregistration,
    build_terminal_set,
    deep_copy,
    depth,
    is_emblag_funcdef,
    node_count,
    to_str,
    validate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grammar():
    """Standard grammar with default terminal/function sets."""
    return Grammar(max_depth=6, max_nodes=25)


@pytest.fixture
def small_grammar():
    """Smaller grammar for deterministic tests."""
    terms = [
        TermDef("X", GType.REAL),
        TermDef("Y", GType.REAL),
        TermDef("EphReal", GType.REAL, sampler=lambda: round(random.uniform(-1, 1), 4)),
        TermDef("EphInt", GType.INT, sampler=lambda: random.choice((1, 3, 5))),
        TermDef("LOW_VOL", GType.REGIME, value=Regime.LOW_VOL),
        TermDef("CALL", GType.SIDE, value=Side.CALL),
        TermDef("PUT", GType.SIDE, value=Side.PUT),
        TermDef("NEUTRAL", GType.SIDE, value=Side.NEUTRAL),
    ]
    return Grammar(functions=ALL_FUNCTIONS, terminals=terms, max_depth=4, max_nodes=20)


# ---------------------------------------------------------------------------
# Test 1: Type safety — generated trees validate
# ---------------------------------------------------------------------------

def test_generated_trees_validate(grammar):
    """Trees from grow/full must pass type validation for all return types."""
    random.seed(42)
    for ret in (GType.REAL, GType.BOOL, GType.SIDE):
        for _ in range(20):
            tree = grammar.grow(ret, max_d=4)
            assert validate(tree), f"grow({ret}) produced invalid tree: {to_str(tree)}"
            tree = grammar.full(ret, max_d=3)
            assert validate(tree), f"full({ret}) produced invalid tree: {to_str(tree)}"


# ---------------------------------------------------------------------------
# Test 2: Crossover preserves types
# ---------------------------------------------------------------------------

def test_crossover_preserves_types(small_grammar):
    """Crossover of two valid trees produces valid trees of same root type."""
    random.seed(123)
    g = small_grammar
    for ret in (GType.REAL, GType.BOOL):
        pa = g.grow(ret, max_d=3)
        pb = g.grow(ret, max_d=3)
        c1, c2 = g.crossover(pa, pb)
        assert c1.ret_type == ret
        assert c2.ret_type == ret
        assert validate(c1), f"Crossover child 1 invalid: {to_str(c1)}"
        assert validate(c2), f"Crossover child 2 invalid: {to_str(c2)}"


# ---------------------------------------------------------------------------
# Test 3: Point mutation preserves types
# ---------------------------------------------------------------------------

def test_point_mutation_preserves_types(small_grammar):
    """Point mutation preserves tree validity and root return type."""
    random.seed(7)
    g = small_grammar
    for _ in range(30):
        tree = g.grow(GType.REAL, max_d=3)
        mutant = g.point_mutate(tree, p=0.5)
        assert mutant.ret_type == GType.REAL
        assert validate(mutant), f"Point mutant invalid: {to_str(mutant)}"


# ---------------------------------------------------------------------------
# Test 4: Subtree mutation preserves types
# ---------------------------------------------------------------------------

def test_subtree_mutation_preserves_types(small_grammar):
    """Subtree mutation preserves tree validity and root return type."""
    random.seed(11)
    g = small_grammar
    for ret in (GType.REAL, GType.BOOL):
        for _ in range(20):
            tree = g.grow(ret, max_d=4)
            mutant = g.subtree_mutate(tree, max_d=2)
            assert mutant.ret_type == ret
            assert validate(mutant), f"Subtree mutant invalid: {to_str(mutant)}"


# ---------------------------------------------------------------------------
# Test 5: Depth and node limits enforced
# ---------------------------------------------------------------------------

def test_depth_and_node_limits(grammar):
    """Crossover, subtree_mutate, and ramped_half_and_half respect limits."""
    random.seed(99)
    g = grammar
    pop = g.ramped_half_and_half(GType.REAL, pop_size=50)
    for tree in pop:
        assert depth(tree) <= g.max_depth, f"Tree depth {depth(tree)} > {g.max_depth}"
        assert node_count(tree) <= g.max_nodes, f"Node count {node_count(tree)} > {g.max_nodes}"

    # Crossover output respects limits
    for _ in range(30):
        pa, pb = random.sample(pop, 2)
        c1, c2 = g.crossover(pa, pb)
        assert depth(c1) <= g.max_depth
        assert depth(c2) <= g.max_depth
        assert node_count(c1) <= g.max_nodes
        assert node_count(c2) <= g.max_nodes


# ---------------------------------------------------------------------------
# Test 6: Bool type requires depth >= 1 (no Bool terminals)
# ---------------------------------------------------------------------------

def test_bool_requires_depth_ge_1(grammar):
    """GType.BOOL has no terminals, so min depth is 1 (needs a function node)."""
    g = grammar
    assert g._min_depth[GType.BOOL] >= 1
    # Direct generation at depth=0 would fail or produce function anyway
    tree = g.grow(GType.BOOL, max_d=1)
    assert isinstance(tree, FuncNode), "Bool at depth<=1 must be a FuncNode"
    assert tree.ret_type == GType.BOOL


# ---------------------------------------------------------------------------
# Test 7: Ephemeral constants are frozen after creation
# ---------------------------------------------------------------------------

def test_ephemeral_constants_frozen():
    """Once a TermNode is created, its value does not change on access."""
    eph_def = TermDef("EphReal", GType.REAL, sampler=lambda: round(random.uniform(-1, 1), 4))
    random.seed(42)
    node = TermNode(defn=eph_def)
    val1 = node.value
    val2 = node.value
    assert val1 == val2, "Ephemeral constant value changed after creation"
    # Creating a new node yields potentially different value
    random.seed(99)
    node2 = TermNode(defn=eph_def)
    # Values are frozen independently
    assert node.value == val1


# ---------------------------------------------------------------------------
# Test 8: to_str produces valid S-expression
# ---------------------------------------------------------------------------

def test_to_str_s_expression():
    """to_str produces readable S-expression for hand-built trees."""
    # GT(X, EphReal(0.5))
    x_term = TermNode(defn=TermDef("X", GType.REAL), value=None)
    eph_term = TermNode(defn=TermDef("EphReal", GType.REAL), value=0.5)
    gt_node = FuncNode(defn=GT, children=[x_term, eph_term])
    s = to_str(gt_node)
    assert s == "GT(X, EphReal(0.5))"

    # Nested: Add(X, Mul(Y, EphReal(-0.3)))
    y_term = TermNode(defn=TermDef("Y", GType.REAL), value=None)
    eph2 = TermNode(defn=TermDef("EphReal", GType.REAL), value=-0.3)
    mul_node = FuncNode(defn=MUL, children=[y_term, eph2])
    add_node = FuncNode(defn=ADD, children=[x_term, mul_node])
    s2 = to_str(add_node)
    assert s2 == "Add(X, Mul(Y, EphReal(-0.3)))"


# ---------------------------------------------------------------------------
# Test 9: ramped_half_and_half generates diverse trees
# ---------------------------------------------------------------------------

def test_ramped_half_and_half_diversity(grammar):
    """Population has diverse depths (not all identical)."""
    random.seed(13)
    pop = grammar.ramped_half_and_half(GType.REAL, pop_size=50)
    assert len(pop) == 50
    depths = {depth(t) for t in pop}
    # Should have at least 3 distinct depth values for good diversity
    assert len(depths) >= 3, f"Only {len(depths)} distinct depths: {depths}"


# ---------------------------------------------------------------------------
# Test 10: typed-vector operators replace retired EmbDot/EmbDist
# ---------------------------------------------------------------------------

def test_typed_vector_operators_cover_all_emb_types():
    """Typed-vector operators per each of 10 GTypes (7 group + 3 per-variate):
    v5:  4 operators (EmbNorm / EmbCos / EmbSub / EmbLag) × 7 types = 28.
    v8: +5 EmbProj_*_k operators (PC projections, k∈[0..4]) × 7 types = 35.
    v9: +84 EmbProj_*_k operators (per-group K capped 15) — replaces v8 35.
    R2: +3 per-variate types × 4 base ops = 12, + 3×5 EmbProj = 15.
    R2 total: 4×10 + 99 = 139 EMB_OPERATORS.
    """
    from layer2.grammar import PER_GROUP_K_GRAMMAR_V9
    expected_embproj = sum(PER_GROUP_K_GRAMMAR_V9.values())
    assert expected_embproj == 99, (
        f"PER_GROUP_K_GRAMMAR_V9 sums to {expected_embproj}, expected 99 "
        f"(84 group-pooled + 15 R2 per-variate)"
    )
    assert len(EMB_OPERATORS) == 4 * len(EMB_TYPES) + expected_embproj
    for emb_t in EMB_TYPES:
        n = emb_t.name
        # EmbNorm_{t}: (t,) → REAL
        norm = next(f for f in EMB_OPERATORS if f.name == f"EmbNorm_{n}")
        assert norm.arg_types == (emb_t,)
        assert norm.ret_type == GType.REAL
        # EmbCos_{t}: (t, t) → REAL — same-type constraint
        cos = next(f for f in EMB_OPERATORS if f.name == f"EmbCos_{n}")
        assert cos.arg_types == (emb_t, emb_t)
        assert cos.ret_type == GType.REAL
        # EmbSub_{t}: (t, t) → t — returns same type, enables chaining
        sub = next(f for f in EMB_OPERATORS if f.name == f"EmbSub_{n}")
        assert sub.arg_types == (emb_t, emb_t)
        assert sub.ret_type == emb_t
        # EmbLag_{t}: (t, INT) → t — lag lookup preserves type
        lag = next(f for f in EMB_OPERATORS if f.name == f"EmbLag_{n}")
        assert lag.arg_types == (emb_t, GType.INT)
        assert lag.ret_type == emb_t
        # v9: EmbProj_{t}_{k}: (t,) → REAL — per-group K PC projections
        k_for_group = PER_GROUP_K_GRAMMAR_V9[n]
        for k in range(k_for_group):
            proj = next(f for f in EMB_OPERATORS
                        if f.name == f"EmbProj_{n}_{k}")
            assert proj.arg_types == (emb_t,)
            assert proj.ret_type == GType.REAL
    # All 63 operators should be in ALL_FUNCTIONS
    for op in EMB_OPERATORS:
        assert op in ALL_FUNCTIONS


def test_typed_vector_tree_construction():
    """Typed-vector trees validate with correct ret_type propagation."""
    grid_term = TermNode(defn=TermDef("EMB_GRID", GType.EMB_GRID))
    # EmbNorm(grid) → REAL
    norm_op = next(f for f in EMB_OPERATORS if f.name == "EmbNorm_EMB_GRID")
    norm_tree = FuncNode(defn=norm_op, children=[grid_term])
    assert validate(norm_tree)
    assert norm_tree.ret_type == GType.REAL
    # EmbSub(grid, grid) → EMB_GRID (same-type chaining works)
    sub_op = next(f for f in EMB_OPERATORS if f.name == "EmbSub_EMB_GRID")
    sub_tree = FuncNode(defn=sub_op, children=[grid_term, grid_term])
    assert validate(sub_tree)
    assert sub_tree.ret_type == GType.EMB_GRID
    # EmbCos can chain onto EmbSub output (both EMB_GRID-typed)
    cos_op = next(f for f in EMB_OPERATORS if f.name == "EmbCos_EMB_GRID")
    cos_tree = FuncNode(defn=cos_op, children=[sub_tree, grid_term])
    assert validate(cos_tree)
    assert cos_tree.ret_type == GType.REAL


# ---------------------------------------------------------------------------
# Fix 1 tests: grammar-size drift guard
# ---------------------------------------------------------------------------

def test_grammar_size_matches_preregistration():
    """Live grammar counts must match the H2 pre-registration.
    v5: (46 functions, 29 terminals).
    v8: (81 functions, 29 terminals) — added 35 EmbProj_*_k functions
        for scalar PCA projections.
    v9: (130 functions, 29 terminals) — bumped EmbProj from flat K=5 (35
        ops) to per-group K capped at 15 (84 ops). 81 - 35 + 84 = 130.
    R2: (157 functions, 32 terminals) — added 3 per-variate types with
        4 base ops + 5 EmbProj each. 130 + 12 + 15 = 157; 29 + 3 = 32.
    Audit round: (157 functions, 36 terminals) — added SessionReturn,
        ATM_IV_5m, RealizedVol30m_5m, RawSpread_5m. 32 + 4 = 36.
    """
    # Should not raise under the current state.
    assert_grammar_matches_preregistration()
    # Sanity-check the underlying numbers are what the pre-reg commits to.
    assert len(ALL_FUNCTIONS) == 157
    # 53 live terminals (measured): the running total through the inter-day /
    # market-structure additions plus the 2026-06-03 UnrealizedProfitPct
    # position-aware exit terminal. Must equal the H2 YAML real-l1 grammar count
    # and the live build_terminal_set() (the drift guard compares all three).
    assert len(build_terminal_set()) == 53


def test_prereg_yaml_parses_grammar_size(tmp_path):
    """Regex-extraction of the (N functions, M terminals) tuple is robust
    against YAML whitespace / casing variations; tolerate reasonable drift
    in the human-readable description string without breaking the drift
    guard."""
    # (157, 53) after the 2026-06-03 UnrealizedProfitPct position-aware exit
    # terminal (and the inter-day / market-structure terminals added since 47).
    n_funcs, n_terms = _read_preregistration_grammar_counts()
    assert (n_funcs, n_terms) == (157, 53)

    # Synthetic YAML with whitespace variations — regex must tolerate them.
    fake_yaml = tmp_path / "fake_prereg.yaml"
    fake_yaml.write_text(
        "conditions:\n"
        "  real-l1:\n"
        "    grammar: full D84 typed-vector grammar  (  7  functions , 3 terminals  )\n"
        "  shuffled-l1:\n"
        "    grammar: identical to real-l1\n"
    )
    assert _read_preregistration_grammar_counts(fake_yaml) == (7, 3)


def test_prereg_yaml_missing_field_raises(tmp_path):
    """If the real-l1 grammar field disappears (YAML drift), the parser
    raises ValueError — not a silent 0/0 mismatch."""
    fake_yaml = tmp_path / "fake_prereg.yaml"
    fake_yaml.write_text(
        "conditions:\n"
        "  real-l1:\n"
        "    data: something\n"
    )
    with pytest.raises(ValueError, match="did not find conditions.real-l1.grammar"):
        _read_preregistration_grammar_counts(fake_yaml)


def test_assert_grammar_matches_raises_on_mismatch(tmp_path):
    """AssertionError (not ValueError) when the YAML advertises a different
    count from the live grammar."""
    fake_yaml = tmp_path / "fake_prereg.yaml"
    fake_yaml.write_text(
        "conditions:\n"
        "  real-l1:\n"
        "    grammar: synthetic (99 functions, 99 terminals)\n"
    )
    with pytest.raises(AssertionError, match="grammar drifted from H2 pre-registration"):
        assert_grammar_matches_preregistration(fake_yaml)


# ---------------------------------------------------------------------------
# Fix 2 tests: EmbLag first-arg-terminal constraint
# ---------------------------------------------------------------------------

def test_emblag_rejects_non_terminal_first_child():
    """An EmbLag_<T>(FuncNode, int) tree is invalid — the evaluator silently
    falls back to no-lag for a non-terminal first child, so the grammar
    must reject such trees at validation time."""
    grid_term = TermNode(defn=TermDef("EMB_GRID", GType.EMB_GRID))
    int_term  = TermNode(defn=TermDef("EphInt", GType.INT), value=5)
    sub_op = next(f for f in EMB_OPERATORS if f.name == "EmbSub_EMB_GRID")
    lag_op = next(f for f in EMB_OPERATORS if f.name == "EmbLag_EMB_GRID")
    # First-arg FuncNode → validate() must reject
    sub_tree = FuncNode(defn=sub_op, children=[grid_term, grid_term])
    bad = FuncNode(defn=lag_op, children=[sub_tree, int_term])
    assert not validate(bad), (
        "validate() accepted EmbLag with FuncNode first child — "
        "this allows the evaluator's defensive no-lag fallback to be hit silently"
    )
    # Sanity: the TermNode version is still valid.
    good = FuncNode(defn=lag_op, children=[grid_term, int_term])
    assert validate(good)


def test_emblag_predicate_identifies_operators():
    """is_emblag_funcdef flags exactly the 7 EmbLag_* operators — one per
    typed-vector GType — and nothing else."""
    emblag_ops = [f for f in EMB_OPERATORS if is_emblag_funcdef(f)]
    assert len(emblag_ops) == len(EMB_TYPES)
    for f in emblag_ops:
        assert f.name.startswith("EmbLag_"), f.name
    # Non-EmbLag EMB operators + scalar operators return False
    for f in EMB_OPERATORS:
        if not f.name.startswith("EmbLag_"):
            assert not is_emblag_funcdef(f), f.name
    assert not is_emblag_funcdef(ADD)
    assert not is_emblag_funcdef(LAG)  # scalar Lag is NOT EmbLag


def test_grow_never_produces_invalid_emblag_tree():
    """Random grow() calls must never produce an EmbLag_* node with a
    non-terminal first child. Walk a sampled population of trees and
    check the invariant holds throughout."""
    random.seed(4242)
    g = Grammar(max_depth=5, max_nodes=30)
    # Sample across EMB return types and REAL (where EmbLag appears via chain)
    bad_count = 0
    for ret in (GType.EMB_GRID, GType.EMB_SHARED, GType.EMB_SPX, GType.REAL):
        for _ in range(50):
            tree = g.grow(ret, max_d=5)
            if not validate(tree):
                bad_count += 1
    assert bad_count == 0, f"grow() produced {bad_count} invalid trees"


def test_subtree_mutate_preserves_emblag_invariant():
    """subtree_mutate targeting the first child of an EmbLag_* node must
    install a TermNode, not an arbitrary FuncNode — so the resulting
    tree stays grammar-valid."""
    random.seed(909)
    g = Grammar(max_depth=5, max_nodes=30)
    grid_term = TermNode(defn=TermDef("EMB_GRID", GType.EMB_GRID))
    int_term  = TermNode(defn=TermDef("EphInt", GType.INT), value=3)
    lag_op = next(f for f in EMB_OPERATORS if f.name == "EmbLag_EMB_GRID")
    # Root is EmbLag_EMB_GRID(EMB_GRID, 3) — any non-root subtree mutation
    # either lands on the first child (forced TermNode) or the int child
    # (IF_INT subtree mutation); neither should produce an invalid tree.
    tree = FuncNode(defn=lag_op, children=[grid_term, int_term])
    for _ in range(30):
        mutant = g.subtree_mutate(tree, max_d=3)
        assert validate(mutant), (
            f"subtree_mutate produced invalid EmbLag tree: {to_str(mutant)}"
        )
