import copy
import numpy as np
import torch
import itertools
import operator
import re
import math
import multiprocessing as mp
from itertools import repeat
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
from functools import reduce

from .stl import Atom, Not, Eventually, Globally, Until, And, Or, Boolean
from .formula_utils import from_string_to_formula
from ..kernels.base_measure import BaseMeasure
from ..utils import dump_pickle, chunks, save_file


def time_scaling(phis, points, phi_timespan=101):
    # Rescale the time intervals in the given phis formulae
    current_one_percent = (
        points / phi_timespan
    )  # Calculate the current one percent in terms of the number of points
    phis_scaled = []  # to store the scaled phis

    for phi in phis:  # iterate through each formula
        phi_str = str(phi)  # convert to string
        # Find the indices of the temporal intervals in the string
        temporal_start_idx = [
            i for i in range(len(phi_str)) if phi_str.startswith("[", i)
        ]
        temporal_middle_idx = [
            i for i in range(len(phi_str)) if phi_str.startswith(",", i)
        ]
        temporal_end_idx = [
            i for i in range(len(phi_str)) if phi_str.startswith("]", i)
        ]
        # print(f'{temporal_start_idx=}')
        # print(f'{temporal_middle_idx=}')
        # print(f'{temporal_end_idx=}')
        # Get the start index of the first interval, if any
        start_idx = temporal_start_idx[0] if len(temporal_start_idx) > 0 else None
        str_list = [
            phi_str[:start_idx]
        ]  # Initialize a list to store parts of the new formula
        new_intervals_list = []  # Initialize a list to store the new time intervals
        rights = 0
        # Iterate through the indices of the temporal intervals
        for i, s, m, e in zip(
            range(len(temporal_start_idx)),
            temporal_start_idx,
            temporal_middle_idx,
            temporal_end_idx,
        ):
            right_unbound = (
                True if phi_str[e - 1] == "f" else False
            )  # Check if the interval is right unbounded
            right_bound = (
                -1.0 if right_unbound else float(phi_str[m + 1 : e])
            )  # Determine the right bound of the interval
            current_time_interval = [
                float(phi_str[s + 1 : m]),
                right_bound,
            ]  # this is the original interval
            # these are the changes I was doing (so this is the main part that should be changed)
            current_percentage = (
                0
                if right_unbound
                else current_time_interval[1] - current_time_interval[0]
            )  # current percentage of the time interval
            new_left = math.floor(
                current_time_interval[0] * current_one_percent
            )  # Scale the left bound of the interval
            # Scale the entire interval and ensure it does not exceed the number of points
            
            # Ensure end is at most points and greater than start
            new_right = new_left + max(math.floor(current_percentage * current_one_percent), 1)
            new_right = min(new_right, points)
            rights += new_right
            
            if rights > points:
                diff = rights - points
                new_right = new_right - diff
                
            new_time_interval = [
                new_left,
                new_right
            ]
            new_right_str = (
                "inf" if right_unbound else str(new_time_interval[1])
            )  # Determine the right bound as a string
            # from now on it is changing the formula parameters
            new_intervals_list += [
                "[" + str(new_time_interval[0]) + "," + new_right_str + "]"
            ]  # Construct the new interval string and add it to the list
            # Extract the part of the formula after the current interval
            idx = (
                temporal_start_idx[i + 1]
                if i < len(temporal_start_idx) - 1
                else None
            )
            str_list.append(phi_str[e + 1 : idx])

        # Combine the parts of the formula and the new intervals to form the new formula string
        new_phi_str = ""
        for i in range(len(new_intervals_list)):
            new_phi_str += str_list[i]
            new_phi_str += new_intervals_list[i]
        new_phi_str += str_list[-1]
        # Convert the new formula string back to a formula object and add it to the list
        phis_scaled.append(from_string_to_formula(new_phi_str))

    return phis_scaled

    
def rhos_disjunction(precomputed_rhos, subset):
    z1 = precomputed_rhos[subset[0]]
    for i in range(0, len(subset) - 1):
        # Or.quantitative, vectorize = False
        z2 = precomputed_rhos[subset[i + 1]]
        size: int = min(z1.size()[2], z2.size()[2])
        z1 = z1[:, :, :size]
        z2 = z2[:, :, :size]
        z1 = torch.max(z1, z2)  # new z1
    return torch.reshape(z1[:, 0, 0], (-1,))


def rhos_conjunction(z1, z2):
    size: int = min(z1.size()[2], z2.size()[2])
    z1_ = z1[:, :, :size]
    z2_ = z2[:, :, :size]
    z = torch.min(z1_, z2_)  # new z1
    return z  # torch.reshape(z[:, 0, 0], (-1,))


def disjunction(formulae):
    # Base case: if the list has only one formula, return it
    if len(formulae) == 1:
        return formulae[0]
    # Defensive: print formulae if something goes wrong
    try:
        return Or(formulae[0], disjunction(formulae[1:]))
    except Exception as e:
        print("Error in disjunction, formulae:", formulae)
        raise e


def conjunction(formulae):
    # Base case: if the list has only one formula, return it
    if len(formulae) == 1:
        return formulae[0]
    # Recursive case: Or the first formula with the result of disjunction of the rest
    else:
        return And(formulae[0], conjunction(formulae[1:]))

MAX_DEPTH = 100
def simplify(formula, depth = 0):
    """
    Recursively simplifies STL formulae using logical equivalences and temporal rules.
    """
    if depth > MAX_DEPTH:
        return formula
    try: node = copy.deepcopy(formula)
    except: return formula
    # --- Recursively simplify children first ---
    if hasattr(node, "child"):
        node.child = simplify(node.child, depth + 1)
    elif hasattr(node, "left_child") and hasattr(node, "right_child"):
        node.left_child = simplify(node.left_child, depth + 1)
        node.right_child = simplify(node.right_child, depth + 1)

    # --- NOT operator simplifications ---
    if isinstance(node, Not):
        child = node.child
        if isinstance(child, Not):
            return simplify(child.child, depth + 1)  # Double negation: ¬(¬φ) → φ
        elif isinstance(child, Atom):
            return Atom(  # Negated predicate
                var_index=child.var_index,
                threshold=copy.deepcopy(child.threshold),
                lte=not child.lte,
            )
        elif isinstance(child, Eventually):
            return simplify(
                Globally(  # ¬F_I(φ) → G_I(¬φ)
                    unbound=child.unbound,
                    right_unbound=child.right_unbound,
                    left_time_bound=child.left_time_bound,
                    right_time_bound=child.right_time_bound,
                    child=Not(child.child),
                ), depth + 1
            )
        elif isinstance(child, Globally):
            return simplify(
                Eventually(  # ¬G_I(φ) → F_I(¬φ)
                    unbound=child.unbound,
                    right_unbound=child.right_unbound,
                    left_time_bound=child.left_time_bound,
                    right_time_bound=child.right_time_bound,
                    child=Not(child.child),
                ), depth + 1
            )
        elif isinstance(child, Or):
            # ¬(a ∨ b) → ¬a ∧ ¬b
            return simplify(
                And(
                    left_child=simplify(Not(child.left_child), depth + 1),
                    right_child=simplify(Not(child.right_child), depth + 1),
                ), depth + 1
            )
        elif isinstance(child, And):
            # ¬(a ∧ b) → ¬a ∨ ¬b
            return simplify(
                Or(
                    left_child=simplify(Not(child.left_child), depth + 1),
                    right_child=simplify(Not(child.right_child), depth + 1),
                ), depth + 1
            )

    # --- AND simplifications ---
    if isinstance(node, And):
        l, r = node.left_child, node.right_child
        if l == r:
            return simplify(l, depth + 1)  # a ∧ a → a

        # a ∧ (a ∨ b) → a
        if isinstance(r, Or) and (l == r.left_child or l == r.right_child):
            return simplify(l, depth + 1)
        if isinstance(l, Or) and (r == l.left_child or r == l.right_child):
            return simplify(r, depth + 1)

        # a ∧ (a ∧ b) → a ∧ b
        if isinstance(r, And):
            if l == r.left_child:
                return simplify(And(left_child=l, right_child=r.right_child), depth + 1)
            if l == r.right_child:
                return simplify(And(left_child=l, right_child=r.left_child), depth + 1)
        if isinstance(l, And):
            if r == l.left_child:
                return simplify(And(left_child=r, right_child=l.right_child), depth + 1)
            if r == l.right_child:
                return simplify(And(left_child=r, right_child=l.left_child), depth + 1)

    # --- OR simplifications ---
    if isinstance(node, Or):
        l, r = node.left_child, node.right_child
        if l == r:
            return simplify(l, depth + 1)  # a ∨ a → a

        # a ∨ (a ∧ b) → a
        if isinstance(r, And) and (l == r.left_child or l == r.right_child):
            return simplify(l, depth + 1)
        if isinstance(l, And) and (r == l.left_child or r == l.right_child):
            return simplify(r, depth + 1)

        # a ∨ (a ∨ b) → a ∨ b
        if isinstance(r, Or):
            if l == r.left_child:
                return simplify(Or(left_child=l, right_child=r.right_child), depth + 1)
            if l == r.right_child:
                return simplify(Or(left_child=l, right_child=r.left_child), depth + 1)
        if isinstance(l, Or):
            if r == l.left_child:
                return simplify(Or(left_child=r, right_child=l.right_child), depth + 1)
            if r == l.right_child:
                return simplify(Or(left_child=r, right_child=l.left_child), depth + 1)

    # --- Nested Globally simplification: G_I(G_J(φ)) → G_{I+J}(φ) ---
    if isinstance(node, Globally) and isinstance(node.child, Globally):
        inner = node.child
        child = inner.child
        return simplify(
            Globally(
                unbound=node.unbound or inner.unbound,
                right_unbound=node.right_unbound or inner.right_unbound,
                left_time_bound=node.left_time_bound + inner.left_time_bound,
                right_time_bound=(
                    1
                    if (node.right_unbound or inner.right_unbound)
                    else node.right_time_bound + inner.right_time_bound - 1
                ),
                child=copy.deepcopy(child),
            ), depth + 1
        )

    # --- Nested Eventually simplification: F_I(F_J(φ)) → F_{I+J}(φ) ---
    if isinstance(node, Eventually) and isinstance(node.child, Eventually):
        inner = node.child
        child = inner.child
        return simplify(
            Eventually(
                unbound=node.unbound or inner.unbound,
                right_unbound=node.right_unbound or inner.right_unbound,
                left_time_bound=node.left_time_bound + inner.left_time_bound,
                right_time_bound=(
                    1
                    if (node.right_unbound or inner.right_unbound)
                    else node.right_time_bound + inner.right_time_bound - 1
                ),
                child=copy.deepcopy(child),
            ), depth + 1
        )

    # --- Until simplification: φ U φ → φ ---
    if isinstance(node, Until) and node.left_child == node.right_child:
        return simplify(node.left_child, depth + 1)

    return node


def flatten_phi(node, operator):
    """Flatten nested OR/AND trees and return a simplified STL formula."""
    if isinstance(node, operator):
        # Recursively flatten children
        left = flatten_phi(node.left_child, operator)
        right = flatten_phi(node.right_child, operator)

        # Collect all disjuncts
        disjuncts = []

        def collect_disjuncts(n):
            if isinstance(n, operator):
                collect_disjuncts(n.left_child)
                collect_disjuncts(n.right_child)
            else:
                disjuncts.append(n)

        collect_disjuncts(left)
        collect_disjuncts(right)

        # Remove duplicates if desired (optional)
        # disjuncts = list(dict.fromkeys(disjuncts))  # requires Atom/Eventually/.. to be hashable

        # Rebuild balanced OR-tree
        from functools import reduce

        return reduce(lambda a, b: operator(a, b), disjuncts)

    # Recursively process children for other types
    if isinstance(node, And):
        return And(
            flatten_phi(node.left_child, operator),
            flatten_phi(node.right_child, operator),
        )
    if isinstance(node, Not):
        return Not(flatten_phi(node.child, operator))
    if isinstance(node, Eventually):
        return Eventually(
            child=flatten_phi(node.child, operator),
            left_time_bound=node.left_time_bound,
            right_time_bound=node.right_time_bound,
            unbound=node.unbound,
            right_unbound=node.right_unbound,
        )
    if isinstance(node, Globally):
        return Globally(
            child=flatten_phi(node.child, operator),
            left_time_bound=node.left_time_bound,
            right_time_bound=node.right_time_bound,
            unbound=node.unbound,
            right_unbound=node.right_unbound,
        )
    if isinstance(node, Until):
        return Until(
            left_child=flatten_phi(node.left_child, operator),
            right_child=flatten_phi(node.right_child, operator),
            left_time_bound=node.left_time_bound,
            right_time_bound=node.right_time_bound,
            unbound=node.unbound,
            right_unbound=node.right_unbound,
        )

    # Atoms, Booleans, etc. are returned as-is
    return node


def simplify_by_robustness(node, robustness_map, eps=1e-5):
    if not isinstance(node, (Or, And)):
            return node
        
    left = simplify_by_robustness(node.left_child, robustness_map, eps)
    right = simplify_by_robustness(node.right_child, robustness_map, eps)
    
    subformulae = []
    def collect_subformulae(n):
        if isinstance(n, type(node)):
            collect_subformulae(n.left_child)
            collect_subformulae(n.right_child)
        else:
            subformulae.append(n)
            
    collect_subformulae(left)
    collect_subformulae(right)

    num_traj = len(next(iter(robustness_map.values())))  # Infer trajectory count
    ref_rob = [-float("inf")] * num_traj if isinstance(node, Or) else [float("inf")] * num_traj

    for f in subformulae:
        rob = robustness_map.get(str(f), None)
        if rob is None:
            continue  # Skip if no robustness data (or use useful.append(f) to keep)
        for i in range(num_traj):
            if isinstance(node, Or):
                if rob[i] > ref_rob[i]:
                    ref_rob[i] = rob[i]
            else:  # AND
                if rob[i] < ref_rob[i]:
                    ref_rob[i] = rob[i]

    useful = []
    for f in subformulae:
        rob = robustness_map.get(str(f), None)
        if rob is None:
            useful.append(f)  # Keep if no robustness data
            continue
        # Only keep if both close and sign matches, and skip NaN/Inf/zero
        def is_contributing(i):
            r = rob[i].detach()
            rr = ref_rob[i].detach()
            # Skip if either is nan or inf
            if not np.isfinite(r) or not np.isfinite(rr):
                return False
            # Skip if both are zero
            if r == 0 and rr == 0:
                return False
            # Check closeness and sign
            return abs(r - rr) < eps and (np.sign(r) == np.sign(rr))
        contributes = any(is_contributing(i) for i in range(num_traj))
        if contributes:
            useful.append(f)
        # Debug print for tracing
        # print(f"subformula: {str(f)}\nrob: {rob}\nref_rob: {ref_rob}\ncontributes: {contributes}")

    if not useful:
        return Boolean(False if isinstance(node, Or) else True)
    return reduce(lambda a, b: type(node)(a, b), useful) if len(useful) > 1 else useful[0]


def evaluate_and_simplify(formula, x, operator: str = None):
    # x: trajectories to evaluate
    try: simple = simplify(formula)
    except : simple = formula
    # if operator == 'or': operator = Or
    # elif operator == 'and': operator = And
    # else: raise ValueError('Operator should be either "and" or "or".')
    atoms = extract_all_atoms(str(simple))
    truth_map = get_atom_truth_over_time(x, atoms)
    
    try: simple = simplify(rimplify(simple, truth_map, x))
    except: simple = rimplify(simple, truth_map, x)
    # try: flat = flatten_phi(simplify(simple), operator)
    # except: flat = flatten_phi(simple, operator)
    # robustness_map = get_robustness_map(flat, x, operator)
    def get_all_subformulas(n):
        subformulas = set()
        if isinstance(n, (And, Or)):
            subformulas |= get_all_subformulas(n.left_child)
            subformulas |= get_all_subformulas(n.right_child)
        subformulas.add(str(n))
        return subformulas

    robustness_map = {
        f_str: from_string_to_formula(f_str).quantitative(x).squeeze()
        for f_str in get_all_subformulas(simple)
    }
    out =  simplify_by_robustness(simple, robustness_map)
    return out


def extract_all_atoms(formula_str: str) -> set[str]:
    """
    Extracts all unique atomic conditions from an STL formula string.

    An atom is defined as a string like: 'x_i <= threshold' or 'x_i >= threshold'.
    """
    pattern = r"(x_\d+)\s*(<=|>=)\s*(-?\d+\.\d+)"
    matches = re.findall(pattern, formula_str)
    return [f"{var} {op} {val}" for var, op, val in matches]


def get_atom_truth_over_time(
    trajectories, atom_strs: list[str]
) -> dict[str, list[bool]]:
    _, _, time_len = trajectories.shape
    result = {}

    atom_pattern = re.compile(r"x_(\d+)\s*(<=|>=)\s*(-?\d+\.\d+)")

    for atom in atom_strs:
        match = atom_pattern.fullmatch(atom.strip())
        if not match:
            raise ValueError(f"Invalid atom format: {atom}")
        var_index = int(match[1])
        op = match[2]
        threshold = float(match[3])

        truth_vector = []
        for t in range(time_len):
            values_at_t = trajectories[:, var_index, t]
            alltrue_at_t = torch.all(
                values_at_t >= threshold if op == ">=" else values_at_t <= threshold
            ).item()
            allfalse_at_t = not torch.any(
                values_at_t >= threshold if op == ">=" else values_at_t <= threshold
            ).item()
            out = True if alltrue_at_t else False if allfalse_at_t else torch.nan
            truth_vector.append(out)

        result[atom] = truth_vector

    return result


def rimplify(node, truth_map, x, prevent_final_boolean=True, _depth=0):
    """
    Simplifies an STL formula, optionally preventing the root node from collapsing to a Boolean.
    """
    time_len = len(truth_map[list(truth_map.keys())[0]])

    def _simplify(n, depth):
        if isinstance(n, Atom):
            key = str(n)
            truth = truth_map.get(key)
            if truth is None:
                return n
            if all(v is True for v in truth):
                return Boolean(True)
            elif all(v is False for v in truth):
                return Boolean(False)
            return n

        if isinstance(n, Not):
            child = _simplify(n.child, depth + 1)
            if isinstance(child, Boolean):
                return Boolean(not child.value)
            return Not(child)

        if isinstance(n, And):
            l = _simplify(n.left_child, depth + 1)
            r = _simplify(n.right_child, depth + 1)
            if isinstance(l, Boolean) and isinstance(r, Boolean):
                return Boolean(l.value and r.value)
            if isinstance(l, Boolean):
                return r if l.value else Boolean(False)
            if isinstance(r, Boolean):
                return l if r.value else Boolean(False)
            return And(l, r)

        if isinstance(n, Or):
            l = _simplify(n.left_child, depth + 1)
            r = _simplify(n.right_child, depth + 1)
            if isinstance(l, Boolean) and isinstance(r, Boolean):
                return Boolean(l.value or r.value)
            if isinstance(l, Boolean):
                return r if not l.value else Boolean(True)
            if isinstance(r, Boolean):
                return l if not r.value else Boolean(True)
            return Or(l, r)

        if isinstance(n, (Globally, Eventually)) and n.unbound:
            child = _simplify(n.child, depth + 1)
            if isinstance(child, Boolean):
                return child
            # start = n.left_time_bound
            # end = n.right_time_bound if not n.right_unbound else time_len - 1
            child_truth = evaluate_truth(child, truth_map)
            if child_truth is None:
                return type(n)(
                    child=child,
                    left_time_bound=n.left_time_bound,
                    right_time_bound=n.right_time_bound,
                    unbound=n.unbound,
                    right_unbound=n.right_unbound,
                )
            # interval = child_truth[start : end + 1]
            if all(v is True for v in child_truth):
                return Boolean(True)
            if all(v is False for v in child_truth):
                return Boolean(False)
            return type(n)(
                child=child,
                left_time_bound=n.left_time_bound,
                right_time_bound=n.right_time_bound,
                unbound=n.unbound,
                right_unbound=n.right_unbound,
            )

        if isinstance(n, Until):
            l = _simplify(n.left_child, depth + 1)
            r = _simplify(n.right_child, depth + 1)

            if isinstance(l, Boolean) and isinstance(r, Boolean):
                return Boolean(r.value or (l.value and r.value))
            if isinstance(r, Boolean):
                if r.value:
                    if n.unbound or n.left_time_bound == 0: # Globally[0,0]
                        return l
                    return Globally(
                        l,
                        left_time_bound=0,
                        right_time_bound=n.left_time_bound,
                        unbound=n.unbound,
                    )
                else:
                    return Boolean(False)
            if isinstance(l, Boolean):
                if l.value:
                    return Eventually(
                        r,
                        left_time_bound=n.left_time_bound,
                        right_time_bound=n.right_time_bound,
                        unbound=n.unbound,
                        right_unbound=n.right_unbound,
                    )
                else:
                    return Boolean(False)
            return Until(
                left_child=l,
                right_child=r,
                left_time_bound=n.left_time_bound,
                right_time_bound=n.right_time_bound,
                unbound=n.unbound,
                right_unbound=n.right_unbound,
            )

        return n  # default fallback

    # Run simplification
    result = _simplify(node, _depth)

    # If top-level and the result is Boolean, avoid final simplification
    if prevent_final_boolean and isinstance(result, Boolean):
        # print('would be boolean: ', node)
        return node  # return the original unsimplified root
    
    # #! safety exits
    # se non mantiene i segni
    formula_rho = node.quantitative(x)
    result_rho = result.quantitative(x)
    if not torch.all(torch.sign(formula_rho) == torch.sign(result_rho)):
        # print('safety exit: ', node)
        return node  # return the original unsimplified root

    return result


def evaluate_truth(node, truth_map):
    if isinstance(node, Atom):
        return truth_map.get(str(node), None)
    if isinstance(node, Boolean):
        return torch.full_like(next(iter(truth_map.values())), node.value)
    return None  # too complex, skip


def contains_boolean(node):
    """
    Recursively checks if a formula (STL AST node) contains a Boolean value.

    Args:
        node: STL formula node.

    Returns:
        True if any subnode is a Boolean, False otherwise.
    """
    if isinstance(node, Boolean):
        return True
    # Check children recursively depending on node type
    if hasattr(node, "child"):
        return contains_boolean(node.child)
    if hasattr(node, "left_child") and hasattr(node, "right_child"):
        return contains_boolean(node.left_child) or contains_boolean(node.right_child)
    return False

MAX_DEPTH = 100
def rescale_var_thresholds(formula, val, depth = 0):
    # Function to rescale the threshold values of formulae by a given value
    # Deep copy the formula to prevent modification of the original
    if depth > MAX_DEPTH:
        return formula
    try: current_node = copy.deepcopy(formula)
    except: return formula
    # Traverse the formula recursively
    if type(current_node) is not Atom:
        if type(current_node) is Not:
            child = rescale_var_thresholds(current_node.child, -val, depth + 1)
            current_node.child = copy.deepcopy(child)
        if (
            (type(current_node) is And)
            or (type(current_node) is Or)
            or (type(current_node) is Until)
        ):
            # Process left child recursively
            left_child = rescale_var_thresholds(current_node.left_child, val, depth + 1)
            current_node.left_child = copy.deepcopy(left_child)
        else:
            if (type(current_node) is Eventually) or (
                type(current_node) is Globally
            ):
                child = rescale_var_thresholds(current_node.child, val, depth + 1)
                current_node.child = copy.deepcopy(child)
        if (
            (type(current_node) is And)
            or (type(current_node) is Or)
            or (type(current_node) is Until)
        ):
            # Process right child recursively
            right_child = rescale_var_thresholds(current_node.right_child, val, depth + 1)
            current_node.right_child = copy.deepcopy(right_child)
    else:
        # Treat `<=` and `>=` differently
        if current_node.lte:  # If the Atom uses `<=`
            current_node.threshold = current_node.threshold + val
        else:  # If the Atom uses `>=`
            current_node.threshold = current_node.threshold - val
    return current_node


def _inverse_normalize_phis(formula, mean, std):
    # Function to inverse normalize the thresholds of formulae based on mean and standard deviation
    # Deep copy the formula to prevent modification of the original
    current_node = copy.deepcopy(formula)
    # Traverse the formula recursively
    if type(current_node) is not Atom:
        if (
            (type(current_node) is And)
            or (type(current_node) is Or)
            or (type(current_node) is Until)
        ):
            # Process left child recursively
            left_child = _inverse_normalize_phis(current_node.left_child, mean, std)
            current_node.left_child = copy.deepcopy(left_child)
        else:
            if (
                (type(current_node) is Eventually)
                or (type(current_node) is Globally)
                or (type(current_node) is Not)
            ):
                # Process right child recursively
                child = _inverse_normalize_phis(current_node.child, mean, std)
                current_node.child = copy.deepcopy(child)
        if (
            (type(current_node) is And)
            or (type(current_node) is Or)
            or (type(current_node) is Until)
        ):
            right_child = _inverse_normalize_phis(current_node.right_child, mean, std)
            current_node.right_child = copy.deepcopy(right_child)
    else:
        # Update threshold using inverse normalization
        current_node.threshold = (
            current_node.threshold * std[current_node.var_index]
        ) + mean[current_node.var_index]
    return current_node


def inverse_normalize_phis(mean, std, phis):
    # Function to inverse normalize a list of formulae based on mean and standard deviation
    # Get mean and standard deviation for normalization
    # mean, std = get_mean_std(dataname)
    phis_unnorm = []
    phis = phis if type(phis) is list else [phis]  # also a single formula can be passed
    for phi in phis:  # Inverse normalize each formula in the list
        result = _inverse_normalize_phis(phi, mean, std)
        phis_unnorm.append(result)
    return phis_unnorm


def set_temporal_bounds(node, time_bounds):
    # set temporal bounds of a node in a formula
    if time_bounds[0] == -1:
        node.unbound = True
        node.left_time_bound = 0
        node.right_time_bound = 1
    elif time_bounds[1] == -1:
        node.unbound = False
        node.left_time_bound = time_bounds[0]
        node.right_unbound = True
    else:
        node.unbound = False
        node.right_unbound = False
        node.left_time_bound = time_bounds[0]
        node.right_time_bound = time_bounds[1]
    return copy.deepcopy(node)


def traverse_formula(formula, task="var_thresh", prop_list=None):
    # DFS (depth-first search) traversal of the syntax tree of the formula
    # task = 'var_thresh' --> set variable thresholds of the formula
    # task = 'temp thresh --> set temporal thresholds of the temporal operators of the formula
    current_node = formula
    if type(current_node) is not Atom:
        if (type(current_node) is And) or (type(current_node) is Or):
            prop_list, left_child = traverse_formula(
                current_node.left_child, task, prop_list
            )
            current_node.left_child = copy.deepcopy(left_child)
        else:
            if (type(current_node) is Eventually) or (
                type(current_node) is Globally
            ):
                if task in ["temp_thresh", "all_thresh"]:
                    assert prop_list is not None
                    if type(prop_list) is not list:
                        prop_list = list(prop_list)
                    time_bounds = prop_list.pop(0)
                    current_node = set_temporal_bounds(current_node, time_bounds)
            if type(current_node) is not Until:
                prop_list, child = traverse_formula(current_node.child, task, prop_list)
                current_node.child = copy.deepcopy(child)
        if (type(current_node) is And) or (type(current_node) is Or):
            if (type(current_node) is Until) and (
                task in ["temp_thresh", "all_thresh"]
            ):
                time_bounds = prop_list.pop(0)
                current_node = set_temporal_bounds(current_node, time_bounds)
            prop_list, right_child = traverse_formula(
                current_node.right_child, task, prop_list
            )
            current_node.right_child = copy.deepcopy(right_child)
        if type(current_node) is Until:
            prop_list, left_child = traverse_formula(
                current_node.left_child, task, prop_list
            )
            current_node.left_child = copy.deepcopy(left_child)
            if task in ["temp_thresh", "all_thresh"]:
                assert prop_list is not None
                if type(prop_list) is not list:
                    prop_list = list(prop_list)
                time_bounds = prop_list.pop(0)
                current_node = set_temporal_bounds(current_node, time_bounds)
            prop_list, right_child = traverse_formula(
                current_node.right_child, task, prop_list
            )
            current_node.right_child = copy.deepcopy(right_child)
    else:
        if task == "var_index":
            current_node.var_index = prop_list.pop(0)
        elif task == "var_sign":
            current_node.lte = prop_list.pop(0) > 0
        elif task in ["var_thresh", "all_thresh"]:
            current_node.threshold = round(prop_list.pop(0), 4)
        elif task == "get_var_idx":
            if prop_list is None:
                prop_list = []
            prop_list.append(current_node.var_index)
    return prop_list, current_node


def find_all_paths(formula, part, tot):
    if type(formula) is Atom:
        tot.append(part[:] + [0])
        return
    elif (type(formula) is Globally) or (type(formula) is Eventually):
        part.append(1)
        left_child = formula.child
    else:
        if type(formula) is Until:
            part.append(1)
        else:
            part.append(0)
        left_child = (
            formula.left_child if type(formula) is not Not else formula.child
        )
    find_all_paths(left_child, part, tot)
    if (
        (type(formula) is Until)
        or (type(formula) is And)
        or (type(formula) is Or)
    ):
        right_child = formula.right_child
        find_all_paths(right_child, part, tot)
    part.pop()


def max_nested_temp(formula, max_n_until):
    n_until_f = str(formula).count("until")
    if n_until_f > max_n_until:
        return np.inf
    all_paths = []
    find_all_paths(formula, [], all_paths)
    sum_temp = list(map(sum, all_paths))
    return max(sum_temp)


def get_number_leaves(phi, idx=True):
    # given a formula, find the number of variables (i.e. leaves of the syntax tree)
    phi_str = str(phi)
    if idx:
        phi_split = phi_str.split()
        phi_var = [sub for sub in phi_split if sub.startswith("x_")]
        var_idx = [int(sub[2:]) for sub in phi_var]
        return len(phi_var), var_idx
    return phi_str.count("x_")


def get_temp(phi):
    # needed when one only wants to set temporal thresholds
    phi_str = str(phi)
    phi_split = phi_str.split()
    phi_temp = [sub for sub in phi_split if sub[:2] in ["ev", "al", "un"]]
    return len(phi_temp), phi_temp


def get_thresh_nodes(phi, max_nvars=3):
    # needed when one wants to set both var and temporal thresholds
    phi_str = str(phi)
    phi_split = phi_str.split()
    phi_thresh = list(filter(lambda x: x[:2] in ["x_", "ev", "al", "un"], phi_split))
    vars_str = [str(i) for i in range(max_nvars)]

    def set_type_list(p):
        return int(p[-1]) if p[-1] in vars_str else p

    phi_type = list(map(set_type_list, phi_thresh))
    return len(phi_thresh), phi_type


def get_var_thresh_list(var_idx, var_domain, n_var_thresh, linspace=False):
    if linspace:
        return torch.linspace(
            var_domain[var_idx][0], var_domain[var_idx][1], n_var_thresh
        ).tolist()
    return np.sort(
        np.random.normal(
            np.mean(var_domain[var_idx]),
            np.std(var_domain[var_idx]),
            size=(n_var_thresh,),
        )
    ).tolist()
    # return np.sort(np.random.uniform(low=var_domain[var_idx][0], high=var_domain[var_idx][1], size=(n_var_thresh,))).tolist()


def get_temp_thresh_list(time_domain, n_time_thresh, linspace=False):
    if linspace:
        bound_thresh = (
            torch.linspace(time_domain[0], time_domain[1], n_time_thresh).int().tolist()
        )
    else:
        bound_thresh = np.sort(
            random.sample(range(time_domain[0], time_domain[1]), n_time_thresh)
        ).tolist()
    bounded_pairs = list(itertools.combinations(bound_thresh, 2))
    right_unbounded_pairs = [(bound_thresh[0], -1)]
    unbounded_pair = [(-1, -1)]
    return bounded_pairs + right_unbounded_pairs + unbounded_pair


# this is useful for setting thresholds on templates in which variable operations have already been performed
def set_grid_thresh(
    phi,
    thresh_type,
    var_domain=None,
    n_var_thresh=15,
    time_domain=None,
    n_time_thresh=10,
    max_nvars=3,
    linspace=False,
):

    if thresh_type == "var":
        # set only var thresholds
        n_leaves, l_idx = get_number_leaves(phi)
        f_thresh = [
            get_var_thresh_list(i, var_domain, n_var_thresh, linspace) for i in l_idx
        ]
        task = "var_thresh"
    elif thresh_type == "temp":
        # set only temporal thresholds
        n_temp, _ = get_temp(phi)
        f_thresh = [
            get_temp_thresh_list(time_domain, n_time_thresh, linspace)
            for _ in range(n_temp)
        ]
        task = "temp_thresh"
    else:
        # set both var and temporal thresholds
        n_thresh_nodes, node_type = get_thresh_nodes(phi, max_nvars)
        f_thresh = [
            (
                get_var_thresh_list(t, var_domain, n_var_thresh, linspace)
                if isinstance(t, int)
                else get_temp_thresh_list(time_domain, n_time_thresh, linspace)
            )
            for t in node_type
        ]
        task = "all_thresh"
    f_thresh_comb = list(itertools.product(*f_thresh))

    def current_traverse(c):
        return copy.deepcopy(traverse_formula(phi, task=task, prop_list=list(c))[1])

    phi_thresh = list(map(current_traverse, f_thresh_comb))
    return phi_thresh


def var_ops(phi_list, max_idx, task="var_index"):
    var_idx_phi = []
    for f in phi_list:
        nvars, _ = get_number_leaves(f)
        prod = [None]
        if task == "var_index":
            prod = (
                list(itertools.product(range(max_idx), repeat=nvars))
                if nvars > 1
                else [[i] for i in range(max_idx)]
            )
        elif task == "var_sign":
            prod = list(itertools.product(range(2), repeat=nvars))
        for p in prod:
            new_p, new_f = traverse_formula(f, task=task, prop_list=list(p))
            if task == "var_sign":
                new_p, new_f = traverse_formula(
                    new_f, task="remove_redundant", prop_list=list(p)
                )
            if type(new_p) is not str:
                var_idx_phi.append(copy.deepcopy(new_f))
    return var_idx_phi


# this is useful for exhaustive search starting from raw templates (without indexed variables)
def from_template_to_instances_serial(
    phi_list,
    max_idx,
    var_domain=None,
    n_var_thresh=15,
    time_domain=None,
    n_time_thresh=10,
):
    phi_idx = var_ops(phi_list, max_idx, task="var_index")
    phi_sign = var_ops(phi_idx, max_idx, task="var_sign")
    phi_instances = []
    for ps in phi_sign:
        phi_threshed = set_grid_thresh(
            ps,
            "all_thresh",
            var_domain,
            n_var_thresh,
            time_domain,
            n_time_thresh,
            max_nvars=max_idx,
        )
        phi_instances += [copy.deepcopy(pt) for pt in phi_threshed]
    return phi_instances


def from_template_to_instances(
    phi_list,
    max_idx,
    var_domain=None,
    n_var_thresh=15,
    time_domain=None,
    n_time_thresh=10,
    procs=16,
):
    if procs == 0:
        return from_template_to_instances_serial(
            phi_list, max_idx, var_domain, n_var_thresh, time_domain, n_time_thresh
        )

    phi_idx = var_ops(phi_list, max_idx, task="var_index")
    phi_sign = var_ops(phi_idx, max_idx, task="var_sign")
    phi_instances = []
    pool_obj = mp.Pool(processes=procs)
    phi_instances = pool_obj.starmap(
        _from_template_to_instances_pll,
        zip(
            phi_sign,
            repeat(max_idx),
            repeat(var_domain),
            repeat(n_var_thresh),
            repeat(time_domain),
            repeat(n_time_thresh),
        ),
        chunksize=50,
    )
    return [x for sub in phi_instances for x in sub]


def _from_template_to_instances_pll(
    ps, max_idx, var_domain, n_var_thresh, time_domain, n_time_thresh
):
    phi_threshed = set_grid_thresh(
        ps,
        "all_thresh",
        var_domain,
        n_var_thresh,
        time_domain,
        n_time_thresh,
        max_nvars=max_idx,
    )
    return [copy.deepcopy(pt) for pt in phi_threshed]


def filter_time_depth(phi_list, max_time_depth):
    def check_time_depth(phi):
        return phi.time_depth() < max_time_depth

    phi_ok = list(filter(check_time_depth, phi_list))
    return phi_ok


def filter_nested_temp(phi_list, max_nested_ops, max_n_until):
    def check_nested_ops(phi):
        return max_nested_temp(phi, max_n_until) <= max_nested_ops

    phi_ok = list(filter(check_nested_ops, phi_list))
    return phi_ok


def filter_tautology_contradictions_serial(
    phi_list, device, max_nvars, points, samples=1000
):
    meas = BaseMeasure(device=device)
    traj = meas.sample(
        samples, varn=max_nvars, points=points
    )  # sample n_traj trajectories

    def get_sum_sat(phi, sig=traj):
        return (
            torch.sum(phi.boolean(sig)).cpu().numpy()
        )  # sum of satisfactions over all traj

    sat_sum = np.array(list(map(get_sum_sat, phi_list)))  # compute over each phi
    taut_contr_idx = list(
        np.append(np.where(sat_sum == 0)[0], np.where(sat_sum == samples)[0])
    )  # find contradictions (0) and tautologies (n_traj)
    valid_idx = set(np.arange(len(phi_list))).difference(
        set(taut_contr_idx)
    )  # remove invalid indices
    valid_phis = [phi_list[i] for i in valid_idx]  # retrieve valid formulae
    return valid_phis


def _get_sum_sat(args):
    phi, sig = args
    return torch.sum(phi.boolean(sig)).cpu().numpy()


def filter_tautology_contradictions(
    phi_list, device, max_nvars, points, chunksize=200, n_traj=1000, procs=1
):
    # IMPORTANT:
    # if __name__ == '__main__':
    #      mp.freeze_support()
    meas = BaseMeasure(device=device)
    traj = meas.sample(
        n_traj, varn=max_nvars, points=points
    )  # sample n_traj trajectories
    mp.set_start_method("spawn", force=True)
    with mp.Pool(processes=procs + 1) as pool:
        args_list = [(phi, traj) for phi in phi_list]
        sat_sum = np.array(pool.map(_get_sum_sat, args_list, chunksize=chunksize))
        # sat_sum = np.array(tqdm(pool.imap(_get_sum_sat, args_list, chunksize=chunksize), total = len(phi_list)))

    taut_contr_idx = list(
        np.append(np.where(sat_sum == 0)[0], np.where(sat_sum == n_traj)[0])
    )  # find contradictions (0) and tautologies (n_traj)
    valid_idx = set(list(np.arange(len(phi_list)))).difference(
        set(taut_contr_idx)
    )  # remove invalid indices
    valid_phis = list(
        operator.itemgetter(*valid_idx)(phi_list)
    )  # retrieve valid formulae
    return valid_phis


def filter_by_similarity(
    phis,
    kernel,
    threshold=0.9,
    chunk_size=100,
    pll=0,
    verbose=True,
    checkpoints_file=None,
    reduction_threshold=0,
):
    # mu0 = BaseMeasure(device=dev, sigma0=1., sigma1=1., q=0.1)
    # kernel = StlKernel(mu0, sigma2=0.44, varn=nvars)
    reduction = 1000000000
    if verbose:
        print(f"initial length in similarity filter: {len(phis)}")
    previous = phis
    while reduction > reduction_threshold:
        if verbose:
            print("... filtering ...")
        partial_phis = []
        with ThreadPoolExecutor(max_workers=pll + 1) as executor:
            futures = []
            phi_chunks = list(chunks(previous, chunk_size))
            for chunk in phi_chunks:
                futures.append(
                    executor.submit(
                        _select_from_similarity, chunk, kernel, threshold, pll
                    )
                )
            if verbose:
                for f in tqdm(as_completed(futures), total=len(futures)):
                    partial_phis.extend(f.result())
            else:
                for f in as_completed(futures):
                    partial_phis.extend(f.result())
        reduction = len(previous) - len(partial_phis)
        previous = partial_phis
        if verbose:
            print(f"removed {reduction} formulae ...")
        if checkpoints_file:
            save_file(partial_phis, checkpoints_file, path="formulae_sets/checkpoints/")
            if verbose:
                print("saved similarity filter checkpoint")
    if verbose:
        print(f"done! final length: {len(partial_phis)}")
    return partial_phis


def _select_from_similarity(phis, kernel, threshold, pll):
    gram = kernel.compute_bag(phis, pll=pll)  # compute gram matrix
    low_tri = torch.tril(gram, diagonal=-1)  # extract lower triangular
    phi_similarity = [
        np.where(low_tri[r, :] > threshold)[0].tolist() for r in range(low_tri.shape[0])
    ]  # find similar items
    set_similarity = set(list(np.arange(len(phis)))).difference(
        set([i for sublist in phi_similarity for i in sublist])
    )  # create set of unique indices
    selected_phis = [phis[i] for i in set_similarity]  # select non-similar items
    return selected_phis


# from old file


def set_var_threshold(phi, n_thresh, domain):
    # set n_thresh different combinations of the variable thresholds of a given formula
    # domain is a list of 2 values: highest and lowest, allowed for thresholds
    phi_thresh = []
    n_leaves = get_number_leaves(phi, idx=False)  # number of variables
    f_thresh = (domain[0] - domain[1]) * torch.rand(n_thresh, n_leaves) + domain[1]
    f_thresh_comb = f_thresh.tolist()
    ret_params = copy.deepcopy(f_thresh_comb)
    for comb in f_thresh_comb:
        _, new_ff = traverse_formula(phi, task="var_thresh", prop_list=comb)
        phi_thresh.append(copy.deepcopy(new_ff))
    return phi_thresh, ret_params


def set_temp_threshold(phi, n_thresh, max_time_depth, max_t=None):
    # set n_thresh different temporal thresholds for a given formula phi
    # max_t is the highest time allowed
    # max_time_depth is the largest temporal window of the formula allowed
    phi_thresh = []
    n_temp_ops = get_number_temporal(phi)  # number of temporal operators
    f_thresh = torch.randint(low=1, high=max_t, size=(n_thresh, n_temp_ops))
    f_thresh_comb = f_thresh.tolist()
    ret_params = copy.deepcopy(f_thresh_comb)
    for comb in f_thresh_comb:
        _, new_ff = traverse_formula(phi, task="temp_thresh", prop_list=comb)
        if new_ff.time_depth() < max_time_depth:
            phi_thresh.append(copy.deepcopy(new_ff))
    return phi_thresh, ret_params


def set_all_thresholds(
    phi_list, n_thresholds, var_domain, time_domain, time_depth, save=True, path=None
):
    # put together everything above to obtain dataset of formulae from dataset of templates
    thresh_phis, v_thresh_phis = [[] for _ in range(2)]
    for phi in phi_list:
        phi_v_thresh, _ = set_var_threshold(
            phi, n_thresholds, var_domain
        )  # set n_thresholds different combinations of the variable thresholds of a given formula
        v_thresh_phis += phi_v_thresh
    for phi_v in v_thresh_phis:
        phi_t_thresh, _ = set_temp_threshold(
            phi_v, n_thresholds, time_depth, max_t=time_domain
        )  # # set n_thresholds different temporal thresholds
        thresh_phis += phi_t_thresh
    if save:
        dump_pickle(path, thresh_phis)
    return thresh_phis


def get_number_temporal(phi):
    # find the number of temporal operators in a given formula
    phi_str = str(phi)
    phi_split = phi_str.split()
    phi_temp = [sub for sub in phi_split if sub[:2] in ["ev", "al", "un"]]
    return len(phi_temp)
