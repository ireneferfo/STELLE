"""
Formula Utilities - Utility functions for STL formula manipulation and analysis.
"""

import re
import copy
from collections import deque
from itertools import product, permutations
from typing import List, Optional
from .stl import Atom, Not, Eventually, Globally, Until, And, Or


def get_formula_template(formula) -> str:
    """
    Convert formula to variable-agnostic template.
    
    Example: 'x_0 > 1' becomes 'x_# > 1'
    
    Args:
        formula: STL formula object
        
    Returns:
        Template string with variable placeholders
    """
    formula_str = str(formula)
    return re.sub(r"x_\d+", "x_#", formula_str)


def get_unique_variables(template: str) -> List[str]:
    """
    Extract unique variable placeholders from template.
    
    Args:
        template: Formula template string
        
    Returns:
        List of unique variable patterns found
    """
    return list(set(re.findall(r"x_\d+", template)))


def create_variable_permutations(
    base_formulae: List,
    n_vars: int,
    existing_formulae: Optional[List] = None
) -> List:
    """
    Create all variable permutations for a set of base formulae.
    
    Args:
        base_formulae: Base formulae to permute
        n_vars: Number of variables to permute over
        existing_formulae: Existing formulae to avoid duplicates
        
    Returns:
        List of permuted formulae
    """
    if existing_formulae is None:
        existing_formulae = []
    
    existing_strings = {str(f) for f in existing_formulae}
    all_permutations = []
    
    for base_formula in base_formulae:
        permutations = compute_permutations([base_formula], n_vars)
        
        # Filter out duplicates
        new_permutations = [
            p for p in permutations 
            if str(p) not in existing_strings
        ]
        
        all_permutations.extend(new_permutations)
        existing_strings.update(str(p) for p in new_permutations)
    
    return all_permutations


def find_n_nodes(phi):
    # given a formula, find the number of nodes in its syntax tree
    elements = ["not", "and", "or", "always", "eventually", "until", "<=", ">="]
    pattern = r"|".join(map(re.escape, elements))
    s = str(phi)
    return len(re.findall(pattern, s))


def find_complete_permutation_groups(
    formulae: List,
    n_vars: int
) -> dict:
    """
    Find complete groups of variable permutations in a formula set.
    
    Args:
        formulae: List of formulae to analyze
        n_vars: Number of variables in the system
        
    Returns:
        Dictionary mapping templates to their complete permutation sets
    """
    template_groups = {}
    
    for formula in formulae:
        template = get_formula_template(formula)
        if template not in template_groups:
            template_groups[template] = []
        template_groups[template].append(formula)
    
    # Identify complete groups
    complete_groups = {}
    for template, group_formulae in template_groups.items():
        num_vars_in_template = len(get_unique_variables(template))
        expected_permutations = n_vars ** num_vars_in_template
        
        if len(group_formulae) >= expected_permutations:
            complete_groups[template] = group_formulae
    
    return complete_groups


def compute_variations(string, nvars):
    # Find all unique variable patterns and their order of appearance
    variables = re.findall(r"x_\d+", string)

    # Create a mapping from original variables to their first occurrence index
    var_map = {}
    unique_vars = []
    for var in variables:
        if var not in var_map:
            var_map[var] = len(unique_vars)
            unique_vars.append(var)

    # Generate template with placeholders that preserve relationships
    template_parts = []
    last_pos = 0
    for match in re.finditer(r"x_\d+", string):
        template_parts.append(string[last_pos : match.start()])
        template_parts.append(
            f"x_{{{var_map[match.group()]}}}"
        )  # Use format-style placeholders
        last_pos = match.end()
    template_parts.append(string[last_pos:])
    template = "".join(template_parts)

    variations = []
    # Generate all possible assignments to unique variable groups
    for perm in product(range(nvars), repeat=len(unique_vars)):
        # Replace each group with the same variable index
        formatted = template.format(*perm)
        variations.append(formatted)

    return variations


def check_validity_atoms(formula):
    pattern = re.compile(r'[<>]=\s*(-?\d+\.?\d*)')
    matches = pattern.findall(str(formula))
    for match in matches:
        if not (-4 <= float(match) <= 4):
            return False
    return True


def set_time_thresholds(st):
    unbound, right_unbound = [True, False]
    left_time_bound, right_time_bound = [0, 0]
    if st[-1] == ']':
        unbound = False
        time_thresholds = st[st.index('[')+1:-1].split(",")
        left_time_bound = int(time_thresholds[0])
        if time_thresholds[1] == 'inf':
            right_unbound = True
        else:
            right_time_bound = int(time_thresholds[1])
    return unbound, right_unbound, left_time_bound, right_time_bound


def from_string_to_formula(st):
    #! doesnt support Boolean
    root_arity = 2 if st.startswith('(') else 1
    st_split = st.split()
    if root_arity <= 1:
        root_op_str = copy.deepcopy(st_split[0])
        if root_op_str.startswith('x'):
            atom_sign = True if st_split[1] == '<=' else False
            root_phi = Atom(var_index=int(st_split[0][2:]), lte=atom_sign, threshold=float(st_split[2]))
            return root_phi
        else:
            assert (root_op_str.startswith('not') or root_op_str.startswith('eventually')
                    or root_op_str.startswith('always'))
            current_st = copy.deepcopy(st_split[2:-1])
            if root_op_str == 'not':
                root_phi = Not(child=from_string_to_formula(' '.join(current_st)))
            elif root_op_str.startswith('eventually'):
                unbound, right_unbound, left_time_bound, right_time_bound = set_time_thresholds(root_op_str)
                root_phi = Eventually(child=from_string_to_formula(' '.join(current_st)), unbound=unbound,
                                      right_unbound=right_unbound, left_time_bound=left_time_bound,
                                      right_time_bound=right_time_bound)
            else:
                unbound, right_unbound, left_time_bound, right_time_bound = set_time_thresholds(root_op_str)
                root_phi = Globally(child=from_string_to_formula(' '.join(current_st)), unbound=unbound,
                                    right_unbound=right_unbound, left_time_bound=left_time_bound,
                                    right_time_bound=right_time_bound)
    else:
        # 1 - delete everything which is contained in other sets of parenthesis (if any)
        current_st = copy.deepcopy(st_split[1:-1])
        if '(' in current_st:
            par_queue = deque()
            par_idx_list = []
            for i, sub in enumerate(current_st):
                if sub == '(':
                    par_queue.append(i)
                elif sub == ')':
                    par_idx_list.append(tuple([par_queue.pop(), i]))
            # open_par_idx, close_par_idx = [current_st.index(p) for p in ['(', ')']]
            # union of parentheses range --> from these we may extract the substrings to be the children!!!
            children_range = []
            for begin, end in sorted(par_idx_list):
                if children_range and children_range[-1][1] >= begin - 1:
                    children_range[-1][1] = max(children_range[-1][1], end)
                else:
                    children_range.append([begin, end])
            n_children = len(children_range)
            assert (n_children in [1, 2])
            if n_children == 1:
                # one of the children is a variable --> need to individuate it
                var_child_idx = 1 if children_range[0][0] <= 1 else 0  # 0 is left child, 1 is right child
                if children_range[0][0] != 0 and current_st[children_range[0][0] - 1][0:2] in ['no', 'ev', 'al']:
                    children_range[0][0] -= 1
                left_child_str = current_st[:3] if var_child_idx == 0 else \
                    current_st[children_range[0][0]:children_range[0][1] + 1]
                right_child_str = current_st[-3:] if var_child_idx == 1 else \
                    current_st[children_range[0][0]:children_range[0][1] + 1]
                root_op_str = current_st[children_range[0][1] + 1] if var_child_idx == 1 else \
                    current_st[children_range[0][0] - 1]
                assert (root_op_str[:2] in ['an', 'or', 'un'])
            else:
                if children_range[0][0] != 0 and current_st[children_range[0][0] - 1][0:2] in ['no', 'ev', 'al']:
                    children_range[0][0] -= 1
                if current_st[children_range[1][0] - 1][0:2] in ['no', 'ev', 'al']:
                    children_range[1][0] -= 1
                # if there are two children, with parentheses, the element in the middle is the root
                root_op_str = current_st[children_range[0][1] + 1]
                assert (root_op_str[:2] in ['an', 'or', 'un'])
                left_child_str = current_st[children_range[0][0]:children_range[0][1] + 1]
                right_child_str = current_st[children_range[1][0]:children_range[1][1] + 1]
        else:
            # no parentheses means that both children are variables
            left_child_str = current_st[:3]
            right_child_str = current_st[-3:]
            root_op_str = current_st[3]
        left_child_str = ' '.join(left_child_str)
        right_child_str = ' '.join(right_child_str)
        if root_op_str == 'and':
            root_phi = And(left_child=from_string_to_formula(left_child_str),
                           right_child=from_string_to_formula(right_child_str))
        elif root_op_str == 'or':
            root_phi = Or(left_child=from_string_to_formula(left_child_str),
                          right_child=from_string_to_formula(right_child_str))
        else:
            unbound, right_unbound, left_time_bound, right_time_bound = set_time_thresholds(root_op_str)
            root_phi = Until(left_child=from_string_to_formula(left_child_str),
                             right_child=from_string_to_formula(right_child_str),
                             unbound=unbound, right_unbound=right_unbound, left_time_bound=left_time_bound,
                             right_time_bound=right_time_bound)
    return root_phi


def change_variable_idx(phis, old, new):
    ps_strings = [str(p).replace(f'x_{old}', f'x_{new}') for p in phis]
    return list(map(from_string_to_formula, ps_strings))


def change_vars_thresholds(phis, val):
    # changes the floating numbers by 'val', keeping it inside [-4,4].
    # where more than one atom, creates as many phis as atoms changing one each time
    strings = [str(i) for i in phis]
    newphis = []
    for i in strings:
        floats = re.findall(r'[-+]?\d*\.\d+|\d+\.\d*',  i)
        updated_floats = [str(round(float(float_str) + val, 4)) for float_str in floats]
        updated_string = i
        for orig_float, updated_float in zip(floats, updated_floats):
            updated_string = updated_string.replace(orig_float, updated_float)
        updated_string.replace('--', '-')
        newphis.append(updated_string)    
    return list(map(from_string_to_formula, newphis))


def from_str_to_n_nodes(f):
    f_str = str(f)
    f_split = f_str.split()
    f_nodes_list = [
        sub_f
        for sub_f in f_split
        if re.sub(r"[\d\[\],]", "", sub_f).replace("inf", "")
        in ["not", "and", "or", "always", "eventually", "<=", ">=", "until"]
    ]
    return len(f_nodes_list)


def get_num_vars(phi):
    p = phi[0] if type(phi) == list else phi
    phi_str = str(p)
    phi_split = phi_str.split()
    phi_var = [sub for sub in phi_split if sub.startswith("x_")]
    return len(list(set(phi_var)))


def _get_variable_relationships(formula_str):
    """Identify which variables must stay distinct in the formula"""
    vars_in_formula = sorted(list(set(re.findall(r"x_(\d+)", formula_str))))
    return vars_in_formula


def _compute_relationship_preserving_permutations(formula_str, nvars, orig_vars):
    """Generate permutations that maintain original variable relationships"""
    k = len(orig_vars)

    # Generate all possible variable mappings that maintain distinctness
    if k > nvars:
        return []  # Not enough variables to maintain relationships

    # Generate all possible assignments of k distinct variables
    for var_assignment in permutations(range(nvars), k):
        mapping = {orig: new for orig, new in zip(orig_vars, var_assignment)}
        yield re.sub(r"x_(\d+)", lambda m: f"x_{mapping[m.group(1)]}", formula_str)


def compute_permutations(formulae, nvars, maintain_relationships=False):
    out = []
    if maintain_relationships:
        for f in formulae:
            out.append(compute_variations(str(f), nvars))
        return [from_string_to_formula(i) for j in out for i in j]
    for f in formulae:
        f_str = str(f)
        orig_vars = _get_variable_relationships(f_str)

        if len(orig_vars) == 1:
            # Simple case - just replace single variable
            out.extend(compute_variations(f_str, nvars))
        else:
            # Complex case - maintain relationships
            perms = list(
                _compute_relationship_preserving_permutations(f_str, nvars, orig_vars)
            )
            out.extend(perms)
    return [from_string_to_formula(i) for i in out]
