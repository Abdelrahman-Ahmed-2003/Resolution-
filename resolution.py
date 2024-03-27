from nltk.sem import logic
from nltk.sem.logic import *

# Define test cases
test_cases = [
    # Implication Test Case with Predicates
    "some x all y (P(x) | P(y) -> (P(x) & P(y)))",

    # Biconditional Test Case with Predicates
    "all x ((P(x) <-> A) & (P(y) <-> B))",

    # Mixed Test Case with Predicates
    "some x all y (((P(x) | -P(y)) -> (P(x) & P(y))) & ((A | B) <-> (C & D)))",

    # Multiple Implications Test Case with Predicates
    "some x all y (((P(x) | P(y)) -> (P(x) & P(y))) & ((A | B) -> (C | D)))",

    # Implication with Negation Test Case with Predicates
    "some x all y ((-P(x) -> (P(y) | A)) & ((-B | P(z)) -> P(w)))"
]

test_cases = [logic.Expression.fromstring(i) for i in test_cases]


def eliminate_implication(expression):
    if isinstance(expression, IffExpression):  # <=> = (=> & <=)
        # (p <=> q) = (p => q) & (p <= q) = (~p | q) & (p | ~q)
        return AndExpression(
            eliminate_implication(
                OrExpression(-expression.first, expression.second)
            ),
            eliminate_implication(
                OrExpression(expression.first, -expression.second)
            )
        )
    elif isinstance(expression, ImpExpression):  # =>
        # (p => q) = ~p | q
        return eliminate_implication(
            OrExpression(-expression.first, expression.second)
        )
    elif isinstance(expression, (AndExpression, OrExpression)):  # &, |
        # (p & q) = p & q
        return type(expression)(
            eliminate_implication(expression.first),
            eliminate_implication(expression.second)
        )
    elif isinstance(expression, NegatedExpression):  # ~
        # ~p = ~p
        return NegatedExpression(
            eliminate_implication(expression.term)
        )
    elif isinstance(expression, (AllExpression, ExistsExpression)):  # A, E
        # (Ax p) = Ax p, (Ex p) = Ex p
        return type(expression)(
            expression.variable,
            eliminate_implication(expression.term)
        )
    else:
        # p = p
        return expression


def move_negation_inwards(expression):
    if isinstance(expression, NegatedExpression):
        if isinstance(expression.term, NegatedExpression):
            # ~~p = p
            return move_negation_inwards(expression.term.term)
        elif isinstance(expression.term, AndExpression):
            # ~(p & q) = ~p | ~q
            return OrExpression(
                move_negation_inwards(
                    NegatedExpression(expression.term.first)
                ),
                move_negation_inwards(
                    NegatedExpression(expression.term.second)
                )
            )
        elif isinstance(expression.term, OrExpression):
            # ~(p | q) = ~p & ~q
            return AndExpression(
                move_negation_inwards(
                    NegatedExpression(expression.term.first)
                ),
                move_negation_inwards(
                    NegatedExpression(expression.term.second)
                )
            )
        elif isinstance(expression.term, AllExpression):
            # ~Ax p = Ex ~p
            return ExistsExpression(
                expression.term.variable,
                move_negation_inwards(
                    NegatedExpression(expression.term.term)
                )
            )
        elif isinstance(expression.term, ExistsExpression):
            # ~Ex p = Ax ~p
            return AllExpression(
                expression.term.variable,
                move_negation_inwards(
                    NegatedExpression(expression.term.term)
                )
            )
        else:
            # ~p = ~p
            return expression
    elif isinstance(expression, (AndExpression, OrExpression)):
        # (p & q) = p & q, (p | q) = p | q
        return type(expression)(
            move_negation_inwards(expression.first),
            move_negation_inwards(expression.second)
        )
    elif isinstance(expression, (AllExpression, ExistsExpression)):
        # (Ax p) = Ax p, (Ex p) = Ex p
        return type(expression)(
            expression.variable,
            move_negation_inwards(expression.term)
        )
    else:
        # p = p
        return expression


def standardize_variables(expression, mapping=None):
    if mapping is None:
        mapping = set()

    if isinstance(expression, AllExpression):
        # Ax P(x) and x is in mapping then change x to a new variable and add it to mapping = Ay P(y)
        if expression.variable in mapping:
            expression = expression.alpha_convert(unique_variable(ignore=mapping))
        mapping.add(expression.variable)
        return AllExpression(
            expression.variable,
            standardize_variables(expression.term, mapping)
        )
    elif isinstance(expression, ExistsExpression):
        # Ex P(x) and x is in mapping then change x to a new variable and add it to mapping = Ey P(y)
        if expression.variable in mapping:
            expression = expression.alpha_convert(unique_variable(ignore=mapping))
        mapping.add(expression.variable)
        return ExistsExpression(
            expression.variable,
            standardize_variables(expression.term, mapping)
        )
    elif isinstance(expression, (AndExpression, OrExpression)):
        # (p & q) = (p & q), (p | q) = (p | q)
        return type(expression)(
            standardize_variables(expression.first, mapping),
            standardize_variables(expression.second, mapping)
        )
    elif isinstance(expression, NegatedExpression):
        # ~p = ~p
        return NegatedExpression(
            standardize_variables(expression.term, mapping)
        )
    else:
        # p = p
        return expression


def prenex_form(expression):
    def extract_quantifiers(expression):
        if isinstance(expression, (AndExpression, OrExpression)):
            # (p & q) return (p & q), [], [] and (p | q) return (p | q), [], []
            first = extract_quantifiers(expression.first)
            second = extract_quantifiers(expression.second)
            return type(expression)(first[0], second[0]), first[1] + second[1], first[2] + second[2]
        elif isinstance(expression, NegatedExpression):
            # ~p return ~p, [], []
            term = extract_quantifiers(expression.term)
            return NegatedExpression(term[0]), term[1], term[2]
        elif isinstance(expression, AllExpression):
            # Ax p return p, [AllExpression], [x]
            term = extract_quantifiers(expression.term)
            return term[0], term[1] + [AllExpression], term[2] + [expression.variable]
        elif isinstance(expression, ExistsExpression):
            # Ex p return p, [ExistsExpression], [x]
            term = extract_quantifiers(expression.term)
            return term[0], term[1] + [ExistsExpression], term[2] + [expression.variable]
        else:
            # p return p, [], []
            return expression, [], []

    expression, quantifiers, variables = extract_quantifiers(expression)
    # sort quantifiers by AllExpression
    for quantifier, variable in sorted(zip(quantifiers, variables), key=lambda x: 1 if x[0] == AllExpression else 0):
        # p = quantifier(variable, p)
        expression = quantifier(variable, expression)
    return expression


def skolemization(expression, mapping=None):
    if mapping is None:
        mapping = set()
    if isinstance(expression, ExistsExpression):
        # Ex P(x) = P(sk(x))
        return skolemization(
            expression.term.replace(expression.variable, skolem_function(mapping)),
            mapping
        )
    elif isinstance(expression, AllExpression):
        # Ax P(x) = Ax P(x)
        return AllExpression(
            expression.variable,
            skolemization(expression.term, mapping | {expression.variable})
        )
    else:
        # P(x) = P(x)
        return expression


def rename_variables(expressions, mapping=None):
    if mapping is None:
        mapping = set()

    def rename(expression):
        if isinstance(expression, AllExpression):
            # Ax P(x) and x is in mapping then change x to a new variable and add it to mapping = Ay P(y)
            if expression.variable in mapping:
                expression = expression.alpha_convert(unique_variable(ignore=mapping))
            mapping.add(expression.variable)
            return AllExpression(
                expression.variable,
                rename(expression.term)
            )
        else:
            # P(x) = P(x)
            return expression

    return [rename(expression) for expression in expressions]


def eliminate_universal_quantifiers(expression):
    if isinstance(expression, AllExpression):
        # Ax P(x) = P(x)
        return eliminate_universal_quantifiers(expression.term)
    else:
        # P(x) = P(x)
        return expression


def conjunctive_normal_form(expression):
    if isinstance(expression, AndExpression):
        # (p & q) = (p & q)
        return AndExpression(
            conjunctive_normal_form(expression.first),
            conjunctive_normal_form(expression.second)
        )
    elif isinstance(expression, OrExpression):
        if isinstance(expression.first, AndExpression):
            # ((p & q) | r) = (p | r) & (q | r)
            return AndExpression(
                conjunctive_normal_form(
                    OrExpression(expression.first.first, expression.second)
                ),
                conjunctive_normal_form(
                    OrExpression(expression.first.second, expression.second)
                )
            )
        elif isinstance(expression.second, AndExpression):
            # (r | (p & q)) = (r | p) & (r | q)
            return AndExpression(
                conjunctive_normal_form(
                    OrExpression(expression.first, expression.second.first)
                ),
                conjunctive_normal_form(
                    OrExpression(expression.first, expression.second.second)
                )
            )
        else:
            # (p | q) = (p | q)
            return OrExpression(
                conjunctive_normal_form(expression.first),
                conjunctive_normal_form(expression.second)
            )
    else:
        # p = p
        return expression


def convert_to_clause(expressions):
    def split_conjunctions(expression):
        if isinstance(expression, AndExpression):
            # (p & q) = [p, q]
            return split_conjunctions(expression.first) + split_conjunctions(expression.second)
        else:
            # p = [p]
            return [expression]

    def split_disjunctions(expression):
        if isinstance(expression, OrExpression):
            # (p | q) = [p, q]
            return split_disjunctions(expression.first) + split_disjunctions(expression.second)
        else:
            # p = [p]
            return [expression]

    def flatten(expression):
        if not isinstance(expression, list):
            # p = [p]
            return [expression]
        flat = []
        for item in expression:
            # [] + [p, [q]] = [p, q]
            flat.extend(flatten(item))
        return flat

    # (p | q) & (r | s | t) = [(p | q), (r | s | t)]
    clauses = flatten([split_conjunctions(expression) for expression in expressions])
    # [(p | q), (r | s | t)] = [[p, q], [r, s, t]]
    clauses = [split_disjunctions(clause) for clause in clauses]
    return clauses


def resolution(expressions):
    # (p => q) = ~p | q
    expressions = [eliminate_implication(expression) for expression in expressions]
    print("After eliminating implication:")
    for i in expressions:
        print('', i)
    print()

    # ~(p & q) = ~p | ~q, ~(p | q) = ~p & ~q
    expressions = [move_negation_inwards(expression) for expression in expressions]
    print("After moving negation inwards:")
    for i in expressions:
        print('', i)
    print()

    # Ax P(x) & Ax Q(x) = Ax P(x) & Ay Q(y)
    expressions = [standardize_variables(expression) for expression in expressions]
    print("After standardizing variables:")
    for i in expressions:
        print('', i)
    print()

    # Ax (P(x) & Ey Q(y)) = Ax Ey (P(x) & Q(y))
    expressions = [prenex_form(expression) for expression in expressions]
    print("After prenex form:")
    for i in expressions:
        print('', i)
    print()

    # Ex P(x) = P(sk(x))
    expressions = [skolemization(expression) for expression in expressions]
    print("After skolemization:")
    for i in expressions:
        print('', i)
    print()

    # Ax P(x) and Ax Q(x) = Ax P(x) and Ay Q(y)
    expressions = rename_variables(expressions)
    print("After renaming variables:")
    for i in expressions:
        print('', i)
    print()

    # Ax P(x) = P(x)
    expressions = [eliminate_universal_quantifiers(expression) for expression in expressions]
    print("After eliminating universal quantifiers:")
    for i in expressions:
        print('', i)
    print()

    # (p & q) | r = (p | r) & (q | r)
    expressions = [conjunctive_normal_form(expression) for expression in expressions]
    print("After conjunctive normal form:")
    for i in expressions:
        print('', i)
    print()

    # (p | q) = [[p, q]]
    clauses = convert_to_clause(expressions)
    # Turn clauses into a string
    clauses = [[str(i) for i in clause] for clause in clauses]
    print("After converting to clause:")
    for i in clauses:
        print('', i)


resolution([test_cases[0]])
