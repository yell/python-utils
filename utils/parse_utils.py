from ast import literal_eval

from iter_utils import is_iterable


def safe_literal_eval(s):
    """
    Examples
    --------
    >>> type(safe_literal_eval('0.5'))
    <type 'float'>
    >>> type(safe_literal_eval('1'))
    <type 'int'>
    >>> type(safe_literal_eval('True'))
    <type 'bool'>
    """
    try:
        x = literal_eval(s)
    except (SyntaxError, ValueError):
        x = s
    return x


def parse_args(s, separator='__'):
    """
    Can be used for processing complex `argparse` arguments.

    Examples
    --------
    >>> parse_args('__True___False__None___')
    [True, False, None]
    >>> parse_args('0__1')
    [0, 1]
    >>> parse_args('1.__1.0__0.__0.0__1e-2__1e+2__1e2')
    [1.0, 1.0, 0.0, 0.0, 0.01, 100.0, 100.0]
    >>> parse_args('compound_name__another_one')
    ['compound_name', 'another_one']
    >>> parse_args('a__[0,1,2,(3,4,5),6.0]')
    ['a', [0, 1, 2, (3, 4, 5), 6.0]]
    """
    args = []
    for t in s.strip().split(separator):
        for c in separator:
            if t.startswith(c): t = t[1:]
            if t.endswith(c): t = t[:-1]
        if t: args.append(t)
    args = map(safe_literal_eval, args)
    return args


def parse_kwargs(s, separator='__'):
    """
    Can be used for processing complex `argparse` arguments.

    Examples
    --------
    >>> parse_kwargs('a__1___b__None___c__0.3')
    {'a': 1, 'c': 0.3, 'b': None}
    >>> parse_kwargs('header__a__1__b__None')
    ('header', {'a': 1, 'b': None})
    >>> parse_kwargs('key_name__value__another_key__another_value')
    {'key_name': 'value', 'another_key': 'another_value'}
    >>> parse_kwargs('(0,1,2)__[1.0,-2.3,2e3]')
    {(0, 1, 2): [1.0, -2.3, 2000.0]}
    """
    args = parse_args(s, separator=separator)
    it = iter(args[len(args) % 2:])
    kwargs = {k: v for k, v in zip(it, it)}
    if len(args) % 2 == 1:
        return args[0], kwargs
    return kwargs


def argparse_(tokens):
    """
    Examples
    --------
    >>> argparse_('-ab-c "egg" --flag --d 1 2 3.')
    {'-ab-c': '"egg"', '--flag': '', '--d': '1 2 3.'}
    >>> argparse_('-ab-c "egg" --d 1 2 3. --flag'.split())
    {'-ab-c': '"egg"', '--flag': '', '--d': '1 2 3.'}
    >>> argparse_('1 2')
    {}
    """
    if not is_iterable(tokens):
        tokens = tokens.split()

    dict_ = {}
    k, v = None, None

    for token in tokens:
        if token.startswith('-'):
            if k is not None:
                dict_[k] = v or ''
            k, v = token, None
        else:
            v = token if v is None else v + ' ' + token
    if k is not None:
        dict_[k] = v or ''
    return dict_
