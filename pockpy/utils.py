""" Module containing various auxiliary functions used in other modules."""

import warnings

import numpy as np
import pandas as pd

def pinv(A, n):
    """ Performs a pseudoinverse on a :py:class:`numpy.2darray` or
    :py:class:`pandas.DataFrame` for a given number of singular values.

    Args:
        A: DataFrame or numpy.2darray to be pseudoinverted.
        n(int): The number of singular values to used for the pseudoinverse.
    Returns:
        Pseudoinverse of :py:data:`A` for the given number of singular values.
        If :py:data:`A` is a DataFrame, the pseudoinverse will be a DataFrame
        with index and columns swapped around.
    Raises:
        :py:exc:`ValueError`: If invalid :py:data:`n` is provided.
    """
    S = np.linalg.svd(A, compute_uv=False)

    if n == S.shape[0]:
        rcond = 0.0
    elif n > 0 and n < S.shape[0]:
        rcond = (0.9 * S[n] - 0.1 * S[n-1]) / S[0]
    elif n <= 0:
        raise ValueError(
            'Number of singular values must be strictly positive.')
    else:
        raise ValueError(
            f'{n} singular values was to be used, but there are only '\
            f'{S.shape[0]} singular values in the matrix.')

    A_pinv = np.linalg.pinv(A, rcond=rcond)

    # If applicable, invert index/columns.
    try:
       A_pinv = pd.DataFrame(A_pinv, index=A.columns, columns=A.index)
    except AttributeError:
        pass

    return A_pinv

def pinv_per_dim(df, n_x, n_y=None):
    """ Performs the pseudoinverse on a response matrix independently
    in each dimension.

    Args:
        df(pd.DataFrame): DataFrame to be pseudoinverted.
        n_x(int) : Integer representing the number of singular values to use
            for the pseudo-inverse in the horizontal plane.
        n_y(int) : Integer representing the number of singular values to use
            for the pseudoinverse in the vertical plane.
            Set to :py:data:`n_x` if not provided.
    Returns:
        Pseudoinverse of :py:data:`df` (applied per dimension) expressed as
        a DataFrame with index and columns swapped around.
    """
    if n_y is None:
        n_y = n_x

    df_pinv = pd.DataFrame(0.0, index=df.columns, columns=df.index)
    for plane, n in zip('XY', (n_x, n_y)):
        temp = df.loc[(slice(None), [plane, 'P'+plane], slice(None)), :]

        # Drop all zero columns, e.g. correctors not active in plane or
        # beam if only one beam is corrected.
        temp = temp.loc[:, (temp != 0).any(axis=0)]

        if n > min(temp.shape):
            n_max = min(temp.shape)
            warnings.warn(
                f'{n} singular values was to be used in the {plane}-plane, '\
                f'but the submatrix only has {n_max} singular values. '\
                f'Using {n_max} singular values instead and continuing..',
                UserWarning,
                stacklevel=1
            )
            n = n_max

        df_pinv.loc[temp.columns, (slice(None), [plane, 'P'+plane],
                                   slice(None))] = pinv(temp, n)
    return df_pinv

def remove_duplicates(itr):
    """ Removes duplicate elements from an iterable while retaining order
    and returns result as a list.

    The order of the input is maintained in the sense that the first
    occurrence of any given element defines its order.

    **Example**::

        >>> li = ['a', 'b', 'c', 'b' 'a']
        >>> remove_duplicates(li)
        ['a', 'b', 'c']
        >>> tup = ('b', 'a', 'b')
        >>> remove_duplicates(tup)
        ['b', 'a']

    Args:
        itr: An iterator.
    Returns:
        A list with all duplicate elements removed and with intact ordering.
    """
    temp=[]
    for x in itr:
        if x not in temp:
            temp.append(x)
    return temp

def compute_available_closed_orbit(aperture_scan_df, limit, col='N1'):
    """ Computes available closed orbit on input aperture scan table and limit.

    Args:
        aperture_scan_df(pd.DataFrame): DataFrame from a call to
            :py:func:`core_io.aperture_scan()`.
        limit(pd.Series): Series sharing the index of `py:data:`aperture_scan`
            where each entry specifies the minimum acceptable aperture of the
            kind specified by `py:data:`col`.
        col(str): String specifying which of the aperture outputs is
            constrained. Should be one of 'N1', 'N1X_M' or 'N1Y_M'.
    Returns:
        Series containing the maximal closed orbit uncertainty for which the
        aperture limit is satistfied per element.
    """

    # Dense one-liner. It does the following things in sequence:
    # 1. Retrieves the specified aperture column in the MultiIndex.
    # 2. Evaluates for each added closed orbit uncertainty whether it is
    #   greater than the specified limit.
    # 3. Per element (row), return the last (greatest) closed orbit uncertainty
    #   which evaluated to True.
    return aperture_scan_df.loc[:, col].gt(
        limit, axis=0).apply(lambda x: x[::-1].idxmax(), axis=1)

