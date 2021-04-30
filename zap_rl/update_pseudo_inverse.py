import numpy as np


def update_pseudo_inverse(array_a, vector_c, vector_d, array_a_dagger):
    """
    Update the pseudo-inverse of a matrix of real numbers.

    Bibliography: "Generalized Inversion of Modified Matrices", Meyer, 1973.

    Parameters
    ----------
    array_a : ndarray
        Rectangular numpy array.
    vector_c : ndarray
        Numpy array (vector). Length = number of rows of `array_a`.
    vector_d : ndarray
        Numpy array (vector). Length = number of columns of `array_a`.
    array_a_dagger : ndarray
        Pseudo-inverse of `array_a` (same shape as `array_a.T`).

    Returns
    -------
    None
        `array_a_dagger` is updated to become the pseudo-inverse of `array_a + np.outer(vector_c, vector_d)`.

    Notes
    -----
        All arrays must have real coefficients (not complex numbers).
    """
    vector_k = array_a_dagger @ vector_c
    vector_h = vector_d @ array_a_dagger
    beta = 1 + vector_h @ vector_c
    vector_u = vector_c - array_a @ vector_k
    vector_v = vector_d - vector_h @ array_a
    bool_c_in_r_of_a = (np.count_nonzero(vector_u) == 0)
    bool_d_in_r_of_a_transpose = (np.count_nonzero(vector_v) == 0)

    def dagger(vector):
        return vector / (vector @ vector)

    if bool_c_in_r_of_a:
        if beta == 0:
            if bool_d_in_r_of_a_transpose:
                # Theorem 6
                array_a_dagger += (
                    -np.outer(vector_k, dagger(vector_k) @ array_a_dagger)
                    - np.outer(array_a_dagger @ dagger(vector_h), vector_h)
                    + (dagger(vector_k) @ array_a_dagger @ dagger(vector_h)) * np.outer(vector_k, vector_h)
                )
            else:
                # Theorem 2
                array_a_dagger += (
                    - np.outer(vector_k, dagger(vector_k) @ array_a_dagger)
                    - np.outer(dagger(vector_v), vector_h)
                )
        else:
            # Theorem 3
            k_square = vector_k @ vector_k
            v_square = vector_v @ vector_v
            vector_p_1 = - k_square / beta * vector_v - vector_k
            vector_q_1 = - v_square / beta * vector_k @ array_a_dagger - vector_h
            sigma_1 = k_square * v_square + beta**2
            array_a_dagger += (
                1 / beta * np.outer(vector_v, vector_k @ array_a_dagger)
                - beta / sigma_1 * np.outer(vector_p_1, vector_q_1)
            )
    else:  # not bool_c_in_r_of_a
        if bool_d_in_r_of_a_transpose:
            if beta == 0:
                # Theorem 4
                array_a_dagger += (
                    - np.outer(array_a_dagger @ dagger(vector_h), vector_h)
                    - np.outer(vector_k, dagger(vector_u))
                )
            else:
                # Theorem 5
                h_square = vector_h @ vector_h
                u_square = vector_u @ vector_u
                vector_p_2 = - u_square / beta * array_a_dagger @ vector_h - vector_k
                vector_q_2 = - h_square / beta * vector_u - vector_h
                sigma_2 = h_square * u_square + beta**2
                array_a_dagger += (
                    1 / beta * np.outer(array_a_dagger @ vector_h, vector_u)
                    - beta / sigma_2 * np.outer(vector_p_2, vector_q_2)
                )
        else:
            # Theorem 1
            array_a_dagger += (
                - np.outer(vector_k, dagger(vector_u))
                - np.outer(dagger(vector_v), vector_h)
                + beta * np.outer(dagger(vector_v), dagger(vector_u))
            )
