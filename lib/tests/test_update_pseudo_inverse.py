import numpy as np
from fractions import Fraction
from zap_rl import update_pseudo_inverse


def test():
    """
    Preliminaries:

        >>> array_a = np.array((
        ...     (2, -1, 1, -1),
        ...     (1, -2, -1, -2),
        ...     (0, 0, 0, 0)
        ... ))
        >>> array_a_dagger_save = Fraction(1, 15) * np.array((
        ...     (5, -1, 0),
        ...     (0, -3, 0),
        ...     (5, -4, 0),
        ...     (0, -3, 0)
        ... ))

    c in R(A), d in R(A*), beta = 0 (Theorem 6):

        >>> array_a_dagger = array_a_dagger_save.copy()
        >>> vector_c = np.array((-1, -1, 0))
        >>> vector_d = np.array((2, -1, 1, -1))
        >>> update_pseudo_inverse(array_a, vector_c, vector_d, array_a_dagger)
        >>> np.allclose(
        ...     np.array(array_a_dagger, dtype=float),
        ...     np.linalg.pinv(array_a + np.outer(vector_c, vector_d))
        ... )
        True

    c in R(A), d not in R(A*), beta = 0 (Theorem 2):

        >>> array_a_dagger = array_a_dagger_save.copy()
        >>> vector_c = np.array((2, 1, 0))
        >>> vector_d = np.array((0, 2, -1, 1))
        >>> update_pseudo_inverse(array_a, vector_c, vector_d, array_a_dagger)
        >>> np.allclose(
        ...     np.array(array_a_dagger, dtype=float),
        ...     np.linalg.pinv(array_a + np.outer(vector_c, vector_d))
        ... )
        True

    c in R(A), d in R(A*), beta != 0 (Theorem 3 or Theorem 5):

        >>> array_a_dagger = array_a_dagger_save.copy()
        >>> vector_c = np.array((-1, -1, 0))
        >>> vector_d = np.array((1, -1, 0, -1))
        >>> update_pseudo_inverse(array_a, vector_c, vector_d, array_a_dagger)
        >>> np.allclose(
        ...     np.array(array_a_dagger, dtype=float),
        ...     np.linalg.pinv(array_a + np.outer(vector_c, vector_d))
        ... )
        True

    c in R(A), d not in R(A*), beta != 0 (Theorem 3):

        >>> array_a_dagger = array_a_dagger_save.copy()
        >>> vector_c = np.array((1, 1, 0))
        >>> vector_d = np.array((-2, 1, 0, 1))
        >>> update_pseudo_inverse(array_a, vector_c, vector_d, array_a_dagger)
        >>> np.allclose(
        ...     np.array(array_a_dagger, dtype=float),
        ...     np.linalg.pinv(array_a + np.outer(vector_c, vector_d))
        ... )
        True

    c not in R(A), d in R(A*), beta = 0 (Theorem 4):

        >>> array_a_dagger = array_a_dagger_save.copy()
        >>> vector_c = np.array((2, 1, -1))
        >>> vector_d = np.array((-1, 1, 0, 1))
        >>> update_pseudo_inverse(array_a, vector_c, vector_d, array_a_dagger)
        >>> np.allclose(
        ...     np.array(array_a_dagger, dtype=float),
        ...     np.linalg.pinv(array_a + np.outer(vector_c, vector_d))
        ... )
        True

    c not in R(A), d in R(A*), beta != 0 (Theorem 5):

        >>> array_a_dagger = array_a_dagger_save.copy()
        >>> vector_c = np.array((-2, 2, 2))
        >>> vector_d = np.array((-1, 1, 0, 1))
        >>> update_pseudo_inverse(array_a, vector_c, vector_d, array_a_dagger)
        >>> np.allclose(
        ...     np.array(array_a_dagger, dtype=float),
        ...     np.linalg.pinv(array_a + np.outer(vector_c, vector_d))
        ... )
        True

    c not in R(A), d not in R(A*), beta == 0 (Theorem 1):

        >>> array_a_dagger = array_a_dagger_save.copy()
        >>> vector_c = np.array((0, 1, 1))
        >>> vector_d = np.array((2, 2, 1, 2))
        >>> update_pseudo_inverse(array_a, vector_c, vector_d, array_a_dagger)
        >>> np.allclose(
        ...     np.array(array_a_dagger, dtype=float),
        ...     np.linalg.pinv(array_a + np.outer(vector_c, vector_d))
        ... )
        True

    c not in R(A), d not in R(A*), beta != 0 (Theorem 1):

        >>> array_a_dagger = array_a_dagger_save.copy()
        >>> vector_c = np.array((1, 1, -2))
        >>> vector_d = np.array((2, -1, -2, -2))
        >>> update_pseudo_inverse(array_a, vector_c, vector_d, array_a_dagger)
        >>> np.allclose(
        ...     np.array(array_a_dagger, dtype=float),
        ...     np.linalg.pinv(array_a + np.outer(vector_c, vector_d))
        ... )
        True
    """
    pass
