.. _hpl:
=======
LINPACK
=======

The LINPACK benchmark solves a large equation system :math:`A \cdot x = b` for :math:`x`, where :math:`A \in \Bbb R^{n \times n}` and :math:`b,x \in \Bbb R^{n}`.
The implementation does this in two separate steps: First a LU factorization is calculated for matrix :math:`A`. The resulting matrices :math:`L` and :math:`U` are used
to solve the equations :math:`L \cdot y = b` and finally :math:`U \cdot x = y` to get the result for the vector :math:`x`.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :glob:

   */index