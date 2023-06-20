def T_ij(i, j, beta, S_ij, S_ijm2, S_ijp2):
      if (i, j) == (0, 0): return S_ij*beta - 2*S_ijp2*beta**2
      if (i, j) == (0, 1): return 3*S_ij*beta - 2*S_ijp2*beta**2
      if (i, j) == (0, 2): return 5*S_ij*beta + 1.0*S_ijm2 - 2*S_ijp2*beta**2
      if (i, j) == (1, 0): return S_ij*beta - 2*S_ijp2*beta**2
      if (i, j) == (1, 1): return 3*S_ij*beta - 2*S_ijp2*beta**2
      if (i, j) == (1, 2): return 5*S_ij*beta + 1.0*S_ijm2 - 2*S_ijp2*beta**2
      if (i, j) == (2, 0): return S_ij*beta - 2*S_ijp2*beta**2
      if (i, j) == (2, 1): return 3*S_ij*beta - 2*S_ijp2*beta**2
      if (i, j) == (2, 2): return 5*S_ij*beta + 1.0*S_ijm2 - 2*S_ijp2*beta**2
      raise NotImplementedError