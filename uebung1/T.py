def T_ij(i, j, beta, S_ij, S_ijm2, S_ijp2):
      if (i, j) == (0, 0): return S_ij*beta - 2*S_ijp2*beta**2
      if (i, j) == (0, 1): return 3*S_ij*beta - 2*S_ijp2*beta**2
      if (i, j) == (1, 0): return S_ij*beta - 2*S_ijp2*beta**2
      if (i, j) == (1, 1): return 3*S_ij*beta - 2*S_ijp2*beta**2
      raise NotImplementedError