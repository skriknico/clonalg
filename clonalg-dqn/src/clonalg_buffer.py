import numpy as np


class CLONALGBuffer:
    """
    Buffer de replay bioinspirado basado en selección clonal (CLONALG).

    Cada transición es un anticuerpo. La afinidad mide su relevancia
    para el aprendizaje (basada en TD-error). El sampling mezcla
    selección proporcional a afinidad con muestreo uniforme para
    estabilidad, y aplica supresión clonal para garantizar diversidad.
    """

    def __init__(self, capacity=2000, clone_rate=0.5,
                 mutation_rate=0.1, suppression_threshold=0.999):
        self.capacity              = capacity
        self.clone_rate            = clone_rate
        self.mutation_rate         = mutation_rate
        self.suppression_threshold = suppression_threshold

        # Arrays pre-allocados (se redimensionan en el primer add)
        self.obs        = np.zeros((capacity, 1), dtype=np.float32)
        self.actions    = np.zeros(capacity,      dtype=np.int32)
        self.rewards    = np.zeros(capacity,      dtype=np.float32)
        self.next_obs   = np.zeros((capacity, 1), dtype=np.float32)
        self.dones      = np.zeros(capacity,      dtype=np.float32)
        self.affinities = np.ones(capacity,       dtype=np.float32)

        self._size = 0

        # Logs para diagnóstico (requeridos por el enunciado)
        self.mean_affinity_log = []
        self.suppression_log   = []

    # ── Añadir transición ────────────────────────────────────────────────
    def add(self, obs, action, reward, next_obs, done, td_error=None):
        obs      = np.array(obs,      dtype=np.float32)
        next_obs = np.array(next_obs, dtype=np.float32)

        # Inicializa dimensión real en el primer add
        if self._size == 0:
            obs_dim       = obs.shape[0]
            self.obs      = np.zeros((self.capacity, obs_dim), dtype=np.float32)
            self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)

        if self._size < self.capacity:
            idx = self._size
            self._size += 1
        else:
            # Reemplaza el anticuerpo de menor afinidad
            idx = int(np.argmin(self.affinities[:self._size]))

        self.obs[idx]        = obs
        self.actions[idx]    = int(action)
        self.rewards[idx]    = float(reward)
        self.next_obs[idx]   = next_obs
        self.dones[idx]      = float(done)
        self.affinities[idx] = 0.5  # afinidad inicial neutra

    # ── Actualizar afinidades con TD-errors reales ───────────────────────
    def update_affinity(self, indices, td_errors):
        """Flujo bidireccional DQN → CLONALG."""
        for i, td in zip(indices, td_errors):
            if 0 <= i < self._size:
                new_aff = np.tanh(abs(td))
                self.affinities[i] = 0.9 * self.affinities[i] + 0.1 * new_aff

    # ── Sampling con selección clonal ────────────────────────────────────
    def sample(self, batch_size):
        n   = self._size
        aff = self.affinities[:n].copy()

        # Probabilidades: mezcla 50% proporcional a afinidad + 50% uniforme
        aff_shifted = aff - aff.min() + 1e-6
        p_biased    = (aff_shifted ** 0.3)
        p_biased    = p_biased / p_biased.sum()
        p_uniform   = np.ones(n, dtype=np.float64) / n
        probs       = 0.5 * p_biased + 0.5 * p_uniform
        probs       = probs / probs.sum()  # renormaliza por seguridad

        # Selección de candidatos proporcional a afinidad
        n_candidates = min(n, batch_size * 3)
        candidates   = np.random.choice(n, size=n_candidates,
                                        replace=False, p=probs)

        # Supresión clonal vectorizada para garantizar diversidad
        diverse = self._suppress_vectorized(candidates)

        # Rellena si la supresión dejó pocos candidatos
        if len(diverse) < batch_size:
            remaining = list(set(range(n)) - set(diverse))
            if remaining:
                extra = np.random.choice(
                    remaining,
                    size=min(batch_size - len(diverse), len(remaining)),
                    replace=False
                ).tolist()
                diverse.extend(extra)

        final_idx = diverse[:batch_size]
        self.mean_affinity_log.append(float(aff[final_idx].mean()))

        return {
            "obs":      self.obs[final_idx],
            "actions":  self.actions[final_idx],
            "rewards":  self.rewards[final_idx],
            "next_obs": self.next_obs[final_idx],
            "dones":    self.dones[final_idx],
            "indices":  final_idx,
        }

    # ── Supresión clonal vectorizada ─────────────────────────────────────
    def _suppress_vectorized(self, candidates):
        if len(candidates) == 0:
            return []

        vecs      = self.obs[candidates].astype(np.float32)
        norms     = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        vecs_norm = vecs / norms
        sim       = (vecs_norm @ vecs_norm.T + 1.0) / 2.0  # [0, 1]

        selected   = []
        mask       = np.ones(len(candidates), dtype=bool)
        suppressed = 0

        for i in range(len(candidates)):
            if not mask[i]:
                continue
            selected.append(int(candidates[i]))
            too_similar    = sim[i] > self.suppression_threshold
            too_similar[i] = False
            suppressed    += int(too_similar[mask].sum())
            mask          &= ~too_similar

        self.suppression_log.append(suppressed)
        return selected

    def __len__(self):
        return self._size