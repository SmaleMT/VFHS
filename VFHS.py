import numpy as np
import pandas as pd
import collections
import warnings
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")


class VFHS(object):

    def __init__(self, k=5, n=1, random_state=42, gamma_pos=1):

        self.data = None
        self.dimension = None
        self.maj_label = None
        self.min_label = None
        self.k = k
        self.beta_neg = None
        self.alpha_pos = None
        self.beta_pos = None
        self.synthesis = None
        self.n = n
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.gamma_pos = gamma_pos

    def data_pre(self, X, y):
        if isinstance(y, pd.Series):
            y = y.values

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        data_set = np.concatenate((X, y), axis=1)
        self.data = pd.DataFrame(data=data_set)
        self.data.rename(columns={self.data.shape[1] - 1: 'label'}, inplace=True)

        self.dimension = self.data.columns[:-1]

        count = collections.Counter(y.flatten())
        labels = list(count.keys())

        if len(labels) < 2:
            raise ValueError("Data must contain at least two classes.")

        label_1, label_2 = labels[0], labels[1]
        self.min_label, self.maj_label = (
            (label_1, label_2)
            if count[label_1] < count[label_2]
            else (label_2, label_1)
        )


    def cal_info(self):
        X_data = self.data.loc[:, self.dimension].to_numpy(dtype=float)
        n_samples = X_data.shape[0]

        if n_samples <= 1:
            self.data['mass'] = [1e-8] * n_samples
            self.data['snb'] = [0.0] * n_samples
            self.data['force_vector'] = [np.zeros(len(self.dimension)) for _ in range(n_samples)]
            self.data['force_magnitude'] = [0.0] * n_samples
            self.data['direction_consistency'] = [0.0] * n_samples
            self.data['hetero_ratio'] = [0.0] * n_samples
            self.data['support'] = [0.0] * n_samples
            self.data['intrusion'] = [0.0] * n_samples
            self.data['net_score'] = [0.0] * n_samples
            return

        neigh = NearestNeighbors(
            n_neighbors=n_samples,
            algorithm='auto',
            metric='minkowski',
            p=2
        )
        neigh.fit(X_data)
        distances, indices = neigh.kneighbors(X_data)

        masses = []
        snbs = []
        valid_k_distances = []
        valid_k_indices = []

        effective_k = self.k if self.k < n_samples - 1 else n_samples - 1
        effective_k = max(1, effective_k)

        for i in range(n_samples):
            dists = distances[i]
            idxs = indices[i]

            overlap_number = np.sum(dists < 1e-12)

            if overlap_number + effective_k <= n_samples:
                t_dists = dists[overlap_number: overlap_number + effective_k]
                t_idxs = idxs[overlap_number: overlap_number + effective_k]
            else:
                t_dists = dists[-effective_k:]
                t_idxs = idxs[-effective_k:]

            if len(t_dists) == 0 or np.sum(t_dists) <= 1e-12:
                masses.append(1e-8)
                snbs.append(0.0)
                valid_k_distances.append(np.array([], dtype=float))
                valid_k_indices.append(np.array([], dtype=int))
            else:
                masses.append(1.0 / (np.sum(t_dists) + 1e-12))
                snbs.append(np.mean(t_dists))
                valid_k_distances.append(t_dists)
                valid_k_indices.append(t_idxs)

        self.data['mass'] = masses
        self.data['snb'] = snbs

        force_vectors = []
        force_magnitudes = []
        direction_consistencies = []
        hetero_ratios = []
        supports = []
        intrusions = []
        net_scores = []

        all_labels = self.data['label'].values

        for i in range(n_samples):
            x_i = X_data[i]
            mass_i = masses[i]
            t_dists = valid_k_distances[i]
            t_idxs = valid_k_indices[i]

            total_force = np.zeros_like(x_i)
            signed_vectors = []
            hetero_count = 0

            support = 0.0
            intrusion = 0.0

            for j, neighbor_idx in enumerate(t_idxs):
                dist = float(t_dists[j])
                if dist < 1e-8:
                    continue

                neighbor_label = all_labels[neighbor_idx]
                mass_j = masses[neighbor_idx]

                diff_vec = x_i - X_data[neighbor_idx]
                unit_vec = diff_vec / (dist + 1e-12)
                force_weight = (mass_i * mass_j) / np.exp(dist ** 2)

                if neighbor_label != all_labels[i]:
                    hetero_count += 1

                if neighbor_label == self.maj_label:
                    signed_vec = unit_vec
                else:
                    signed_vec = -unit_vec

                total_force += signed_vec * force_weight
                signed_vectors.append(signed_vec)

                if neighbor_label == all_labels[i]:
                    support += force_weight
                else:
                    intrusion += force_weight

            force_mag = np.linalg.norm(total_force)
            force_vectors.append(total_force)
            force_magnitudes.append(force_mag)

            if len(signed_vectors) > 0 and force_mag > 1e-8:
                tf_unit = total_force / (force_mag + 1e-12)
                cos_vals = []
                for vec in signed_vectors:
                    vec_norm = np.linalg.norm(vec)
                    if vec_norm > 1e-8:
                        vec_unit = vec / (vec_norm + 1e-12)
                        cos_vals.append(np.dot(vec_unit, tf_unit))
                consistency = float(np.mean(cos_vals)) if len(cos_vals) > 0 else 0.0
            else:
                consistency = 0.0

            hetero_ratio = hetero_count / max(len(t_idxs), 1)
            net_score = support - intrusion

            direction_consistencies.append(consistency)
            hetero_ratios.append(hetero_ratio)
            supports.append(support)
            intrusions.append(intrusion)
            net_scores.append(net_score)

        self.data['force_vector'] = force_vectors
        self.data['force_magnitude'] = force_magnitudes
        self.data['direction_consistency'] = direction_consistencies
        self.data['hetero_ratio'] = hetero_ratios
        self.data['support'] = supports
        self.data['intrusion'] = intrusions
        self.data['net_score'] = net_scores

    def cal_beta_neg(self):
        maj_data = self.data[self.data['label'] == self.maj_label]
        min_data = self.data[self.data['label'] == self.min_label]

        if maj_data.empty:
            self.beta_neg = 0.0
            return

        if min_data.empty:
            self.beta_neg = float(maj_data['net_score'].min()) - 1e-8
            return

        avg_maj_mass = maj_data['mass'].mean()
        avg_min_mass = min_data['mass'].mean()
        avg_maj_snb = maj_data['snb'].mean()

        maj_boundary = maj_data[maj_data['intrusion'] > 1e-12]

        if maj_boundary.empty:
            r_bar = 1.0 / self.k
            c_bar = 0.0
        else:
            r_bar = maj_boundary['hetero_ratio'].mean()
            c_bar = maj_boundary['direction_consistency'].mean()

            if np.isnan(r_bar):
                r_bar = 1.0 / self.k
            if np.isnan(c_bar):
                c_bar = 0.0

            r_bar = np.clip(r_bar, 0.0, 1.0)
            c_bar = np.clip(c_bar, 0.0, 1.0)

        omega_crit = (r_bar + c_bar) / 2.0
        omega_crit = np.clip(omega_crit, 1.0 / self.k, (self.k - 1.0) / self.k)

        k_diff = int(np.round(self.k * omega_crit))
        k_diff = max(1, min(self.k - 1, k_diff))
        k_same = self.k - k_diff

        same_term = (
            k_same * avg_maj_mass * avg_maj_mass /
            np.exp(avg_maj_snb ** 2)
        )
        diff_term = (
            k_diff * avg_maj_mass * avg_min_mass /
            np.exp(avg_maj_snb ** 2)
        )

        self.beta_neg = same_term - diff_term

    def cal_alpha_beta_pos(self):

        min_data = self.data[self.data['label'] == self.min_label]
        maj_data = self.data[self.data['label'] == self.maj_label]

        if min_data.empty:
            self.alpha_pos = 0.0
            self.beta_pos = 0.0
            return

        if maj_data.empty:
            ref = min_data['net_score']
            mu = float(ref.mean())
            sigma = float(ref.std())
            if np.isnan(sigma) or sigma < 1e-8:
                sigma = 1e-6
            self.alpha_pos = mu + self.gamma_pos * sigma
            self.beta_pos = mu - self.gamma_pos * sigma
            return

        avg_min_mass = float(min_data['mass'].mean())
        avg_maj_mass = float(maj_data['mass'].mean())
        avg_min_snb = float(min_data['snb'].mean())

        min_boundary = min_data[min_data['intrusion'] > 1e-12]

        if min_boundary.empty:
            r_bar = 1.0 / self.k
            c_bar = 0.0
            sigma_pos = float(min_data['net_score'].std())
        else:
            r_bar = float(min_boundary['hetero_ratio'].mean())
            c_bar = float(min_boundary['direction_consistency'].mean())
            sigma_pos = float(min_boundary['net_score'].std())

        if np.isnan(r_bar):
            r_bar = 1.0 / self.k
        if np.isnan(c_bar):
            c_bar = 0.0
        if np.isnan(sigma_pos) or sigma_pos < 1e-8:
            sigma_pos = float(min_data['net_score'].std())
        if np.isnan(sigma_pos) or sigma_pos < 1e-8:
            sigma_pos = 1e-6

        r_bar = np.clip(r_bar, 0.0, 1.0)
        c_bar = np.clip(c_bar, 0.0, 1.0)

        omega_pos = (r_bar + c_bar) / 2.0
        omega_pos = np.clip(omega_pos, 1.0 / self.k, (self.k - 1.0) / self.k)

        k_diff = int(np.round(self.k * omega_pos))
        k_diff = max(1, min(self.k - 1, k_diff))
        k_same = self.k - k_diff

        same_term = (
            k_same * avg_min_mass * avg_min_mass /
            np.exp(avg_min_snb ** 2)
        )
        diff_term = (
            k_diff * avg_min_mass * avg_maj_mass /
            np.exp(avg_min_snb ** 2)
        )

        center_pos = same_term - diff_term

        self.alpha_pos = center_pos + self.gamma_pos * sigma_pos
        self.beta_pos = center_pos - self.gamma_pos * sigma_pos

        if self.alpha_pos < self.beta_pos:
            self.alpha_pos, self.beta_pos = self.beta_pos, self.alpha_pos

    def undersampling(self):

        maj_mask = (
                self.data['label'] == self.maj_label
        )

        min_mask = (
                self.data['label'] == self.min_label
        )

        maj_before = int(maj_mask.sum())
        min_before = int(min_mask.sum())


        target_majority = int(
            self.n * min_before
        )

        target_majority = min(
            target_majority,
            maj_before
        )

        delete_number = max(
            0,
            maj_before - target_majority
        )

        if delete_number <= 0:
            print(
                "[UnderSampling] "
                "No deletion required."
            )

            return

        candidate_data = self.data.loc[
            maj_mask &
            (self.data['net_score'] <= self.beta_neg)
            ].copy()

        if candidate_data.empty:
            print(
                "[UnderSampling] "
                "Danger region empty."
            )

            return

        candidate_data = (
            candidate_data
            .sort_values(
                by='net_score',
                ascending=True
            )
        )

        delete_number = min(
            delete_number,
            len(candidate_data)
        )

        delete_idx = (
            candidate_data
            .iloc[:delete_number]
            .index
        )

        self.data = (
            self.data
            .drop(delete_idx)
            .reset_index(drop=True)
        )

        maj_after = int(
            (
                    self.data['label']
                    ==
                    self.maj_label
            ).sum()
        )

        print(
            f"[UnderSampling] "
            f"maj:{maj_before}->{maj_after}, "
            f"deleted:{delete_number}"
        )

    def oversampling(self):
        maj_data = self.data[self.data['label'] == self.maj_label]
        min_data = self.data[self.data['label'] == self.min_label].copy().reset_index(drop=True)

        self.synthesis = pd.DataFrame()

        if min_data.empty or len(min_data) <= 1:
            return

        target_min = int(
            maj_data.shape[0]
        )

        need_general = max(
            0,
            target_min - min_data.shape[0]
        )


        if need_general <= 0:
            return

        feature_cols = list(self.dimension)
        minority_X = min_data[feature_cols].to_numpy(dtype=float)

        n_min_neigh = min(len(min_data), self.k + 1)
        minority_neigh = NearestNeighbors(
            n_neighbors=n_min_neigh,
            algorithm='auto',
            metric='minkowski',
            p=2
        )
        minority_neigh.fit(minority_X)
        _, m_indices = minority_neigh.kneighbors(minority_X)

        min_safe = min_data[min_data['net_score'] >= self.alpha_pos].copy()
        min_critical = min_data[
            (min_data['net_score'] < self.alpha_pos) &
            (min_data['net_score'] > self.beta_pos)
        ].copy()
        min_danger = min_data[min_data['net_score'] <= self.beta_pos].copy()


        def generate_guided_samples(seed_idx, seed_row, count,
                                    eta_low=0.05, eta_high=0.12,
                                    noise_scale_factor=0.02):
            if count <= 0:
                return []

            seed_vec = seed_row[feature_cols].values.astype(float)
            force_vec = np.asarray(seed_row['force_vector'], dtype=float)
            force_mag = float(seed_row['force_magnitude'])
            local_snb = float(seed_row['snb'])

            neigh_candidates = m_indices[seed_idx][1:]
            if len(neigh_candidates) == 0:
                neigh_candidates = np.array([seed_idx])

            points = []

            for _ in range(count):
                neighbor_local_idx = int(self.rng.choice(neigh_candidates))
                neighbor_vec = minority_X[neighbor_local_idx].astype(float)

                lam = self.rng.uniform(0.2, 0.8)
                base_point = seed_vec + lam * (neighbor_vec - seed_vec)

                if force_mag > 1e-8:
                    direction = force_vec / (force_mag + 1e-12)
                else:
                    direction = np.zeros_like(seed_vec)

                eta = self.rng.uniform(eta_low, eta_high) * max(local_snb, 1e-8)
                guided_point = base_point + eta * direction

                noise = self.rng.normal(
                    0,
                    noise_scale_factor * max(local_snb, 1e-8),
                    size=seed_vec.shape
                )

                new_point = guided_point + noise
                points.append(new_point.tolist())

            return points


        if need_general > 0 and not min_critical.empty:
            crit_score = (
                (min_critical['intrusion'] + 1e-8) /
                (min_critical['support'] + min_critical['intrusion'] + 1e-8)
            )

            density_factor = np.sqrt(np.maximum(min_critical['snb'].to_numpy(dtype=float), 1e-8))
            crit_weight = crit_score.to_numpy(dtype=float) * density_factor
            crit_weight = np.maximum(crit_weight, 1e-8)
            crit_weight = crit_weight / crit_weight.sum()

            syn_numbers = np.floor(crit_weight * need_general).astype(int)
            syn_numbers = pd.Series(syn_numbers, index=min_critical.index)

            remainder = int(need_general - syn_numbers.sum())
            if remainder > 0:
                extra_indices = self.rng.choice(
                    min_critical.index.to_numpy(),
                    size=remainder,
                    replace=True if remainder > len(min_critical) else False
                )
                for idx in extra_indices:
                    syn_numbers.loc[idx] += 1

            for idx, row in min_critical.iterrows():
                count = int(syn_numbers.loc[idx]) if idx in syn_numbers.index else 0
                if count <= 0:
                    continue

                points = generate_guided_samples(
                    seed_idx=idx,
                    seed_row=row,
                    count=count,
                    eta_low=0.06,
                    eta_high=0.15,
                    noise_scale_factor=0.02
                )

                if len(points) > 0:
                    df_temp = pd.DataFrame(points, columns=feature_cols)
                    df_temp['label'] = self.min_label
                    self.synthesis = pd.concat([self.synthesis, df_temp], ignore_index=True)
                    need_general -= len(points)

                if need_general <= 0:
                    break


        if need_general > 0 and not min_safe.empty:
            replace_flag = len(min_safe) < need_general
            seeds = min_safe.sample(
                n=need_general,
                replace=replace_flag,
                random_state=self.random_state
            )

            generated_count = 0

            for idx, row in seeds.iterrows():
                points = generate_guided_samples(
                    seed_idx=idx,
                    seed_row=row,
                    count=1,
                    eta_low=0.03,
                    eta_high=0.08,
                    noise_scale_factor=0.01
                )

                if len(points) == 0:
                    continue

                df_temp = pd.DataFrame(points, columns=feature_cols)
                df_temp['label'] = self.min_label
                self.synthesis = pd.concat([self.synthesis, df_temp], ignore_index=True)
                generated_count += len(points)

            need_general = max(0, need_general - generated_count)

        if not self.synthesis.empty:
            self.synthesis.reset_index(drop=True, inplace=True)

    def fit_resample(self, X, y):
        self.data_pre(X, y)


        self.cal_info()
        self.cal_beta_neg()
        self.undersampling()


        self.cal_info()
        self.cal_alpha_beta_pos()
        self.oversampling()

        if self.synthesis is not None and not self.synthesis.empty:
            final_data = pd.concat([self.data, self.synthesis], ignore_index=True)
        else:
            final_data = self.data

        X_res = final_data.loc[:, self.dimension].to_numpy(dtype=float)
        y_res = final_data.loc[:, 'label'].to_numpy().flatten()

        return X_res, y_res
