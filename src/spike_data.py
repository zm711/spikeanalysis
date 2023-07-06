from pathlib import Path
from typing import Union
import os

import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm


class SpikeData:
    """class for loading "phy" type data files including generating some qc metrics and raw
    waveforms. To be used by SpikeAnalysis"""

    def __init__(self, file_path: Union[str, Path]):
        """
        class for loading spiking data and performing quality metrics

        Parameters
        ----------
        file_path : str, Path
              string or Path to the root directory of the in vivo data from Phy


        """

        file_path = Path(file_path)
        assert Path.is_dir(
            file_path
        ), "Enter root directory with *rhd file. If having problems for \
        windows prepend r in front of the str."

        self._file_path = file_path
        self.CACHING = False
        self.QC_RUN = False
        import glob

        current_dir = os.getcwd()

        os.chdir(file_path)

        assert len(glob.glob("spike_times.npy")) != 0, "This folder doesn't contain Phy files"
        self._filename = glob.glob("*bin")[0]

        with open("params.py", "r") as p:
            params = p.readlines()

        sampling_rate = float(params[4].split()[-1])
        self._binary_file_info = {}
        self._binary_file_info["n_channels"] = int(params[1].split()[-1])
        self._binary_file_info["offset"] = int(params[3].split()[-1])
        self._binary_file_info["dtype"] = str(params[2].split()[-1].strip("'"))
        self._sampling_rate = sampling_rate
        self.raw_spike_times = np.squeeze(np.load("spike_times.npy"))
        self._spike_templates = np.squeeze(np.load("spike_templates.npy"))

        if os.path.isfile("spike_clusters.npy"):
            self.spike_clusters = np.squeeze(np.load("spike_clusters.npy"))
        else:
            self.spike_clusters = self._spike_templates

        self._cids = np.array(list(set(self.spike_clusters)))

        self.template_scaling_amplitudes = np.squeeze(np.load("amplitudes.npy"))

        coords = np.load("channel_positions.npy")
        self.x_coords = coords[:, 0]
        self.y_coords = coords[:, 1]

        self._templates = np.load("templates.npy")

        self.whitening_matrix_inverse = np.load("whitening_mat_inv.npy")

        self._return_to_dir(current_dir)

    def __repr__(self):
        var_methods = dir(self)
        var = list(vars(self).keys())  # get our currents variables
        methods = list(set(var_methods) - set(var))
        final_methods = [method for method in methods if "__" not in method and method[0] != "_"]
        return f"The methods are {final_methods}"

    def set_caching(self, cache: bool = True):
        self.CACHING = cache

    def run_all(
        self,
        ref_dur_ms: float,
        idthres: float,
        rpv: float,
        sil: float,
        recurated: bool = False,
        set_caching: bool = True,
        depth: float = 0,
    ):
        current_dir = os.getcwd()
        self._goto_file_path()
        try:
            self._qc_threshold = np.load("qc_threshold.npy")
            self._return_to_dir(current_dir)
            return
        except FileNotFoundError:
            self.set_caching(set_caching)
            self.refractory_violation(ref_dur_ms=ref_dur_ms)
            self.generate_pcs()
            self.generate_qcmetrics()
            self.qc_preprocessing(idthres=idthres, rpv=rpv, sil=sil, recurated=recurated)
            self.set_qc()
            self.denoise_data()
            self.get_waveforms()
            self.get_waveform_values(depth=depth)
            self._return_to_dir(current_dir)

    def denoise_data(self):
        """
        Function for removing clusters labeled as noise in Phy

        """
        print("Denoising Data")
        current_dir = os.getcwd()
        self._goto_file_path()

        cids, cgs = self._read_cgs()
        self.cgs = cgs
        if isinstance(cids, int):
            return
        noise_clusters = []
        for index, label in enumerate(cgs):
            if label == 0:
                noise_clusters.append(cids[index])

        noise_clusters = np.array(noise_clusters)

        self.raw_spike_times = self.raw_spike_times[np.isin(self.spike_clusters, noise_clusters, invert=True)]
        self.template_scaling_amplitudes = self.template_scaling_amplitudes[
            np.isin(self.spike_clusters, noise_clusters, invert=True)
        ]
        self._spike_templates = self._spike_templates[np.isin(self.spike_clusters, noise_clusters, invert=True)]
        self.spike_clusters = self.spike_clusters[np.isin(self.spike_clusters, noise_clusters, invert=True)]
        # self.cgs = self.cgs[np.isin(cids, noise_clusters, invert=True)]
        self.noise = np.isin(cids, noise_clusters)

        # if len(cids) > len(self._cids):
        #    cids = self._cids
        # self._cids = self._cids[np.isin(cids, noise_clusters, invert=True)]

        self._return_to_dir(current_dir)

    def samples_to_seconds(self):
        """
        utility function which converts spike_times from samples to seconds

        Returns
        -------
        None.

        """

        self.spike_times = self.raw_spike_times / self._sampling_rate


    def refractory_violation(self, ref_dur_ms: float):
        """

        Calculates the number of refractory period violations for each cluster of spikes

        Parameters
        ----------
        ref_dur_ms : float
            The biological refractory period given in milliseconds. This is the value in which
            neurons should not be able to fire again.

        Returns
        -------
        None, but value stored as refractory_period_violations

        """
        print("calculating refractory period violation fraction")
        self._goto_file_path()
        ref_dur = ref_dur_ms / 1000
        spike_clusters = np.squeeze(np.load("spike_clusters.npy"))
        violations = np.zeros((len(set(spike_clusters))))
        violations[:] = np.nan

        try:
            spike_times = self.spike_times
        except AttributeError:
            spike_times = self.raw_spike_times / self._sampling_rate

        for idx, cluster in enumerate(tqdm(set(spike_clusters))):
            spikes = spike_times[self.spike_clusters == cluster]
            # print(len(spikes))
            if len(spikes) < 10:
                continue
            else:
                num_violations = float(len(np.where(np.diff(spikes) <= ref_dur)[0]))
                total_spikes = len(spikes)
                violations[idx] = num_violations / total_spikes

        self.refractory_period_violations = violations
        if self.CACHING:
            np.save("refractory_period_violations.npy", violations)

    def generate_pcs(self):
        """
        Reorganizes the Phy pc values based on manual curation

        Returns
        -------
        None, stored as pc_feat, and pc_feat_ind

        """
        print("generating pcs")
        current_dir = os.getcwd()
        self._goto_file_path()
        pc_features = np.load("pc_features.npy")
        pc_features_ind = np.load("pc_feature_ind.npy")
        try:
            spike_clusters = np.squeeze(np.load("spike_clusters.npy"))
        except FileNotFoundError:
            spike_clusters = np.load("spike_templates.npy")
        spike_templates = np.squeeze(np.load("spike_templates.npy"))
        # spike_clusters = self.spike_clusters
        # spike_templates = self._spike_templates

        cluster_ids = list(set(spike_clusters))
        n_clusters = len(cluster_ids)
        n_spikes = len(spike_clusters)
        n_feat = 8
        n_feat_per_chan = np.shape(pc_features)[1]

        new_feat = np.zeros((n_spikes, n_feat_per_chan, n_feat))
        new_feat_ind = np.zeros((n_clusters, n_feat))

        for cluster in tqdm(range(len(cluster_ids))):
            this_id = cluster_ids[cluster]
            these_spikes = spike_clusters == this_id
            these_templates = spike_templates[these_spikes]

            include_templates, inst = self._count_unique(these_templates)

            this_template = include_templates[inst == max(inst)]

            these_chans = pc_features_ind[this_template, :n_feat]
            new_feat_ind[cluster] = these_chans

            for feature in range(n_feat):
                this_chan_inds = pc_features_ind == these_chans[feature]
                temps_with_this_chan, chan_index = self._find_index(this_chan_inds)

                include_templates_with_this_feature = np.where(np.isin(include_templates, temps_with_this_chan))[0]
                for template in range(len(include_templates_with_this_feature)):
                    this_sub_template = include_templates[include_templates_with_this_feature[template]]
                    temp_channel_index = temps_with_this_chan == this_sub_template
                    this_template_feature_ind = chan_index[temp_channel_index]
                    these_spike_templates = spike_templates == this_sub_template
                    these_spikes_temps = np.squeeze(np.logical_and(these_spikes, these_spike_templates))

                    new_feat[these_spikes_temps, :, feature] = pc_features[
                        these_spikes_temps, :, this_template_feature_ind
                    ]

        if np.shape(pc_features)[1] != 3:
            raise ("Error generating pc features")

        self.pc_feat = new_feat
        self.pc_feat_ind = new_feat_ind

        self._return_to_dir(current_dir)

    def generate_qcmetrics(self):
        """
        Using pc_feat from `generate_pcs()` this runs isolation distance (mahal distance) as well as
        simplified silhouette score on all clusters. isolation distance ranges from 0-inf and
        silhouette score ranges from -1, 1

        Returns
        -------
        None, values stored as isolation_distances and silhouette_scores, respectively

        """
        print("generating qc metrics")
        current_dir = os.getcwd()
        self._goto_file_path()

        try:
            pc_feat = self.pc_feat
        except AttributeError:
            raise Exception("Run generate_pcs() first")

        pc_feat = np.reshape(pc_feat, (np.shape(pc_feat)[0], -1))
        labels = np.squeeze(np.load("spike_clusters.npy"))
        isolation_distances = np.zeros((len(set(labels))))
        isolation_distances[:] = np.nan
        silhouette_scores = np.zeros((len(set(labels))))
        silhouette_scores[:] = np.nan

        for idx, cluster in enumerate(tqdm(np.unique(labels))):
            isolation_distances[idx] = self._isolation_distance(pc_feat, labels, cluster)
            silhouette_scores[idx] = self._simplified_silhouette_score(pc_feat, labels, cluster)

        self.isolation_distances = isolation_distances
        self.silhouette_scores = silhouette_scores
        if self.CACHING:
            np.save("silhouette_scores.npy", silhouette_scores)
            np.save("isolation_distances.npy", isolation_distances)

        self._return_to_dir(current_dir)

    def get_waveforms(self, wf_window: tuple = (-40, 41), n_wfs: int = 500):
        """
        collects raw waveforms from the associated binary file used in Kilosort/Phy.

        Parameters
        ----------
        wf_window : tuple, optional
            This is the sample window to assess around each spike. The default is (-40,41).
        n_wfs : int, optional
            The number of waveforms to save for each cluster. The default is 500.

        Returns
        -------
        None, saves the spikes used in waveform_spike_times and the waveforms as waveforms

        """
        from .utils import NumpyEncoder
        import json

        
        current_dir = os.getcwd()
        self._goto_file_path()
        try:
            with open("waveforms.json", "r") as read_file:
                print("loading raw waveforms")
                self.waveforms = json.load(read_file)
                self.waveforms = np.array(self.waveforms)
            return
        except FileNotFoundError:
            print("generating raw waveforms")

        sample_rate = self._sampling_rate
        spike_times = self.raw_spike_times
        spike_clusters = self.spike_clusters
        ch_map = np.squeeze(np.load("channel_map.npy"))
        n_chan_map = len(ch_map)
        n_samples = self._get_file_size()

        wf_n_samples = (wf_window[1] - wf_window[0]) + 1
        filename = self._filename
        dtype = self._binary_file_info["dtype"]
        shape = (self._binary_file_info["n_channels"], n_samples)
        offset = self._binary_file_info["offset"]

        binary_memmap = np.memmap(
            filename=filename,
            dtype=dtype,
            offset=offset,
            shape=shape,
            order="F",
        )

        cluster_ids = self._cids
        n_clusters = len(cluster_ids)
        spike_time_keeps = np.empty((n_clusters, n_wfs))
        spike_time_keeps[:] = np.nan
        waveforms = np.empty((n_clusters, n_wfs, n_chan_map, wf_n_samples), dtype=self._binary_file_info["dtype"])

        for this_unit in tqdm(range(n_clusters)):
            current_spike_times = spike_times[spike_clusters == cluster_ids[this_unit]]
            current_unit_n_spikes = np.shape(current_spike_times)[0]
            spike_times_random_perm = current_spike_times[np.random.permutation(current_unit_n_spikes)]
            spike_time_keeps[this_unit, : np.min([n_wfs, current_unit_n_spikes])] = np.sort(
                spike_times_random_perm[: np.min([n_wfs, current_unit_n_spikes])]
            )

            for current_spike_time in range(np.min([n_wfs, current_unit_n_spikes])):
                spike_keep_index_start = int(spike_time_keeps[this_unit, current_spike_time] + wf_window[0])
                spike_keep_index_end = int(spike_time_keeps[this_unit, current_spike_time] + wf_window[1] + 1)
                current_waveform = binary_memmap[:n_chan_map, spike_keep_index_start:spike_keep_index_end]
                if np.shape(current_waveform)[1] < 82:
                    continue
                else:
                    waveforms[this_unit, current_spike_time, :, :] = current_waveform

        self.waveform_spike_times = spike_time_keeps / sample_rate
        self.waveforms = waveforms
        if self.CACHING:
            with open("waveforms.json", "w") as write_file:
                json.dump(self.waveforms, write_file, cls=NumpyEncoder)

        self._return_to_dir(current_dir)

    def qc_preprocessing(self, idthres: float, rpv: float, sil: float, recurated: bool = False):
        """
        function for curating data based on qc metrics and refractory periods

        Parameters
        ----------
        idthres : float
            The cutoff isolation distance, 0 means no curation.
        rpv : float
            Fractional rate of refractory period violations, 0 is no violations and 1 would be all violations okay
        sil : float
            Minimum silhouette score, [-1, 1], where bigger is better.
        recurated : bool, optional
            If data has been recurated in phy since the last data run. The default is False.

        Raises
        ------
        Exception
            If various functions haven't been run

        Returns
        -------
        None.

        """
        current_dir = os.getcwd()
        self._goto_file_path()
        try:
            threshold = np.load("qc_threshold.npy")
            MUST_CALCULATE = False
        except FileNotFoundError:
            MUST_CALCULATE = True

        if not recurated and not MUST_CALCULATE:
            self._qc_threshold = threshold
        else:
            try:
                _ = self.silhouette_scores
                _ = self.isolation_distances
            except AttributeError:
                try:
                    self.silhouette_scores = np.load("silhouette_scores.npy")
                    self.isolation_distances = np.load("isolation_distances.npy")
                except FileNotFoundError:
                    raise Exception("qc metrics has not been run")
            try:
                _ = self.refractory_period_violations
            except AttributeError:
                try:
                    self.refractory_period_violations = np.load("refractory_period_violations.npy")
                except FileNotFoundError:
                    raise Exception("refractory period violations not calculated")

            assert len(self.silhouette_scores) == len(self.isolation_distances), "Qc metrics should be same length"
            assert len(self.silhouette_scores) == len(
                self.refractory_period_violations
            ), "Refractory period violations should be same length as qc"

            iso_d_thres = np.where(self.isolation_distances > idthres, True, False)
            sil_thres = np.where(self.silhouette_scores > sil, True, False)
            rpv_thres = np.where(self.refractory_period_violations < rpv, True, False)

            threshold = np.logical_and(iso_d_thres, sil_thres)
            threshold = np.logical_and(threshold, rpv_thres)

            self._qc_threshold = threshold

            self._isolation_threshold = idthres
            self._sil_threshold = sil
            self._rpv = rpv

            if self.CACHING:
                np.save("qc_threshold.npy", threshold)

        self._return_to_dir(current_dir)

    def set_qc(self):
        """
        Function to load the qc mask onto the cluster ids.

        """
        current_dir = os.getcwd()
        self._goto_file_path()
        try:
            threshold = self._qc_threshold
        except AttributeError:
            raise Exception(
                f"Must run qc functions first ('generate_pcs', 'generate_qcmetrics', 'refractory_violation')"
            )

        self._cids = self._cids[threshold]
        self.QC_RUN = True
        self._return_to_dir(current_dir)

    def get_waveform_values(self, depth: float = 0):
        """
        Function that uses weighted average of waveforms to calculate waveform metrics
        such as duration, amplitude, depths

        Parameters
        ----------
        depth : float, optional
            If given this is the true depth of the probe in the tissue and will cause
            the depths to corrected to depth in tissue rather than distance from
            probe tip. The default is 0.

        Returns
        -------
        None.

        """
        xcoords = self.x_coords
        ycoords = self.y_coords

        mean_waveforms = np.nanmean(self.waveforms, axis=1)
        mean_amplitudes = mean_waveforms.max(axis=2) - mean_waveforms.min(axis=2)

        depth_raw = np.sum((mean_amplitudes * ycoords), axis=1) / np.sum(mean_amplitudes, axis=1)
        if depth:
            final_depth = depth - depth_raw
        else:
            final_depth = depth_raw

        max_site = np.argmax(mean_waveforms.max(axis=2), axis=1)
        waveforms_max = np.zeros((np.shape(mean_waveforms)[0], np.shape(mean_waveforms)[2]))
        for current_wf in range(np.shape(mean_waveforms)[0]):
            waveforms_max[current_wf] = mean_waveforms[current_wf, max_site[current_wf], :]

        waveform_trough = np.argmin(waveforms_max, axis=1)
        waveform_dur_raw = np.array(
            [np.argmax(waveforms_max[x, waveform_trough[x] :]) + 1 for x in range(np.shape(waveforms_max)[0])]
        )
        waveform_duration = waveform_dur_raw / self._sampling_rate

        amp_list = [
            waveforms_max[x, np.argmax(waveforms_max[x, waveform_trough[x] :])] - waveforms_max[x, waveform_trough[x]]
            for x in range(np.shape(waveforms_max)[0])
        ]
        waveform_amplitudes = np.array(amp_list)

        self.waveform_duration = waveform_duration
        self.waveform_amplitude = waveform_amplitudes
        self.waveform_depth = final_depth

    def save_qc_parameters(self):
        from .utils import jsonify_parameters

        try:
            idthres = self._isolation_threshold
            rpv = self._rpv
            sil = self._sil_threshold
        except:
            raise Exception(
                f"no qc run for saving parameters run ('refractory_violation', 'generate_pcs', 'generate_qcmetrics')"
            )

        qc_params = {
            "isolation distance": idthres,
            "refractory period violation fraction": rpv,
            "silhouette score": sil,
        }
        jsonify_parameters(qc_params)

    def _get_file_size(self) -> int:
        """
        Utility function to calculate a binary file size in samples

        Returns
        -------
        n_samples : int
            number of samples in a file.

        """
        file_size = os.path.getsize(self._filename)
        dtype = self._binary_file_info["dtype"]
        temp = np.array([0, 0, 0], dtype=dtype)
        temp2 = temp.view(np.uint8)
        data_type_n_bytes = len(temp2) / 3
        n_channels = self._binary_file_info["n_channels"]
        n_samples = int(file_size / (n_channels * data_type_n_bytes))

        return n_samples

    def _isolation_distance(self, pc_feat: np.array, labels: np.array, this_id: int) -> float:
        """
        Function for calculating isolation distances

        Parameters
        ----------
        pc_feat : np.array
            the pc features to be assessed over.
        labels : np.array
            cluster identities for each pc row.
        this_id : int
            current cluster identity currently being assessed.

        Returns
        -------
        float
            isolation distance for the cluster given by this_id.

        """
        pc_this_cluster = pc_feat[labels == this_id, :]
        pc_other_clusters = pc_feat[labels != this_id, :]

        n_spikes = np.shape(pc_this_cluster)[0]
        n_other_spikes = np.shape(pc_other_clusters)[0]
        if n_other_spikes > n_spikes:
            try:
                cov_matrix = np.linalg.inv(np.cov(pc_this_cluster, rowvar=False))
            except np.linalg.linalg.LinAlgError:
                return np.nan

            mean_this_cluster = np.reshape(np.mean(pc_this_cluster, axis=0), (1, -1))

            md_other = np.sort(cdist(pc_other_clusters, mean_this_cluster, "mahalanobis", VI=cov_matrix))

            # md_self = cdist(pc_this_cluster,mean_this_cluster,"mahalanobis", VI=cov_matrix)

            isolation_dist = (md_other[n_spikes - 1]) ** 2
        else:
            isolation_dist = np.nan

        return isolation_dist

    def _simplified_silhouette_score(self, pc_feat: np.array, labels: np.array, this_id: int) -> float:
        """
        the simplified silhouette score (Hrushka et al.) which uses centroid rather than pairwise comparisons
        for generating distances

        Parameters
        ----------
        pc_feat : np.array
            the pc features to be assessed over.
        labels : np.array
            cluster identities for each pc row.
        this_id : int
            current cluster identity currently being assessed.

        Returns
        -------
        float
            returns the simplified silhouette score.

        """
        pcs_for_this_unit = pc_feat[labels == this_id, :]
        centroid_for_this_unit = np.expand_dims(np.mean(pcs_for_this_unit, 0), 0)
        distances_for_this_unit = cdist(centroid_for_this_unit, pcs_for_this_unit)
        distance = np.inf

        # find centroid of other cluster and measure distances to that rather than pairwise
        # if less than current minimum distance update
        for label in np.unique(labels):
            if label != this_id:
                pcs_for_other_cluster = pc_feat[labels == label, :]
                centroid_for_other_cluster = np.expand_dims(np.mean(pcs_for_other_cluster, 0), 0)
                distances_for_other_cluster = cdist(centroid_for_other_cluster, pcs_for_this_unit)
                mean_distance_for_other_cluster = np.mean(distances_for_other_cluster)
                if mean_distance_for_other_cluster < distance:
                    distance = mean_distance_for_other_cluster
                    distances_for_minimum_cluster = distances_for_other_cluster

        sil_distances = (distances_for_minimum_cluster - distances_for_this_unit) / np.maximum(
            distances_for_minimum_cluster, distances_for_this_unit
        )

        unit_silhouette_score = np.mean(sil_distances)
        return unit_silhouette_score

    def _count_unique(self, x: np.array) -> tuple[int, int]:
        """
        Utility function for counting unique instances for each value in x

        Parameters
        ----------
        x : np.array
            an array of values

        Returns
        -------
        tuple[int, int]
            Each unique value and the number of instances of that value.

        """
        x = [np.int32(val) for val in x]
        values = [np.int32(x_val) for x_val in set(x)]
        instance = [np.int32(ins) for ins in range(0)]
        for val in values:
            instance.append(x.count(val))
        return values, instance

    def _find_index(self, matrix: np.array) -> tuple[np.array, np.array]:
        """
        Utility function for miming the syntax of Matlab find.

        Parameters
        ----------
        matrix : np.array
            a matrix as an np.array

        Returns
        -------
        row : TYPE
           row indices of nonzero values
        col : TYPE
            column indicies of nonzero values

        """
        index = np.transpose(np.nonzero(matrix))

        row = index[:, 0]
        col = index[:, 1]
        return row, col

    def _read_cgs(self):
        """
        Utility function for reading Phy cluster_group files (stored either as csv or tsv)

        Returns
        -------
        TYPE
            cluster ids
        TYPE
           cluster id identity converted to ints

        """
        if os.path.isfile("cluster_group.csv"):
            with open("cluster_group.csv", "r") as c:
                cgsfile = c.readlines()
        elif os.path.isfile("cluster_group.tsv"):
            with open("cluster_group.tsv", "r") as c:
                cgsfile = c.readlines()
        else:
            print("no cgs information provided. Data cannot be denoised.")
            return

        sub_cids = list()
        sub_cgs = list()
        for row in range(1, len(cgsfile)):
            sub_cids.append(int(cgsfile[row].split()[0]))
            sub_cgs.append(cgsfile[row].split()[1])
        cgs = np.zeros((len(sub_cgs)))
        for row in range(len(sub_cgs)):
            if sub_cgs[row] == "mua":
                cgs[row] = 1
            elif sub_cgs[row] == "good":
                cgs[row] = 2
            elif sub_cgs[row] == "unsorted":
                cgs[row] = 3
            else:
                cgs[row] = 0
        """
        number_clusters_missing = len(self._cids) - len(sub_cids)
        print(len(sub_cids), len(cgs))
        if number_clusters_missing > 0:
            missing_cgs = 2 * np.ones((number_clusters_missing))

            missing_cids = np.array(list(set(self._cids) - set(sub_cids)))

            if len(missing_cids) == len(self._cids):
                return 0, 0
            print(len(missing_cgs), len(missing_cids))
            
            assert len(missing_cgs) == len(missing_cids), "lengths of values not the same"

            sub_cids = np.concatenate((sub_cids, missing_cids))

            cgs = np.concatenate((cgs, missing_cgs))

            indices = np.argsort(sub_cids)

            sub_cids = sub_cids[indices]
            cgs = cgs[indices]"""

        return sub_cids, cgs

    def _goto_file_path(self):
        """
        Utility function to make sure in the correct directory

        """
        if os.getcwd() != self._file_path:
            os.chdir(self._file_path)

    def _return_to_dir(self, return_dir: str):
        """
        Utility function to return to whatever is the current directory

        Parameters
        ----------
        return_dir : str
           current directory.

        Returns
        -------
        None.

        """
        os.chdir(return_dir)
