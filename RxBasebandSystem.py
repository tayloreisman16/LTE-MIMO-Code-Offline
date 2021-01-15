import numpy as np
import matplotlib.pyplot as plt


class RxBasebandSystem:
    def __init__(self, multi_ant_sys, Caz, param_est, case):
        self.num_ant_txrx = multi_ant_sys.num_ant_txrx
        self.MIMO_method = multi_ant_sys.MIMO_method
        self.NFFT = multi_ant_sys.NFFT
        self.len_CP = multi_ant_sys.len_CP

        self.rx_buff_len = self.NFFT + self.len_CP
        self.input_time_series_data = multi_ant_sys.buffer_data_rx_time[::]
        # print(multi_ant_sys.buffer_data_rx_time.shape)
        # print(self.rx_buffer_time0.shape)

        self.used_bins_data = multi_ant_sys.used_bins_data
        self.SNR = multi_ant_sys.SNR_lin

        self.channel_freq = multi_ant_sys.channel_freq
        self.symbol_pattern = multi_ant_sys.symbol_pattern

        # Dont know what these are right now
        self.UG = np.zeros((self.num_ant_txrx, self.num_ant_txrx, len(self.used_bins_data)))
        self.SG = np.zeros((self.num_ant_txrx, self.num_ant_txrx, len(self.used_bins_data)))
        self.VG = np.zeros((self.num_ant_txrx, self.num_ant_txrx, len(self.used_bins_data)))

        self.U = multi_ant_sys.U
        self.S = multi_ant_sys.S
        self.V = multi_ant_sys.V

        self.stream_size = multi_ant_sys.stream_size
        self.noise_var = multi_ant_sys.noise_var

        self.ref_only_bins = multi_ant_sys.ref_only_bins
        self.chan_max_offset = multi_ant_sys.chan_max_offset
        self.h_f = multi_ant_sys.h_f  # channel FFT
        self.genie_chan_time = multi_ant_sys.genie_chan_time

        self.synch_reference = Caz.ZChu0
        self.synch_used_bins = multi_ant_sys.used_bins_synch  # Same as Caz.used_bins.astype(int)

        # window: CP to end of symbol
        self.ptr_o = np.array(range(self.len_CP, self.len_CP + self.NFFT)).astype(int)
        self.ptr_i = self.ptr_o - np.ceil(self.len_CP / 2).astype(int)

        # Start from the middle of the CP
        self.rx_buffer_time_data = self.input_time_series_data[:, self.ptr_i]

        # print(self.rx_buffer_time)
        lmax_s = int(len(self.symbol_pattern) - sum(self.symbol_pattern))
        lmax_d = int(sum(self.symbol_pattern))
        print()

        self.correlation_frame_index_value_buffer = np.ones((self.input_time_series_data.shape[0], 3))  # NEED TO CHANGE THIS IN GNURADIO
        self.correlation_frame_index_value_buffer = np.zeros((self.num_ant_txrx, lmax_s, 2))  # ONE OF THESE 2 WILL BE REMOVED

        '''obj.EstChanFreqP=zeros(obj.MIMOAnt,LMAXS,obj.Nfft);
           obj.EstChanFreqN=zeros(obj.MIMOAnt, LMAXS,length(obj.SynchBinsUsed));
           obj.EstChanTim=zeros(obj.MIMOAnt, LMAXS,2);
           obj.EstSynchFreq=zeros(obj.MIMOAnt, LMAXS,length(obj.SynchBinsUsed));'''

        self.est_chan_freq_p = np.zeros((self.num_ant_txrx, lmax_s, self.NFFT), dtype=complex)
        self.est_chan_freq_n = np.zeros((self.num_ant_txrx, lmax_s, len(self.synch_used_bins)), dtype=complex)
        self.est_chan_time = np.zeros((self.num_ant_txrx, lmax_s, 3), dtype=complex)
        self.est_synch_freq = np.zeros((self.num_ant_txrx, lmax_s, len(self.synch_used_bins)), dtype=complex)

        # print(self.est_chan_freq_p.shape, self.est_chan_freq_n.shape, self.est_chan_time.shape, self.est_synch_freq.shape)
        if self.num_ant_txrx == 1:
            self.est_data_freq = np.zeros((self.num_ant_txrx, 1, len(self.used_bins_data)), dtype=complex)
        elif self.num_ant_txrx == 2 and self.MIMO_method == 'STCode':
            self.est_data_freq = np.zeros((1, 1, len(self.used_bins_data)), dtype=complex)
        elif self.num_ant_txrx == 2 and self.MIMO_method == 'SPMult':
            self.est_data_freq = np.zeros((2, 1, len(self.used_bins_data)), dtype=complex)

        # Max length of channel impulse is CP
        self.est_chan_impulse = np.zeros((self.num_ant_txrx, lmax_s, self.NFFT), dtype=complex)

        self.num_of_synchs_and_synch_bins = Caz.M.astype(int)
        self.synch_data_pattern = Caz.synch_data
        # print("CAZ: ", Caz.synch_state)
        self.synch_state = Caz.synch_state

        self.param_est = param_est
        self.case = case

        self.SNR_analog = multi_ant_sys.SNR_analog
        self.stride_value = None
        self.correlation_observations = None
        self.start_sample = None
        self.correlation_matrix = None

    def param_est_synch(self, sys_model):

        self.stride_value = np.ceil(self.len_CP / 2)
        self.correlation_frame_index_value_buffer = np.zeros((self.num_ant_txrx, 250, 3))  # There are two more in the init.

        for antenna_index in range(1):
            self.correlation_observations = -1

            chan_q = self.genie_chan_time[antenna_index, 0, :]  # 2048
            self.start_sample = (self.len_CP - 4) - 1

            total_loops = int(np.ceil(self.input_time_series_data.shape[1] / self.stride_value))
            # print(total_loops)
            correlation_val_vector = np.zeros(total_loops)

            ptr_adj, loop_count, pattern_count = 0, 0, 0

            tap_delay = 5
            symbol_count = np.zeros(tap_delay)
            corrected_ptr = np.zeros(1000)
            while loop_count <= total_loops:
                # print(loop_count)
                if self.correlation_observations == -1:
                    ptr_frame = loop_count * self.stride_value + self.start_sample + ptr_adj
                elif self.correlation_observations < 5:
                    ptr_frame += sum(self.synch_data_pattern) * (self.NFFT + self.len_CP)
                else:
                    ptr_frame = (np.ceil(np.dot(x_symbol_count_lookahead[-1:], m_c_coefficients) - self.len_CP / 4))[0]

                # print(self.rx_buffer_time0.shape[1])
                if (self.num_of_synchs_and_synch_bins[0] - 1) * self.rx_buff_len + self.NFFT + ptr_frame < self.input_time_series_data.shape[1]:
                    # if (self.MM[0] - 1)*self.rx_buff_len + self.NFFT + ptr_frame - 1 < self.rx_buffer_time0.shape[1]:
                    for i in range(self.num_of_synchs_and_synch_bins[0]):
                        # print(i)
                        start = int(i * self.rx_buff_len + ptr_frame)
                        fin = int(i * self.rx_buff_len + ptr_frame + self.NFFT)
                        self.rx_buffer_time_data[i * self.NFFT: (i + 1) * self.NFFT] = self.input_time_series_data[antenna_index, start:fin]

                    # Take FFT of the window
                    fft_vec = np.zeros((self.num_of_synchs_and_synch_bins[0], self.NFFT), dtype=complex)
                    for i in range(self.num_of_synchs_and_synch_bins[0]):
                        start = i * self.NFFT
                        fin = (i + 1) * self.NFFT
                        fft_vec[i, 0:self.NFFT] = np.fft.fft(self.rx_buffer_time_data[start: fin], self.NFFT)

                    # print(self.used_bins_synch)
                    # print(self.used_bins_synch.shape)
                    synch_freq_data = fft_vec[:, self.synch_used_bins]
                    synch_freq_data_vector = np.reshape(synch_freq_data, (1, synch_freq_data.shape[0] * synch_freq_data.shape[1]))
                    synch_pow_est = sum(sum(synch_freq_data_vector * np.conj(synch_freq_data_vector))).real / synch_freq_data_vector.shape[1]
                    synch_freq_data_normalized = synch_freq_data_vector / np.sqrt(synch_pow_est)

                    # from transmit antenna 1 only?
                    chan_freq0 = np.reshape(self.channel_freq[antenna_index, 0, self.synch_used_bins], (1, np.size(self.synch_used_bins)))

                    chan_freq = np.tile(chan_freq0, (1, self.num_of_synchs_and_synch_bins[0]))

                    bin_index = self.synch_used_bins[:, None]
                    cp_delays = np.array(range(self.len_CP + 1))[:, None]
                    delay_matrix = np.exp(1j * 2 * (np.pi / self.NFFT) * np.dot(bin_index, cp_delays.T))

                    tiled_delay_matrix = np.tile(delay_matrix, (self.num_of_synchs_and_synch_bins[0], 1))

                    # maybe replace index 0 with antenna_index
                    self.correlation_matrix = np.dot(np.conj(self.synch_reference)[None, :], np.dot(np.diag(synch_freq_data_normalized[0]), tiled_delay_matrix))
                    abs_correlation_matrix = abs(self.correlation_matrix[0, :])
                    correlation_value, correlation_index = abs_correlation_matrix.max(0), abs_correlation_matrix.argmax(0)
                    correlation_index = correlation_index - 1
                    correlation_val_vector[loop_count] = correlation_value
                    # print('no')
                    if correlation_value > 0.5 * synch_freq_data_normalized.shape[1] or self.correlation_observations > -1:
                        if correlation_index > np.ceil(0.75 * self.len_CP):
                            if self.correlation_observations == -1:  # 0
                                ptr_adj += np.ceil(0.5 * self.len_CP)
                                ptr_frame = loop_count * self.stride_value + self.start_sample + ptr_adj
                            elif self.correlation_observations < 5:
                                ptr_frame += np.ceil(0.5 * self.len_CP)

                            # Take FFT of the window
                            fft_vec = np.zeros((self.num_of_synchs_and_synch_bins[0], self.NFFT), dtype=complex)
                            for i in range(self.num_of_synchs_and_synch_bins[0]):
                                start = i * self.NFFT
                                fin = (i + 1) * self.NFFT
                                fft_vec[i, 0:self.NFFT] = np.fft.fft(self.rx_buffer_time_data[start: fin], self.NFFT)

                            synch_freq_data = fft_vec[:, self.synch_used_bins]
                            synch_freq_data_vector = np.reshape(synch_freq_data, (1, synch_freq_data.shape[0] * synch_freq_data.shape[1]))
                            synch_pow_est = sum(sum(synch_freq_data_vector * np.conj(synch_freq_data_vector))).real / synch_freq_data_vector.shape[1]
                            synch_freq_data_normalized = synch_freq_data_vector / np.sqrt(synch_pow_est)

                            # from transmit antenna 1 only?
                            chan_freq0 = np.reshape(self.channel_freq[antenna_index, 0, self.synch_used_bins],
                                                    (1, np.size(self.synch_used_bins)))

                            chan_freq = np.tile(chan_freq0, (1, self.num_of_synchs_and_synch_bins[0]))

                            bin_index = self.synch_used_bins[:, None]
                            cp_delays = np.array(range(self.len_CP + 1))[:, None]
                            delay_matrix = np.exp(1j * 2 * (np.pi / self.NFFT) * np.dot(bin_index, cp_delays.T))

                            tiled_delay_matrix = np.tile(delay_matrix, (self.num_of_synchs_and_synch_bins[0], 1))

                            # maybe replace index 0 with antenna_index
                            self.correlation_matrix = np.dot(np.conj(self.synch_reference)[None, :],
                                                             np.dot(np.diag(synch_freq_data_normalized[0]), tiled_delay_matrix))
                            abs_correlation_matrix = abs(self.correlation_matrix[0, :])
                            correlation_value, correlation_index = abs_correlation_matrix.max(0), abs_correlation_matrix.argmax(0)
                            correlation_val_vector[loop_count] = correlation_value

                        time_synch_ind = self.correlation_frame_index_value_buffer[antenna_index, max(self.correlation_observations, 1), 0]
                        # print("Current Time Synch Index: ", time_synch_ind)
                        if ptr_frame - time_synch_ind > (2 * self.len_CP + self.NFFT) or self.correlation_observations == -1:
                            self.correlation_observations += 1

                            self.correlation_frame_index_value_buffer[antenna_index, self.correlation_observations, 0] = ptr_frame
                            self.correlation_frame_index_value_buffer[antenna_index, self.correlation_observations, 1] = correlation_index
                            self.correlation_frame_index_value_buffer[antenna_index, self.correlation_observations, 2] = correlation_value

                            corrected_ptr[pattern_count % tap_delay] = sum(self.correlation_frame_index_value_buffer[antenna_index, self.correlation_observations, 0:2])
                            symbol_count[pattern_count % tap_delay] = pattern_count * sum(sys_model.synch_data)  # No need for +1 on lhs
                            pattern_count += 1

                            symbol_count_current = symbol_count[0:min(self.correlation_observations, tap_delay)]
                            symbol_count_lookahead = np.concatenate((symbol_count_current, np.atleast_1d(pattern_count * sum(sys_model.synch_data))))
                            x_symbol_count_lookahead = np.zeros((len(symbol_count_lookahead), 2))
                            x_symbol_count_lookahead[:, 0] = np.ones(len(symbol_count_lookahead))
                            x_symbol_count_lookahead[:, 1] = symbol_count_lookahead

                            if self.correlation_observations > 3:
                                y_time_series_current_ptr = corrected_ptr[0:min(tap_delay, self.correlation_observations)]
                                # print(y_time_series_current_ptr)
                                x_symbol_count_current = np.zeros((len(symbol_count_current), 2))
                                x_symbol_count_current[:, 0] = np.ones(len(symbol_count_current))
                                x_symbol_count_current[:, 1] = symbol_count_current

                                m_c_coefficients = np.linalg.lstsq(x_symbol_count_current, y_time_series_current_ptr, rcond=None)[0]
                                # print(m_c_coefficients)

                            # recovered data with delay removed - DataRecov in MATLAB code
                            input_data_freq_normalized = np.dot(np.diag(synch_freq_data_normalized[0]), tiled_delay_matrix[:, correlation_index])  # -1

                            h_est1 = np.zeros((self.NFFT, 1), dtype=complex)
                            # TmpV1 in MATLAB code
                            input_data_freq_rotated = (input_data_freq_normalized * np.conj(self.synch_reference)) / (1 + (1 / self.SNR))

                            h_est00 = np.reshape(input_data_freq_rotated, (input_data_freq_rotated.shape[0], self.num_of_synchs_and_synch_bins[0]))
                            h_est0 = h_est00.T

                            channel_estimate_avg_across_synchs = np.sum(h_est0, axis=0) / (self.num_of_synchs_and_synch_bins[0])

                            h_est1[self.synch_used_bins, 0] = channel_estimate_avg_across_synchs
                            # print("Correlation Obs:", self.corr_obs)
                            # print("Shape of Est_Chan_Freq_P: ", self.est_chan_freq_n.shape)
                            self.est_chan_freq_p[antenna_index, self.correlation_observations, 0:len(h_est1)] = h_est1[:, 0]
                            self.est_chan_freq_n[antenna_index, self.correlation_observations, 0:len(channel_estimate_avg_across_synchs)] = channel_estimate_avg_across_synchs

                            # if sys_model.diagnostic == 1 and loop_count == 0:
                            #     xax = np.array(range(0, self.NFFT)) * sys_model.fs / self.NFFT
                            #     yax1 = 20 * np.log10(abs(h_est1 + 1 * 10 ** -10))
                            #     yax2 = 20 * np.log10(abs(np.fft.fft(chan_q, self.NFFT)))
                            #
                            #     plt.plot(xax, yax1, 'r')
                            #     plt.plot(xax, yax2, 'm_c_coefficients')
                            #     plt.show()

                            h_est_time = np.fft.ifft(h_est1[:, 0], self.NFFT)
                            self.est_chan_impulse[antenna_index, self.correlation_observations, 0:len(h_est_time)] = h_est_time

                            h_est_ext = np.tile(channel_estimate_avg_across_synchs, (1, self.num_of_synchs_and_synch_bins[0])).T
                            # print("equalized synch")

                            synch_equalized = (input_data_freq_normalized * np.conj(h_est_ext[:, 0])) / ((np.conj(h_est_ext[:, 0]) * h_est_ext[:, 0]) + (1 / self.SNR))
                            self.est_synch_freq[antenna_index, self.correlation_observations, 0:len(self.synch_used_bins) * self.num_of_synchs_and_synch_bins[0]] = synch_equalized
                            # print("SYNCH EQ: ", self.est_synch_freq)
                            # if sys_model.diagnostic == 1 and loop_count == 0:
                            #     plt.plot(synch_equalized.real, synch_equalized.imag, '.')
                            #     plt.show()

                loop_count += 1
                # print(loop_count)

    def rx_data_demod(self):
        if self.num_ant_txrx == 1:
            antenna_index = 0  # Just an antenna index
            # BIG CHANGE HERE!!!!!!!!!!!
            for corr_obs_index in range(self.correlation_observations):
                for data_symbol_index in range(self.synch_data_pattern[1]):
                    if sum(self.correlation_frame_index_value_buffer[antenna_index, corr_obs_index, :]) + self.NFFT < self.input_time_series_data.shape[1]:
                        data_ptr = int(self.correlation_frame_index_value_buffer[antenna_index, corr_obs_index, 0] + (data_symbol_index + 1) * self.rx_buff_len)
                        self.rx_buffer_time_data = self.input_time_series_data[antenna_index, data_ptr: data_ptr + self.NFFT]  # -1

                        fft_vec = np.fft.fft(self.rx_buffer_time_data, self.NFFT)
                        # print('Frequency Data after RX: ', fft_vec[self.used_bins_data])
                        input_data_freq = fft_vec[self.used_bins_data]

                        input_pow_est = sum(input_data_freq * np.conj(input_data_freq)) / len(input_data_freq)

                        input_data_freq_normalized = input_data_freq / (np.sqrt(input_pow_est) + 1e-10)
                        # print('Data Recovered after RX and Bin Selection: ', input_data_freq_normalized)
                        if self.param_est == 'Estimated':
                            # print('hello')
                            channel_estimate_avg_across_synchs = self.est_chan_freq_p[antenna_index, corr_obs_index, self.used_bins_data]
                        elif self.param_est == 'Ideal':
                            # print('bye')
                            channel_estimate_avg_across_synchs = self.h_f[antenna_index, 0, :]

                        del_rotate = np.exp(1j * 2 * (np.pi / self.NFFT) * self.used_bins_data * self.correlation_frame_index_value_buffer[antenna_index, corr_obs_index, 1])
                        input_data_freq_rotated = np.dot(np.diag(input_data_freq_normalized), del_rotate)
                        # print('Frequency Data after RX: ', input_data_freq_rotated)
                        input_data_freq_equalized = (input_data_freq_rotated * np.conj(channel_estimate_avg_across_synchs)) / ((np.conj(channel_estimate_avg_across_synchs) * channel_estimate_avg_across_synchs) + (1 / self.SNR))
                        # print('Frequency Data (Equalized) after RX: ', input_data_freq_equalized)
                        # print('corr_obs_index * synch_dat * data+symb', corr_obs_index * self.synch_data[1] + data_symbol_index)
                        # print('0:len(self.used_bins_data', len(self.used_bins_data))
                        # print("self.synch_data[1]", self.synch_data[1])
                        # print("corr_obs_index", corr_obs_index)
                        # print("data_symbol_index", data_symbol_index)
                        # print("Data_Equalized: ", input_data_freq_equalized)
                        if corr_obs_index * self.synch_data_pattern[1] + data_symbol_index == 0:
                            self.est_data_freq[antenna_index, corr_obs_index, :] = self.est_data_freq[antenna_index, corr_obs_index, :] + input_data_freq_equalized
                        else:
                            self.est_data_freq = np.vstack((self.est_data_freq[antenna_index, :], input_data_freq_equalized))
                            self.est_data_freq = self.est_data_freq[np.newaxis, :, :]
                        data = self.est_data_freq[antenna_index, corr_obs_index, 0:len(self.used_bins_data)]
                        p_est1 = sum(data * np.conj(data)) / len(data)
                        self.est_data_freq[antenna_index, corr_obs_index * self.synch_data_pattern[1] + data_symbol_index, 0:len(self.used_bins_data)] /= (np.sqrt(p_est1) + 1e-10)

        elif self.num_ant_txrx == 2 and self.MIMO_method == 'STCode':
            print('STCode currently not supported')
            exit(0)
        elif self.num_ant_txrx == 2 and self.MIMO_method == 'SpMult':
            print('SpMult currently not supported')
            exit(0)
