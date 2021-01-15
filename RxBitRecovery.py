import numpy as np

class RxBitRecovery:
    def __init__(self, est_data_freq, used_bins_data, corr_obs, symbol_pattern, binary_info, num_symbols):

        self.data_in = est_data_freq
        self.used_bins_data = used_bins_data
        self.corr_obs = corr_obs
        self.symbol_pattern = symbol_pattern
        self.binary_info = binary_info
        self.num_symbols = num_symbols


        self.mapping_type = 'QPSK'
        self.EC_code = None
        self.BER_calc = None
        self.est_bits = None
        self.symbol_count = 0

        self.softbit0 = np.zeros((self.data_in.shape[0], 2*self.data_in.shape[0]*self.data_in.shape[1]*self.data_in.shape[2]))
        self.softbit1 = np.zeros((self.data_in.shape[0], 2*self.data_in.shape[0]*self.data_in.shape[1]*self.data_in.shape[2]))
        self.hardbit = np.zeros((self.data_in.shape[0], 2*self.data_in.shape[0]*self.data_in.shape[1]*self.data_in.shape[2]))

        self.raw_BER = np.zeros(self.data_in.shape[0])
        self.rawerror_rate = np.zeros(self.data_in.shape[0])

    def soft_bit_recovery(self):
        self.symbol_count += 1

        data_shp0 = self.data_in.shape[0]
        data_shp1 = self.data_in.shape[1]
        data_shp2 = self.data_in.shape[2]

        num_el_data = data_shp0 * data_shp1 * data_shp2

        if self.mapping_type == 'BPSK':
            print('BPSK demod not currently implemented')
            exit(0)
        elif self.mapping_type == 'QPSK':
            cmplx_phsrs = np.exp(1j*2*(np.pi/8)*np.array([1, -1, 3, 5]))

            # permute the dimensions of data_in
            data_rearrange = np.transpose(self.data_in, [2, 1, 0])

            data0 = np.reshape(data_rearrange.T, (int(num_el_data/data_shp0), int(data_shp0)))

            for loop in range(data_shp0):
                iq_data = data0[:, loop]

                llrp0 = np.zeros(2*data_shp1*len(self.used_bins_data))  # 2 for QPSK - each IQ phasor corresponds to 2 bits
                llrp1 = np.zeros(2*data_shp1*len(self.used_bins_data))

                cmplx_phsrs_ext = np.tile(cmplx_phsrs, (len(iq_data), 1)).T
                data_ext = np.tile(iq_data, (4, 1))

                dist = abs(data_ext - cmplx_phsrs_ext)

                dmin = np.min(dist, 0)
                dmin_ind = np.argmin(dist, 0)

                dz = cmplx_phsrs[dmin_ind]

                ez = iq_data  -  dz

                sigma00 = np.mean(abs(dmin))
                sigma0 = np.sqrt(0.5 * sigma00 * sigma00)
                d_factor = 1/sigma0 ** 2

                K = 2/np.sqrt(2)

                for kk in range(len(iq_data)):
                    if dz[kk].real >= 0 and dz[kk].imag >= 0:
                        llrp0[2*kk] = -0.5 * abs(ez[kk].real)
                        llrp1[2*kk] = -0.5 * (K - abs(ez[kk].real))

                        llrp0[2*kk + 1] = -0.5 * abs(ez[kk].imag)
                        llrp1[2*kk + 1] = -0.5 * (K - abs(ez[kk].imag))

                    elif dz[kk].real <= 0 and dz[kk].imag >= 0:
                        llrp0[2 * kk] = -0.5 * abs(ez[kk].real)
                        llrp1[2 * kk] = -0.5 * (K - abs(ez[kk].real))

                        llrp1[2 * kk + 1] = -0.5 * abs(ez[kk].imag)
                        llrp0[2 * kk + 1] = -0.5 * (K - abs(ez[kk].imag))
                    elif dz[kk].real <= 0 and dz[kk].imag <=0:
                        llrp1[2 * kk] = -0.5 * abs(ez[kk].real)
                        llrp0[2 * kk] = -0.5 * (K - abs(ez[kk].real))

                        llrp1[2 * kk + 1] = -0.5 * abs(ez[kk].imag)
                        llrp0[2 * kk + 1] = -0.5 * (K - abs(ez[kk].imag))
                    elif dz[kk].real >= 0 and dz[kk].imag <= 0:
                        llrp1[2 * kk] = -0.5 * abs(ez[kk].real)
                        llrp0[2 * kk] = -0.5 * (K - abs(ez[kk].real))

                        llrp0[2 * kk + 1] = -0.5 * abs(ez[kk].imag)
                        llrp1[2 * kk + 1] = -0.5 * (K - abs(ez[kk].imag))

                llrp0 *= d_factor
                llrp1 *= d_factor

                # softbit00 = np.zeros(np.size(llrp0))
                # softbit11 = np.zeros(np.size(llrp1))

                self.softbit0[loop, 1::2] = llrp0[0::2]
                self.softbit0[loop, 0::2] = llrp0[1::2]

                self.softbit1[loop, 1::2] = llrp1[0::2]
                self.softbit1[loop, 0::2] = llrp1[1::2]

                self.hardbit[loop, :] = np.ceil(0.5*(np.sign(self.softbit1[loop, :] - self.softbit0[loop, :]) + 1))

                actual_bits = self.binary_info[loop, self.binary_info.shape[1] - self.hardbit.shape[1]:]
                est_bits = self.hardbit[loop, :]

                # Number of errors
                self.raw_BER[loop] = sum(np.logical_xor(actual_bits, est_bits).astype(int))
                # print(self.raw_BER[loop])
                # Probability
                self.rawerror_rate[loop] = self.raw_BER[loop] / self.binary_info.shape[1]



                dbg77 = 1










