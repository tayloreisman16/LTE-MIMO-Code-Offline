#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 gr-RX_OFDM author.
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#

import numpy as np


class FrequencyOffsetTuner:
    def __init__(self):
        self.quad1 = np.array([np.sqrt(2) / 2 + (np.sqrt(2) / 2) * 1j], dtype=np.complex64)
        self.quad2 = np.array([-np.sqrt(2) / 2 + (np.sqrt(2) / 2) * 1j], dtype=np.complex64)
        self.quad3 = np.array([-np.sqrt(2) / 2 - (np.sqrt(2) / 2) * 1j], dtype=np.complex64)
        self.quad4 = np.array([np.sqrt(2) / 2 - (np.sqrt(2) / 2) * 1j], dtype=np.complex64)

    def constellation_deviation(self, iq_data):
        mag_min_buffer = np.zeros([1])
        clear_condition = 0
        iq_data = iq_data[0]
        quad1_min = np.zeros([1])
        quad2_min = np.zeros([1])
        quad3_min = np.zeros([1])
        quad4_min = np.zeros([1])

        for constellation in iq_data:
            # print(constellation)
            if constellation != (0 + 0j):
                temp = np.min([np.absolute(self.quad1 - constellation), np.absolute(self.quad2 - constellation),
                           np.absolute(self.quad3 - constellation), np.absolute(self.quad4 - constellation)])
                temp_arg = np.argmin([np.absolute(self.quad1 - constellation), np.absolute(self.quad2 - constellation),
                           np.absolute(self.quad3 - constellation), np.absolute(self.quad4 - constellation)])
                # if temp_arg == 0:
                #     quad1_min = np.append(quad1_min, temp)
                # if temp_arg == 1:
                #     quad2_min = np.append(quad2_min, temp)
                # if temp_arg == 2:
                #     quad3_min = np.append(quad3_min, temp)
                # if temp_arg == 3:
                #     quad4_min = np.append(quad4_min, temp)
                mag_min = temp
                if mag_min < 2.0:
                    mag_min_buffer = np.append(mag_min_buffer, mag_min)
                    if clear_condition is 0:
                        mag_min_buffer = np.delete(mag_min_buffer, 0)

                        quad1_min = np.delete(quad1_min, 0)
                        quad2_min = np.delete(quad2_min, 0)
                        quad3_min = np.delete(quad3_min, 0)
                        quad4_min = np.delete(quad4_min, 0)

                        clear_condition = 1
        mu = np.average(mag_min_buffer)
        print("Mu: ", mu)
        sigma = np.std(mag_min_buffer)
        print("Sigma: ", sigma)

        return mu, sigma
