from __future__ import print_function
from __future__ import division

import os
import operator

import numpy
import torch

import nmt.utils as ut
from nmt.model import Model
import nmt.configurations as configurations


class Extractor(object):
    def __init__(self, args):
        super(Extractor, self).__init__()
        config = getattr(configurations, args.proto)()
        self.logger = ut.get_logger(config['log_file'])
        self.model_file = args.model_file

        var_list = args.var_list
        save_to = args.save_to

        if var_list is None:
            raise ValueError('Empty var list')

        if self.model_file is None or not os.path.exists(self.model_file):
            raise ValueError('Input file or model file does not exist')

        if not os.path.exists(save_to):
            os.makedirs(save_to)

        self.logger.info('Extracting these vars: {}'.format(', '.join(var_list)))

        model = Model(config)
        model.load_state_dict(torch.load(self.model_file))
        var_values = operator.attrgetter(*var_list)(model)

        if len(var_list) == 1:
            var_values = [var_values]

        for var, var_value in zip(var_list, var_values):
            var_path = os.path.join(save_to, var + '.npy')
            numpy.save(var_path, var_value.numpy())
