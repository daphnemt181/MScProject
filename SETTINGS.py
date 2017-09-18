# -*- coding: utf-8 -*-

"""
Settings, paths and various constants that doesn't change.
"""

import os
import sys
import string

# paths

curfilePath = os.path.abspath(__file__)
cur_dir = os.path.abspath(os.path.join(curfilePath, os.pardir)) # this will return current directory in which python file resides.
PROJECT_ROOT = cur_dir # Parent directory

PATHS = {
    "project_root": PROJECT_ROOT,
    "data_raw_dir": os.path.join(PROJECT_ROOT, 'data', 'raw'),
    "data_processed_dir": os.path.join(PROJECT_ROOT, 'data', 'processed'),
    "data_interim_dir": os.path.join(PROJECT_ROOT, 'data', 'interim'),
    "data_external_dir": os.path.join(PROJECT_ROOT, 'data', 'external'),
    "save_dir": os.path.join(PROJECT_ROOT, 'models'),
    "fig_dir": os.path.join(PROJECT_ROOT, '..', 'Dissertation', 'latex', 'ucl-latex-thesis-templates', 'figures'),
    "src_vis": os.path.join(PROJECT_ROOT, 'src', 'visualization'),
    "src_utils": os.path.join(PROJECT_ROOT, 'src', 'tools'),
    "src_models": os.path.join(PROJECT_ROOT, 'src', 'models'),
    "src_data": os.path.join(PROJECT_ROOT, 'src', 'data'),
    "src_external": os.path.join(PROJECT_ROOT, 'src', 'external'),
    "src_nn": os.path.join(PROJECT_ROOT, 'src', 'nn')
}

# valid vocabularies (unicode and ascii where appropriate)
valid_vocab_fr = u''.join([u'F', u'', string.ascii_lowercase, u"âêîôûàèùéëïüç,.`'?!E "])
valid_vocab_en = u''.join([u'F', u'', string.ascii_lowercase, u",.`'?!E "])
