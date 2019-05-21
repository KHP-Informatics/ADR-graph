#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:36:52 2019

@author: danielbean
"""

from utils import merge_files as utils

file_a = "demo/merge_a.csv"
file_b = "demo/merge_b.csv"
new_rel_type = "linked_gene_in_pathway"

res = utils.create_indirect_rels(file_a, file_b, new_rel_type)

res.to_csv('demo/merged.csv', index=False)