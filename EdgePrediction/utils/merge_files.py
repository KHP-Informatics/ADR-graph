"""
Created on Mon May 20 18:20:01 2019

@author: danielbean
"""

#merge input files
import pandas as pd
from edges import Directed


def create_indirect_rels(file_a, file_b, new_rel_type):
     """utils.create_indirect_rels
     
     Create indirect relationships by merging files (a and b) that are already in the input
     format required by the EdgePrediction algorithm. The target in file A must be the
     source in file B. 

     E.g. for this path
     (a_source) --> (a_target = b_source) --> (b_target)

     create the relationship
     (a_source) --> (b_target)
     
     Returns
     ----------
     pandas.DataFrame
     

     Parameters
     ----------
     file_a : str
          Path to file a, the source in this file will be the source in the output.
          
     file_b : str
          Path to file b, the target in this file will be the target in the output.

     new_rel_type : str
          Used to populate the "Relationship type" column of the output dataframe
     
     """
     df_a = pd.read_csv(file_a)
     df_b = pd.read_csv(file_b)
     
     edges_b = Directed({})
     for index, row in df_b.iterrows():
          edges_b.add(row['Source node name'], row['Target node name'])
     
     edges_a = Directed({})
     for index, row in df_a.iterrows():
          edges_a.add(row['Source node name'], row['Target node name'])
     
     out_edges = Directed({})
     for source_a in edges_a.edges:
          for target_a in edges_a.edges[source_a]:
               if target_a in edges_b.edges:
                    targets_from_b = edges_b.edges[target_a]
                    for tgt in targets_from_b:
                         out_edges.add(source_a, tgt)
     out_rows = [{'Source node name':x[0], 'Target node name': x[1]} for x in out_edges.list()]
     
     out_df = pd.DataFrame(out_rows)
     out_df['Source node type'] = df_a['Source node type'].iloc[0]
     out_df['Target node type'] = df_b['Target node type'].iloc[0]
     out_df['Relationship type'] = new_rel_type
     return out_df

if __name__ == "__main__":

     file_a = "demo/merge_a.csv"
     file_b = "demo/merge_b.csv"
     new_rel_type = "linked_gene_in_pathway"
     
     res = create_indirect_rels(file_a, file_b, new_rel_type)
     
     res.to_csv('demo/merged.csv', index=False)

