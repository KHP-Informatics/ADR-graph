# EdgePrediction class documentation

Created on Mon May 20 18:20:01 2019

@author: danielbean


### EdgePrediction.utils.merge_files.create_indirect_rels(file_a, file_b, new_rel_type)
Create indirect relationships by merging files (a and b) that are already in the input
format required by the EdgePrediction algorithm. The target in file A must be the
source in file B.

E.g. for this path
(a_source) –> (a_target = b_source) –> (b_target)

create the relationship
(a_source) –> (b_target)


* **Returns**

    


* **Return type**

    pandas.DataFrame



* **Parameters**

    
    * **file_a** (*str*) – Path to file a, the source in this file will be the source in the output.


    * **file_b** (*str*) – Path to file b, the target in this file will be the target in the output.


    * **new_rel_type** (*str*) – Used to populate the “Relationship type” column of the output dataframe
