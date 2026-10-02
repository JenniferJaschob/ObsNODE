#%%
library(ricu)

data_path = '/Users/jaschob/Desktop/Amsterdam/AmsterdamPhil/Hertzsch_Data/'

import_src('aumc',data_dir=data_path+'AmsterdamUMCdb-v1.0.2')

attach_src('aumc',data_dir=data_path+'AmsterdamUMCdb-v1.0.2')

sofa=load_concepts('sofa','aumc')
write.csv(sofa,data_path+'Sofascore.csv')

sepsis3=load_concepts('sep3','aumc')
write.csv(sofa,path_data+'Documents/Amsterdamdata/Sep3.csv')

