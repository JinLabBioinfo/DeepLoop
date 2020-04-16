#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
options(scipen = 999)

model_dir=args[1]
data_depth=args[2]

model_lis=list.files(model_dir,pattern="\\.json$")
if(length(which(grepl("LoopDenoise",model_lis))>0)){
	model_lis=model_lis[-which(grepl("LoopDenoise",model_lis))]
}
depth=c()
for(i in 1:length(model_lis)){
	depth=c(depth,unlist(strsplit(model_lis[i],split='.json'))[1])
}
depth_name=depth
ind1=which(grepl("M",depth))
ind2=which(grepl("k",depth))
for(i in ind1){
	depth[i]=as.numeric(as.character(unlist(strsplit(depth[i],split='M'))[1]))*1000000
}

for(i in ind2){
        depth[i]=as.numeric(as.character(unlist(strsplit(depth[i],split='k'))[1]))*1000
}
matching_model=depth_name[which.min(abs(as.numeric(data_depth)-as.numeric(depth)))]
cat(matching_model)
