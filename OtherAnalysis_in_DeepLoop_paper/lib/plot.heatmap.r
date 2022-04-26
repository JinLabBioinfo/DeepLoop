#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

make_symmetric <- function(input_matrix){
	m=matrix(0,nrow(input_matrix),ncol(input_matrix))
	for(i in 1:(nrow(input_matrix)-1)){
		for(j in (i+1):nrow(input_matrix)){
			m[i,j]=0.5*(input_matrix[i,j]+input_matrix[j,i])
			m[j,i]=m[i,j]
		}			
	}
	return(m)
}

file<-args[1]
data<-as.matrix(read.table(file,stringsAsFactors=FALSE))
diag(data)<-0
data[which(data<0)]<-0
n <- 20
col <- rgb(1,(n-2):0/(n-1),(n-2):0/(n-1))
png(paste(file,".png",sep=''))
par(mar=c(0,0,0,0))
if(max(data)<2){
	breaks <- seq(1.001,max(data),(max(data)-1.001)/19)
}else{
	step <- (quantile(data, prob=0.98)-1)/18
	up <- quantile(data, prob=0.98)+0.011
	if(up<2){
		up <- 2
		step <- 0.999/18
	}
	breaks <- c(seq(1.001,up,step),max(data))
}
image(data[,ncol(data):1], zlim=c(1, 50), col=col, breaks=breaks,xaxt="n",yaxt="n",xlab="",ylab="")
dev.off()
