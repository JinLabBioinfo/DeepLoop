#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

outdirfile <- as.character(args[1])
layout.rowN <- as.numeric(args[2])
layout.colN <- as.numeric(args[3])
inputfilelis<- as.character(args[4:length(args)])

plot_heatmap <- function(file){
	data<-as.matrix(read.table(file,stringsAsFactors=FALSE))
	data[which(data<0)]<-0
	n <- 20
	col <- rgb(1,(n-2):0/(n-1),(n-2):0/(n-1))
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
}
png(paste(outdirfile,".plot.png",sep=''),150*layout.colN,150*layout.rowN)
#nf <- layout(matrix(c(1:6),2,3,byrow=TRUE), widths=c(rep(4,3)), heights=c(rep(4,2), TRUE))
par(mar=c(1,1,1,1))
par(mfrow=c(layout.rowN, layout.colN))
for(inputfile in inputfilelis){
	plot_heatmap(inputfile)
}

dev.off()
