#!/usr/bin/python

import sys

m=open("maternal.reads",'w+')
p=open("paternal.reads","w+")
for line in sys.stdin:
	chr,start,end,id,seq,snp_chr,snp_start,snp_end,snp_seq=line.rstrip().split('\t')
	snp_idx=int(snp_start)-int(start)-1
	#if snp_idx >= len(seq):
	#	print line.rstrip()
	snp=seq[snp_idx]
	p_snp,m_snp=snp_seq.split('/')
	if snp==p_snp:
		p.write(line)
	elif snp==m_snp:
		m.write(line)
	else:
		print line.rstrip()+"\t"+snp+"\t"+p_snp+"\t"+m_snp
