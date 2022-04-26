#!/usr/bin/python

import sys

for line in sys.stdin:
	chr,start,end,id,seq,cigar=line.rstrip().split('\t')
	l=""
	arr=[]
	for x in cigar:
		if x.isdigit():
			l=l+x
		else:
			arr.append((x,int(l)))	
			l=""
	result=""
	ind=0
	temp=""
	for x in arr:
		char,num=x[0],x[1]
		temp=temp+str(x[1])+char
		if char=="M":
			result=result+seq[ind:ind+num]
			ind=ind+num
		elif char=="I":
			ind=ind+num
		elif char=="D":
			result=result+"N"*num
		else:	
			result="error"
			break
	#print chr+"\t"+start+"\t"+end+"\t"+id+"\t"+result
	print line.rstrip()+"\t"+result
