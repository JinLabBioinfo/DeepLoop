#!/usr/bin/python

import sys

start=int(sys.argv[1])-1
length=int(sys.argv[2])

i=1
for line in sys.stdin:
	if i==1:
		print line.rstrip()
	if i==2:
		print line[start:(start+length)]
	if i==3:
		#line=line[start:(start+length)]
                print line.rstrip()
        if i==4:
		print line[start:(start+length)]
		#	print line.rstrip()
	i=i+1
	if i>4:
		i=1

