#!/usr/bin/perl
use strict;

my $bed = "-";
open(IN, $bed) || die("Error: Cannot open file $bed!\n");
while(my $line = <IN>){
	chomp $line;
	my $str1="+";
	my $str2="+";
	my @a = split "\t", $line;
	if($a[0]&16){
		$str1="-";
	}
	if($a[0]&32){
		$str2="-";
	}
	if($a[5] eq "="){
		$a[5]=$a[1];
	}
	print join("\t",$a[1],$a[2],$str1,$a[5],$a[6],$str2)."\n";        
}
close(IN);
exit;

