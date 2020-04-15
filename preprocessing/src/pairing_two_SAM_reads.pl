#!/usr/bin/perl  
use strict;

my $usage = 	"Usage:./pairing_two_SAM_reads.pl <sam1> <sam2>\n".
		"\tAfter mapping each end of a PE run independently, this program take the two SAM files and link them into one pair end SAM file\n";
my ($file1, $file2) = @ARGV;
if(not defined $file2){
	die($usage);
}

open (FILE1, $file1);
open (FILE2, $file2); 

while ((my $line1 = <FILE1>) and (my $line2 = <FILE2>)){
	chomp $line1; chomp $line2;
	while(substr($line1,0,1) eq '@'){
#		print $line1."\n";
		$line1 = <FILE1>; chomp $line1;
	}
	while(substr($line2,0,1) eq '@'){
#		print $line2."\n";
		$line2 = <FILE2>; chomp $line2;
	} 
	my ($id1, $flag1, $chr1, $pos1, $qal1, $cigar1, $mate_chr1, $mate_pos1, $size1, $seq1, @rest1) = split "\t", $line1;
	my ($id2, $flag2, $chr2, $pos2, $qal2, $cigar2, $mate_chr2, $mate_pos2, $size2, $seq2, @rest2) = split "\t", $line2;
	$id1 =~ s/\/.$//; $id2 =~ s/\/.$//;
		

	if($qal1 < 10 || $qal2 < 10) { next;}	
	if($id1 ne $id2){
		die("Error: The two sam files are not pair-end!\n");
	}
	$flag1 = $flag1 | 64; 				# read1 flag
	$flag2 = $flag2 | 128;				# read2 flag
	$flag1 = $flag1 | 1; $flag2 = $flag2 | 1; 	# flag say both reads are one of a pair
	if(($flag1 & 4) || ($flag2 & 4)) { next;}       # both reads must have alignment
	$flag1 = $flag1 | 2; $flag2 = $flag2 | 2;
	my ($strand1, $strand2) = ("+", "+");
	my ($beg1, $beg2) = ($pos1, $pos2);
	if($flag1 & 16) { $flag2 = $flag2 | 32; $strand1 = "-"; }	# mate of read2 is - strand
	if($flag2 & 16) { $flag1 = $flag1 | 32; $strand2 = "-"; }	# mate of read1 is - strand
	
	$mate_pos1 = $pos2; $mate_pos2 = $pos1;
	if($chr1 eq $chr2){
		$mate_chr1 = "="; $mate_chr2 = "=";
		my ($left1, $right1) = ($pos1, $pos1 + length($seq1));
		my ($left2, $right2) = ($pos2, $pos2 + length($seq2));
		$size1 = $pos2 - $pos1;
		if($size1 > 0){
			$size1 += length($seq2);
		}else{
			$size1 -= length($seq1);
		}
		$size2 = 0 - $size1;
	}else{
		$mate_chr2 = $chr1; $mate_chr1 = $chr2;
		$size1 = 0; $size2 = 0;
	}
	print join("\t", $id1, $flag1, $chr1, $pos1, $qal1, $cigar1, $mate_chr1, $mate_pos1, $size1, $seq1, @rest1)."\n";
	print join("\t", $id2, $flag2, $chr2, $pos2, $qal2, $cigar2, $mate_chr2, $mate_pos2, $size2, $seq2, @rest2)."\n";
}

close FILE1;
close FILE2;

exit;
