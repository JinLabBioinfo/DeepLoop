#!/usr/bin/perl

use strict;
my $usage = 	"Usage:./reads_2_trans_frag_loop.pl <frag_bed> <read_length> <trans_outfile> <reads file>\n".
		"\tThis program takes reads summary txt file and output the cis- fragment looping files categoried based on inward, outward and samestrand reads.\n";

my ($frag_bed, $len, $out_trans_file, $reads_file) = @ARGV;
if(not defined $len){
	die($usage);
}

if(not defined $reads_file){
	$reads_file = "-";
}

if(not defined $out_trans_file){
        $out_trans_file = "-";
}

open(OUT_trans, ">$out_trans_file");

my $frag_HASH;
my $frag_loc;
open(IN, $frag_bed);
while(my $line = <IN>){
	chomp $line;
	my ($chr, $beg, $end, $id) = split "\t", $line;
	$frag_loc->{$id} = join(":", $beg, $end);
	for(my $ind = int($beg/10000); $ind <= int($end/10000); $ind++){
		push @{$frag_HASH->{$chr}->{$ind}}, $id;
	}
}
close(IN);

open(FH, $reads_file) || die("Error: Cannot open file $reads_file!\n");
while(my $line = <FH>){
	chomp $line;
	my ($chr1, $beg1, $str1, $chr2, $beg2, $str2) = split "\t", $line;
	if($chr1 eq $chr2){
		next;
	}else{
		my $frag1 = find_frag($frag_HASH->{$chr1}, $frag_loc, $beg1, $str1, $len);
		my $frag2 = find_frag($frag_HASH->{$chr2}, $frag_loc, $beg2, $str2, $len);

		if((!$frag1) || (!$frag2) || ($frag1 eq $frag2)){
			next;
		}
		print OUT_trans join("\t", $frag1, $frag2)."\n";
	}
}
close(FH);

close(OUT_trans);

exit;

########################################################################
sub find_frag{
	my ($hash, $frag_loc, $beg, $strand, $len) = @_;
	my $end = $beg + $len - 1;
	my $left = $beg;
	my $right = $end;
	if($strand eq "+"){
		$right -= 16;
	}else{
		$left += 16;
	}
	
	for(my $ind = int($left/10000); $ind <= int($right/10000); $ind++){
		foreach my $fid (@{$hash->{$ind}}){
			my ($f_beg, $f_end) = split ":", $frag_loc->{$fid};

			if($left >= $f_beg && $right <= $f_end){
				if($strand eq "+" && $left >= ($f_end - 500)){
                                        return $fid;
                                }elsif($strand eq "-" && $right <= ($f_beg + 500)){
                                        return $fid;
                                }else{
                                }
			}
		}
	}
	return 0;
}
