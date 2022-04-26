#!/usr/bin/perl

use strict;
my $usage = 	"Usage:./reads_2_cis_frag_loop.pl <frag_bed> <read_length> <inward_outfile> <outward_outfile> <samestrand_outfile> <count_summary_file> <expt> <reads file>\n".
		"\tThis program takes reads summary txt file and output the cis- fragment looping files categoried based on inward, outward and samestrand reads.\n";

my ($frag_bed, $len, $out_cis_file, $reads_file) = @ARGV;
if(not defined $reads_file){
	$reads_file = "-";
}
open(OUT_cis, ">$out_cis_file");
my $frag_HASH;
my $frag_loc;
my $chr_hash;
open(IN, $frag_bed);
while(my $line = <IN>){
	chomp $line;
	my ($chr, $beg, $end, $id) = split "\t", $line;
	$chr_hash->{$id}=$chr;
	$frag_loc->{$id} = join(":", $beg, $end);
	for(my $ind = int($beg/10000); $ind <= int($end/10000); $ind++){
		push @{$frag_HASH->{$chr}->{$ind}}, $id;
	}
}
close(IN);
open(FH, $reads_file) || die("Error: Cannot open file $reads_file!\n");
while(my $line = <FH>){
	chomp $line;
	my ($tmp_id,$chr1, $beg1, $chr2, $beg2) = split "\t", $line;
	my $frag1 = find_frag($frag_HASH->{$chr1}, $frag_loc, $beg1, $len);
	my $frag2 = find_frag($frag_HASH->{$chr2}, $frag_loc, $beg2, $len);
	my $chr1=$chr_hash->{$frag1};
	my $chr2=$chr_hash->{$frag2};
	if((!$frag1) || (!$frag2) || ($frag1 eq $frag2) || ($chr1 ne $chr2)){
                        next;
        }else{
#		if($beg1 gt $beg2){
#			print OUT_cis join("\t",$frag2,$frag1)."\n";
#		}else{
			print OUT_cis join("\t",$frag1,$frag2)."\n";
#		}
	}
}
close(FH);
close(OUT_cis);
exit;

########################################################################
sub find_frag{
	my ($hash, $frag_loc, $beg, $len) = @_;
	my $end = $beg + $len - 1;
	my $left = $beg;
	my $right = $end;
	for(my $ind = int($left/10000); $ind <= int($right/10000); $ind++){
		foreach my $fid (@{$hash->{$ind}}){
			my ($f_beg, $f_end) = split ":", $frag_loc->{$fid};
			if($left >= $f_beg && $right <= $f_end){
				return $fid;
			}
		}
	}
	return 0;
}
