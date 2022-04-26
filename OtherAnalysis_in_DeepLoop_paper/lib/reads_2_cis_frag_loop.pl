#!/usr/bin/perl

use strict;
my $usage = 	"Usage:./reads_2_cis_frag_loop.pl <frag_bed> <read_length> <inward_outfile> <outward_outfile> <samestrand_outfile> <count_summary_file> <expt> <reads file>\n".
		"\tThis program takes reads summary txt file and output the cis- fragment looping files categoried based on inward, outward and samestrand reads.\n";

my ($frag_bed, $len, $out_inward_file, $out_outward_file, $out_samestrand_file, $count_summary_file, $expt, $reads_file) = @ARGV;
if(not defined $expt){
	die($usage);
}

if(not defined $reads_file){
	$reads_file = "-";
}


open(OUT_inward, ">$out_inward_file");
open(OUT_outward, ">$out_outward_file");
open(OUT_samestrand, ">$out_samestrand_file");

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

my $total_inter = 0;
my $total_intra = 0;
my $total_samestrand = 0;
my $count_samestrand = 0;
my $count_inward = 0;
my $count_outward = 0;
open(FH, $reads_file) || die("Error: Cannot open file $reads_file!\n");
while(my $line = <FH>){
	chomp $line;
	my ($chr1, $beg1, $str1, $chr2, $beg2, $str2) = split "\t", $line;
	if($chr1 ne $chr2){
		$total_inter ++;
		next;
	}else{
		$total_intra ++;
		if($str1 eq $str2){
                        $total_samestrand ++;
		}
		my $frag1 = find_frag($frag_HASH->{$chr1}, $frag_loc, $beg1, $str1, $len);
		my $frag2 = find_frag($frag_HASH->{$chr2}, $frag_loc, $beg2, $str2, $len);

		if((!$frag1) || (!$frag2) || ($frag1 eq $frag2)){
			next;
		}
		if($str1 eq $str2){
			$count_samestrand ++;
			print OUT_samestrand join("\t", $frag1, $frag2)."\n";
		}elsif(($str1 eq "+" && $beg1 < $beg2) || ($str2 eq "+" && $beg2 < $beg1)){
			$count_inward ++;
			print OUT_inward join("\t", $frag1, $frag2)."\n";
		}else{
			$count_outward ++;
			print OUT_outward join("\t", $frag1, $frag2)."\n";
		}
	}
}
close(FH);

close(OUT_inward);
close(OUT_outward);
close(OUT_samestrand);

open(SUMM, ">>$count_summary_file");
print SUMM join("\t", $expt, $total_inter/2, $total_intra/2, $total_samestrand/2, $count_samestrand/2, $count_inward/2, $count_outward/2)."\n";
close(SUMM);

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
