#!/usr/bin/perl

use strict;
use List::MoreUtils qw(uniq);
my $usage = "./generate_matrix.by.loc.pl <anchor_bed> <anchor_file> <chr> <start> <end> <output_dir>";

my ($anchor_bed, $anchor_file, $target_chr, $start, $end, $output_dir) = @ARGV;

my $anchor_loc_hash;
open(IN, $anchor_bed);
while(my $line = <IN>){
        chomp $line;
        my ($chr, $beg, $tmp_end, $A) = split "\t", $line;
	$anchor_loc_hash->{$chr}->{$A} = "$beg:$tmp_end";
}
close(IN);

my $start_A=get_id($start,$anchor_loc_hash->{$target_chr});
my $end_A=get_id($end,$anchor_loc_hash->{$target_chr});
my @start_str = split /\_/, $start_A;
my $start_num=@start_str[1];
my @end_str = split /\_/, $end_A;
my $end_num=@end_str[1];

my @anchor_lis = ();
my $loop_result;

open(IN, $anchor_file);
while(my $line = <IN>){
        chomp $line;
        my ($A1, $A2, $obs, $expt) = split "\t", $line;
	my $ratio=($obs+5)/($expt+5);
        my @str1 = split /\_/, $A1;
        my $a1=@str1[1];
        my @str2 = split /\_/, $A2;
        my $a2=@str2[1];
	if($a1>=$start_num && $a1<=$end_num){
		if($a2>=$start_num && $a2<=$end_num){
			$loop_result->{$a1}->{$a2}=join(",",$obs,$expt,$ratio);
			push @anchor_lis, $a1;
			push @anchor_lis, $a2;
		}
	}
}
close(IN);

my $anchor_idx;
my $len=0;
my $i = 0;
my @anchor_lis = uniq(@anchor_lis);
my @anchor_lis = sort @anchor_lis;
foreach (@anchor_lis){
	$anchor_idx->{$_} = $i;
	$i ++;
	$len ++;
}
my $size=@anchor_lis;
########################################################
my @ratio_matrix;
for(my $i = 0; $i < $len; $i ++){
        my @array = ();
        for(my $j = 0; $j < $len; $j ++){
                push @array, 0;
        }
        push @ratio_matrix, \@array;

}
#####################
my @raw_matrix;
for(my $i = 0; $i < $len; $i ++){
        my @array = ();
        for(my $j = 0; $j < $len; $j ++){
                push @array, 0;
        }
        push @raw_matrix, \@array;

}
#####################
my @expt_matrix;
for(my $i = 0; $i < $len; $i ++){
        my @array = ();
        for(my $j = 0; $j < $len; $j ++){
                push @array, 0;
        }
        push @expt_matrix, \@array;

}

#################### Read frag loops  ###################
foreach my $id1 (sort {$a <=> $b} keys %{$loop_result}){
	foreach my $id2 ( sort {$a <=> $b} keys %{$loop_result->{$id1}}){
		my ($obs,$expt,$ratio) = split ",", $loop_result->{$id1}->{$id2};
		my $i = $anchor_idx->{$id1};
		my $j = $anchor_idx->{$id2};	
        	$ratio_matrix[$i][$j] = $ratio;
	        $ratio_matrix[$j][$i] = $ratio;	
		$raw_matrix[$i][$j] = $obs;
                $raw_matrix[$j][$i] = $obs;
		$expt_matrix[$i][$j] = $expt;
                $expt_matrix[$j][$i] = $expt;	
	}
}

open(OUT,">$output_dir.raw.matrix");
for(my $i = 0; $i < $len; $i++){
        print OUT join("\t", @{$raw_matrix[$i]})."\n";
}
close(OUT);

open(OUT,">$output_dir.expt.matrix");
for(my $i = 0; $i < $len; $i++){
        print OUT join("\t", @{$expt_matrix[$i]})."\n";
}
close(OUT);

open(OUT,">$output_dir.ratio.matrix");
for(my $i = 0; $i < $len; $i++){
        print OUT join("\t", @{$ratio_matrix[$i]})."\n";
}
close(OUT);

exit;
#########################################################
sub get_id{
        my ($loc, $range) = @_;
        foreach my $id (keys %{$range}){
                my ($min, $max) = split ":", $range->{$id};
                if($loc >= $min && $loc <= $max){
                        return $id;
                }
        }
        die("Error: did not find a group\n");
}
