#!/usr/bin/perl

use strict;
use List::MoreUtils qw(uniq);
my $usage = "./generate.raw.HiCorr.DeepLoop.matrix.pl <anchor_bed> <anchor_file> <chr> <start> <end> <output_dir>";

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
        my ($A1, $A2, $obs, $HiCorr, $DeepLoop) = split "\t", $line;
	#	my $HiCorr=($obs+5)/($expt+5);
        my @str1 = split /\_/, $A1;
        my $a1=@str1[1];
        my @str2 = split /\_/, $A2;
        my $a2=@str2[1];
	if($a1>=$start_num && $a1<=$end_num){
		if($a2>=$start_num && $a2<=$end_num){
			$loop_result->{$a1}->{$a2}=join(",",$obs,$HiCorr, $DeepLoop);
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
my @HiCorr_matrix;
for(my $i = 0; $i < $len; $i ++){
        my @array = ();
        for(my $j = 0; $j < $len; $j ++){
                push @array, 0;
        }
        push @HiCorr_matrix, \@array;

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
my @DeepLoop_matrix;
for(my $i = 0; $i < $len; $i ++){
        my @array = ();
        for(my $j = 0; $j < $len; $j ++){
                push @array, 0;
        }
        push @DeepLoop_matrix, \@array;

}

#################### Read frag loops  ###################
foreach my $id1 (sort {$a <=> $b} keys %{$loop_result}){
	foreach my $id2 ( sort {$a <=> $b} keys %{$loop_result->{$id1}}){
		my ($obs,$HiCorr,$DeepLoop) = split ",", $loop_result->{$id1}->{$id2};
		my $i = $anchor_idx->{$id1};
		my $j = $anchor_idx->{$id2};	
        	$HiCorr_matrix[$i][$j] = $HiCorr;
	        $HiCorr_matrix[$j][$i] = $HiCorr;	
		$raw_matrix[$i][$j] = $obs;
                $raw_matrix[$j][$i] = $obs;
		$DeepLoop_matrix[$i][$j] = $DeepLoop;
                $DeepLoop_matrix[$j][$i] = $DeepLoop;	
	}
}

open(OUT,">$output_dir.raw.matrix");
for(my $i = 0; $i < $len; $i++){
        print OUT join("\t", @{$raw_matrix[$i]})."\n";
}
close(OUT);

open(OUT,">$output_dir.HiCorr.matrix");
for(my $i = 0; $i < $len; $i++){
        print OUT join("\t", @{$HiCorr_matrix[$i]})."\n";
}
close(OUT);

open(OUT,">$output_dir.DeepLoop.matrix");
for(my $i = 0; $i < $len; $i++){
        print OUT join("\t", @{$DeepLoop_matrix[$i]})."\n";
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
