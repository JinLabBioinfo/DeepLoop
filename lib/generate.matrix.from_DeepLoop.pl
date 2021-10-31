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
#print join("\t",$start_A, $end_A);
my @start_str = split /\_/, $start_A;
my $start_num=@start_str[1];
my @end_str = split /\_/, $end_A;
my $end_num=@end_str[1];

my @anchor_lis = ();
my $loop_result;
open(IN, $anchor_file);
#my $len=0;
while(my $line = <IN>){
        chomp $line;
        my ($A1, $A2, $denoise) = split " ", $line;
        my @str1 = split /\_/, $A1;
        my $a1=@str1[1];
        my @str2 = split /\_/, $A2;
        my $a2=@str2[1];
#	print join("\t",$start_num,$end_num,$a1,$a2)."\n";
	if($a1>=$start_num && $a1<=$end_num){
		if($a2>=$start_num && $a2<=$end_num){
#			print join("\t",$A1, $A2, $obs, $expt, $ratio, $denoise,$a1,$a2)."\n";
	#		$len ++;
			$loop_result->{$a1}->{$a2}=$denoise;
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
#	print @anchor_lis[$id]."\n";
	$anchor_idx->{$_} = $i;
	$i ++;
	$len ++;
}
my $size=@anchor_lis;
#print $size."\n";


my @denoise_matrix;
for(my $i = 0; $i < $len; $i ++){
        my @array = ();
        for(my $j = 0; $j < $len; $j ++){
                push @array, 0;
        }
        push @denoise_matrix, \@array;

}

foreach my $id1 (sort {$a <=> $b} keys %{$loop_result}){
	foreach my $id2 ( sort {$a <=> $b} keys %{$loop_result->{$id1}}){
		my ($denoise) = $loop_result->{$id1}->{$id2};
		my $i = $anchor_idx->{$id1};
		my $j = $anchor_idx->{$id2};	
	        $denoise_matrix[$i][$j] = $denoise;
	        $denoise_matrix[$j][$i] = $denoise;
	}
}

open(OUT,">$output_dir.denoise.matrix");
for(my $i = 0; $i < $len; $i++){
        print OUT join("\t", @{$denoise_matrix[$i]})."\n";
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
