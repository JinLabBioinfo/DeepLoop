#!/usr/bin/perl

use strict;

my $usage = "Usage:./summary_sorted_frag_loop.pl <frag_bed> <loop_list>\n";

my ($frag_bed, $loop_list) = @ARGV;

if(not defined $frag_bed){
	die($usage);
}

if(not defined $loop_list){
	$loop_list = "-";
}

my $frag_loc;
open(IN, $frag_bed);
while(my $line = <IN>){
        chomp $line;
        my ($chr, $beg, $end, $id) = split "\t", $line;
	$id =~ s/frag_//;
        $frag_loc->{$id} = join(":", $beg, $end);
}
close(IN);

open(IN, $loop_list);
my $prev_id = 0;
my $frag_ref;
while(my $line = <IN>){
	chomp $line;
	my ($fid1, $fid2, $count) = split "\t", $line;
	$fid1 =~ s/frag_//;
	$fid2 =~ s/frag_//;
	
	if(not defined $count){
		$count = 1;
	}
	
	if($fid1 != $prev_id){
		if($prev_id){
			my ($beg1, $end1) = split ":", $frag_loc->{$prev_id};
			foreach my $id (sort {$a <=> $b} keys %$frag_ref){
				my ($beg2, $end2) = split ":", $frag_loc->{$id};
				my $dist = ($beg1 < $beg2)?($beg2 - $end1 - 1):($beg1 - $end2 -1);
#				my $cut = abs($id - $prev_id);
				print join("\t", "frag_$prev_id", "frag_$id", $frag_ref->{$id}, $dist)."\n";
				delete $frag_ref->{$id};
			}
		}
		$prev_id = $fid1;
	}

	if(not defined $frag_ref->{$fid2}){
		$frag_ref->{$fid2} = $count;
	}else{
		$frag_ref->{$fid2} += $count;
	}
}
close(IN);

if($prev_id){
	my ($beg1, $end1) = split ":", $frag_loc->{$prev_id};
	foreach my $id (sort {$a <=> $b} keys %$frag_ref){
		my ($beg2, $end2) = split ":", $frag_loc->{$id};
		my $dist = ($beg1 < $beg2)?($beg2 - $end1 - 1):($beg1 - $end2 -1);
#		my $cut = abs($id - $prev_id);
		print join("\t", "frag_$prev_id", "frag_$id", $frag_ref->{$id}, $dist)."\n";
		delete $frag_ref->{$id};
	}
}

exit;
