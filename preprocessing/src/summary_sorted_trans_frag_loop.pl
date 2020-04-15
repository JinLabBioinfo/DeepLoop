#!/usr/bin/perl

use strict;

my $usage = "Usage:./summary_sorted_trans_frag_loop.pl <loop_list>\n";

my ($loop_list) = @ARGV;


if(not defined $loop_list){
	$loop_list = "-";
}

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
			foreach my $id (sort {$a <=> $b} keys %$frag_ref){
				print join("\t", "frag_$prev_id", "frag_$id", $frag_ref->{$id})."\n";
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
	foreach my $id (sort {$a <=> $b} keys %$frag_ref){
		print join("\t", "frag_$prev_id", "frag_$id", $frag_ref->{$id})."\n";
		delete $frag_ref->{$id};
	}
}

exit;
