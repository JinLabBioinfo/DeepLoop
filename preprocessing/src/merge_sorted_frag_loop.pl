#!/usr/bin/perl

use strict;

my $usage = "Usage:./merge_sorted_frag_loop.pl <loop files>\n";

my @loop_files = @ARGV;
if(! @loop_files){
	die($usage);
}

my $file_handles;
foreach my $file (@loop_files){
	my $fh;
	open($fh, $file);
	$file_handles->{$file} = $fh;
}

my $curr_data;
my $next_id1;
my $next_id2;
my $next_count;
my $next_dist;

foreach my $file (@loop_files){
	my $fh = $file_handles->{$file};
	if(my $line = <$fh>){
		chomp $line;
		($next_id1->{$file}, $next_id2->{$file}, $next_count->{$file}, $next_dist->{$file}) = split "\t", $line;
		$next_id1->{$file} =~ s/frag_//;
		$next_id2->{$file} =~ s/frag_//;
		if(not defined $next_dist->{$file}){
			$next_dist->{$file} = -1;
		}
	}
}
my $curr_id1 = min(values %$next_id1);
my $curr_dist;
while($curr_id1){
	foreach my $file (keys %$file_handles){
		my $fh = $file_handles->{$file};
		while($next_id1->{$file} == $curr_id1){
			my $id2 = $next_id2->{$file};
			if(not defined $curr_data->{$id2}){
				$curr_data->{$id2} = 0;
			}
			$curr_dist->{$id2} = $next_dist->{$file};
			$curr_data->{$id2} += $next_count->{$file};
			if(my $line = <$fh>){
				chomp $line;
				($next_id1->{$file}, $next_id2->{$file}, $next_count->{$file}, $next_dist->{$file}) = split "\t", $line;
				$next_id1->{$file} =~ s/frag_//;
				$next_id2->{$file} =~ s/frag_//;
				if(not defined $next_dist->{$file}){
			                $next_dist->{$file} = -1;
        			}
			}else{
				close($fh);
				delete $file_handles->{$file};
				delete $next_id1->{$file};
				delete $next_id2->{$file};
				delete $next_count->{$file};
				delete $next_dist->{$file};
			}
		}
	}
	if($curr_id1 > 0){
		output($curr_id1, $curr_data, $curr_dist);
	}
	if(keys %$next_id1){
		$curr_id1 = min(values %$next_id1);
	}else{
		$curr_id1 = 0;
	}
}

exit;

###############################################################
sub min{
	my @array = sort {$a<=>$b} @_;
	return $array[0];
}

sub output{
	my ($id1, $ref, $dist_ref) = @_;
	foreach my $id2 (sort {$a<=>$b} keys %$ref){
		print join("\t", "frag_$id1", "frag_$id2", $ref->{$id2});
		delete $ref->{$id2};
		my $dist = $dist_ref->{$id2};
		if($dist < 0){
			print "\n";
		}else{
			print "\t".$dist."\n";
		}
		delete $dist_ref->{$id2};
	}
	return 1;
}

