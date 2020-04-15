#!/usr/bin/perl

use strict;

my $prev_chr = 0;
my $prev_loc = 0;

my $data;

while(my $line = <stdin>){
	chomp $line;
	my ($id, $flag, $chr1, $loc1, $t1, $t2, $chr2, $loc2, $size, @rest) = split "\t", $line;
	if($chr2 eq "="){$chr2 = $chr1;}
	my ($str1, $str2) = ("+", "+");
	if($flag & 16){$str1 = "-";}
	if($flag & 32){$str2 = "-";}
	my $group = join(":", $str1,$chr2,$loc2,$str2);

	if(($chr1 eq $prev_chr) && ($loc1 eq $prev_loc)){
		$data = save_data($data, $group, $line);
	}else{
		print_data($data);
		$prev_chr = $chr1;
		$prev_loc = $loc1;
		$data = save_data($data, $group, $line);
	}
}
print_data($data);
exit;

##########################################################
sub print_data{
	my ($data) = @_;
	foreach my $group (keys %$data){
		print $data->{$group}."\n";
		delete $data->{$group};
	}
	return;
}

sub save_data{
	my ($data, $group, $line) = @_;
	if(not defined $data->{$group}){
		$data->{$group} = $line;
	}else{
		my ($prev_id) = split "\t", $data->{$group};
		my ($id) = split "\t", $line;
		if($id lt $prev_id){
			$data->{$group} = $line;
		}elsif($id eq $prev_id){
			$data->{$group} = join("\n", $data->{$group}, $line);
		}else{
		}
	}
	return $data;
}
