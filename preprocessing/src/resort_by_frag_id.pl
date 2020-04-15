#!/usr/bin/perl

use strict;
my $usage =     "Usage:./resort_by_frag_id.pl <frag_bed> <loop_file_sorted_within_chrom>\n".
                "\tThis program takes loop file, each chrome should already be sorted, but frag_id not sorted because of unclear order between chrome\n".
		"\tTherefore this program first split loop file by chrom and then merge them again by fragment id.\n";

my ($frag_bed, $loop_file) = @ARGV;

if(not defined $loop_file){
	die($usage);
}

my $folder = "tmp_folder.$loop_file";
`mkdir $folder`;

my @temp_files = ();

my $frag_chr;
my $file_handles;
open(IN, $frag_bed);
while(my $line = <IN>){
        chomp $line;
        my ($chr, $beg, $end, $id) = split "\t", $line;
        $frag_chr->{$id} = $chr;
	if(not defined $file_handles->{$chr}){
		my $fh;
		open($fh, ">$folder/$loop_file.$chr");
		push @temp_files, "$folder/$loop_file.$chr";
		$file_handles->{$chr} = $fh;
	}
}
close(IN);

my $prev_chr = 0;
my $curr_fh;
open(IN, $loop_file);
while(my $line = <IN>){
	chomp $line;
	my ($id, @rest) = split "\t", $line;
	my $chr = $frag_chr->{$id};
	if($chr ne $prev_chr){
		if($prev_chr){
			close($curr_fh);
		}
		$curr_fh = $file_handles->{$chr};
		$prev_chr = $chr;
	}
	print $curr_fh $line."\n";
}
close(IN);

close($curr_fh);
my $file_list = join(" ", @temp_files);

`/mnt/rstor/genetics/JinLab/ssz20/zshanshan/lib/merge_sorted_frag_loop.pl $file_list > temp.$loop_file`;
`mv temp.$loop_file $loop_file`;
#`rm -r $folder`;

exit;
