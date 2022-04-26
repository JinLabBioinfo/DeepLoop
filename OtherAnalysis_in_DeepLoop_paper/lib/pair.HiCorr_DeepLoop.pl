#!/usr/bin/perl
use strict;
my $usage = "Usage: ./pair.HiCorr_DeepLoop.pl <anchor_to_anchor_loop> <denoise.loop>\n";
my ($HiCorr,$DeepLoop,$dummy) = @ARGV;
my $A_lis;
open(IN, $HiCorr);
while(my $line= <IN>){
        chomp $line;
        my ($A1,$A2,$obs,$expt)= split "\t", $line;
	my $ratio = ($obs+$dummy)/($expt+$dummy);
        $A_lis->{$A1}->{$A2}=join("\t",$obs,$ratio);
}
close(IN);
open(IN, $DeepLoop);
while(my $line= <IN>){
        chomp $line;
        my ($A1,$A2,$val)= split "\t", $line;
        if($A_lis->{$A1}->{$A2}){
                print join("\t", $A1,$A2, $A_lis->{$A1}->{$A2},$val)."\n";
        }
}
close(IN);
exit;
