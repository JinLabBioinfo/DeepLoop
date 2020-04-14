# prepossessing
Here we take part of GM12878 cell HiC data(GSM1181867) as an example to show the prepossessing, HiCorr, DeepLoop steps.
Before you start, you need to initialize some variables:
"hg19_fa", path to bowtie, path to bowtieindex, "lib" and "ref"
You will also need samtools perl python python3 installed
## Step1: data download mapping

```
lib=path_to_the_download_repository/lib
ref=path_to_the_download_repository/ref
hg19_fa=path_to_genome.fa.fai
bowtie=path_to_bowtie
hg19=path_to_hg19BowtieIndex/prefix
```
```
# data downloading
fastq-dump --split-files SRR927086

#trim to get 36 bp
cat SRR927086_1.fastq  | $lib/reform_fastq.for.public.tissue.py 1 36 > SRR927086.R1.fastq &
cat SRR927086_2.fastq  | $lib/reform_fastq.for.public.tissue.py 1 36 > SRR927086.R2.fastq &
wait

#bowtie mapping using only one CPU to make sure the read name for two fastq are in the same order; Or you can run mapping with multiple CPUs, then you can do samtools sort -n for each sam file
$bowtie -v 3 -m 1 --best --strata --time -p 1 --sam $hg19 SRR927086.R1.fastq SRR927086.R1.sam &
$bowtie -v 3 -m 1 --best --strata --time -p 1 --sam $hg19 SRR927086.R2.fastq SRR927086.R2.sam &
wait

#Pair two sam files and convert to bam file
$lib/pairing_two_SAM_reads.pl SRR927086.R1.sam SRR927086.R2.sam | samtools view -bS -t $hg19_fa -o - - > SRR927086.bam

#Sort bam file
samtools sort SRR927086.bam -o SRR927086.bam.sorted

#For technical replicates or multiple run fastq, you can either merge fastq from the beginning or run each to the "bam.sorted" files and merge bam file for next step "remove duplicates"
# Remove duplicates
samtools view SRR927086.bam.sorted  | $lib/remove_dup_PE_SAM_sorted.pl | samtools view -bS -t $hg19_fa -o - - > SRR927086.sorted.nodup.bam

# extract read info
samtools view SRR927086.sorted.nodup.bam  | cut -f2-8 | $lib/bam_to_temp_HiC.pl > SRR927086.temp

# convert cis reads pair to fragment pairs and split them into "inward", "outward", "samestrand"
$lib/reads_2_cis_frag_loop.pl $ref/hg19.HindIII.frag.bed 36 SRR927086.loop.inward SRR927086.loop.outward SRR927086.loop.samestrand summary.frag_loop.read_count SRR927086 SRR927086.temp

# convert cis reads pair to fragment pairs and take trans fragment pairs
cat SRR927086.temp | $lib/reads_2_trans_frag_loop.pl $ref/hg19.HindIII.frag.bed 36 | $lib/summary_sorted_trans_frag_loop.pl > SRR927086.loop.trans
# sort cis fragment pairs file by fragment id
for file in SRR927086.loop.inward SRR927086.loop.outward SRR927086.loop.samestrand;do
        cat $file | $lib/summary_sorted_frag_loop.pl $ref/hg19.HindIII.frag.bed > temp.$file
        mv temp.$file $file
        $lib/resort_by_frag_id.pl $ref/hg19.HindIII.frag.bed $file
done
# sort trans fragment pairs by fragment id
$lib/resort_by_frag_id.pl $ref/hg19.HindIII.frag.bed SRR927086.loop.trans
```

## Step2: Merge and filter fragment pairs
In most cases, you have biological replicates, repeat the step1 above for each biological replicate and then go through the next step, after this tep, you will have the fragment pairs file for cis and trans.
```
# Merging all the "samestrand" in the current directory
$lib/merge_sorted_frag_loop.pl `ls *.samestrand` > frag_loop.SRR927086.samestrand &
# Merging all the "inward" in the current directory
$lib/merge_sorted_frag_loop.pl `ls *.inward` | awk '{if($4>1000)print $0}' > frag_loop.SRR927086.inward &
# Merging all the "outward" in the current directory
$lib/merge_sorted_frag_loop.pl `ls *.outward` | awk '{if($4>25000)print $0}' > frag_loop.SRR927086.outward &
wait
# Merging samestrand, inward, and outward to cis reads fragment pairs
$lib/merge_sorted_frag_loop.pl frag_loop.SRR927086.samestrand $dir/frag_loop.SRR927086.inward frag_loop.SRR927086.outward > frag_loop.SRR927086.cis
# Merging all the "loop.trans" in the current directory
$lib/merge_sorted_frag_loop.pl `ls *.loop.trans` > frag_loop.SRR927086.trans
```
## Step3: Run HiCorr for bias-correction
HiCorr needs cis frag loop and trans frag loop as input, the "temp.by.chrom" directory in the output will be used for the DeepLoop step

## Step4: Run LoopDenoise or LoopEnhance for the bias-corrected anchor loops
./run_enhance.sh <path_to_temp.by.chrom>
