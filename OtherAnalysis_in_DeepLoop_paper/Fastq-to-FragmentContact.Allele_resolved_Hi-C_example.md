## The GM12878 in-situ Hi-C data is from GEO accession number [GSE63525](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525) PMID: 25497547
## restriction enzyme: Mbol
### Step1: Download data
#### Download data (fastq-dump)
#### Download snp bed file 
```
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_GM12878_SNPs.txt.gz
```
### reformat snp file to a bed file for processing bams
```
cat GSE63525_GM12878_SNPs.txt.gz | gunzip | sed s/":"/\\t/g | awk '{print "chr"$1 "\t" $2 "\t" $2  "\t" $3"/"$4}' > GSE63525_GM12878_SNPs.reformat.bed
cat GSE63525_GM12878_SNPs.reformat.bed | awk '{print NR "\t" $1 "\t" $2 "\t" "-" "\t" $4}' > snp.bed
```
#### mask hg19.fa by GSE63525_GM12878_SNPs.reformat.bed
`bedtools maskfasta -fi hg19.fa -bed GSE63525_GM12878_SNPs.reformat.bed -fo hg19.masked.fa`
#### rebuild hg19 bowtie2Index with hg19.masked.fa
`bowtie2-build hg19.masked.fa`
### Step2: run mapping with bowtie2 (hg19 build)
#### To cover more reads overlapping snps, we use the full length to do mapping (the mapping step is same as [HiC-Pro](https://github.com/nservant/HiC-Pro))
#### Edit the path to bowtiepath and path to lib and fragment bed file
```
hg19=Your_hg19_Bowtie2IndexPath/YourIndexPrefix
hg19Ind=YourIndexPrefix.fa.fai
lib=Path_to_lib
bed=Path_to_fragbed
# There are 29 replicates in total, we only take SRR1658587 as an example here
SRR=SRR1658587
```
```
bowtie2 --very-sensitive -L 30 --score-min L,-0.6,-0.2 --end-to-end --reorder --rg-id BMG --rg SM:${SRR}_R1 -p 4 -x $hg19 -U ${SRR}_1.fastq | samtools view -bS - >  ${SRR}_R1_hg19.masked.bam &
bowtie2 --very-sensitive -L 30 --score-min L,-0.6,-0.2 --end-to-end --reorder --rg-id BMG --rg SM:${SRR}_R2 -p 4 -x $hg19 -U ${SRR}_2.fastq | samtools view -bS - >  ${SRR}_R2_hg19.masked.bam &
wait
samtools sort -@ 2 -n -T ${SRR}_R1_hg19.masked -o ${SRR}_R1_hg19.masked.sorted.bam ${SRR}_R1_hg19.masked.bam &
samtools sort -@ 2 -n -T ${SRR}_R2_hg19.masked -o ${SRR}_R2_hg19.masked.sorted.bam${SRR}_R2_hg19.masked.bam &
wait
mv ${SRR}_R1_hg19.masked.sorted.bam ${SRR}_R1_hg19.masked.bam
mv ${SRR}_R2_hg19.masked.sorted.bam ${SRR}_R2_hg19.masked.bam
python $lib/mergeSAM.py -q 0 -t -v -f ${SRR}_R1_hg19.masked.bam -r ${SRR}_R2_hg19.masked.bam -o ${SRR}_hg19.masked.bwt2pairs.bam
# split bam to two alleles based on snp info
# Here we use [SNPsplit](https://github.com/FelixKrueger/SNPsplit) to split bam files
SNPsplit --snp_file snp.bed ${SRR}_hg19.masked.bwt2pairs.bam --paired --hic 
```
#### Repeat the code above to run all the data

### Step3: Process the bam files
#### For each allele(genome1 or genome2), merge the bam files(*.G1_UA.bam or *.G2_UA.bam) from the same biological replicate, then remove duplicates for each biological replicate bam(Rep1..Rep29), for example, we named them as Rep1.Genome1.bam and Rep1.Genome2.bam for Rep1
```
samtools sort Rep1.Genome1.bam  | samtools view - | $lib/remove_dup_PE_SAM_sorted.pl | samtools view -bS -t $hg19Ind -o - - > Rep1.Genome1.sorted.nodup.bam 
samtools sort Rep1.Genome2.bam  | samtools view - | $lib/remove_dup_PE_SAM_sorted.pl | samtools view -bS -t $hg19Ind -o - - > Rep1.Genome2.sorted.nodup.bam 
```
#### After remove duplicate, for each replicate, they are named as 
```
samtools view  Rep1.Genome1.sorted.nodup.bam | cut -f2-8 | $lib/bam_to_temp_HiC.pl > Rep1.G1.temp 
samtools view  Rep1.Genome2.sorted.nodup.bam | cut -f2-8 | $lib/bam_to_temp_HiC.pl > Rep1.G2.temp 
```
### Step4: map reads pair to fragment pairs, for G1.temp and G2.temp
```
for genome in G1 G2;do
  $lib/reads_2_cis_frag_loop.pl $bed $readlength Rep1.$genome.loop.inward Rep1.$genome.loop.outward Rep1.$genome.loop.samestrand summary.frag_loop.read_count Rep1.$genome Rep1.$genome.temp  &
  $lib/reads_2_trans_frag_loop.pl $bed $readlength $genome.loop.trans Rep1.$genome.temp  &
  wait
  for file in Rep1.$genome.loop.inward Rep1.$genome.loop.outward Rep1.$genome.loop.samestrand;do
        cat $file | $lib/summary_sorted_frag_loop.pl $bed  > temp.$file &
  done
  wait
  for file in Rep1.$genome.loop.inward Rep1.$genome.loop.outward Rep1.$genome.loop.samestrand;do
        $lib/resort_by_frag_id.pl $bed temp.$file &
  done
  cat Rep1.$genome.loop.trans | $lib/summary_sorted_trans_frag_loop.pl - > temp.Rep1.$genome.loop.trans
  wait
done
```
### Step5: After running the steps above for each biological bam file, merge fragment pairs from biological replicates
```
for genome in G1 G2;do
  $lib/merge_sorted_frag_loop.pl <(cat temp.Rep*$genome.loop.samestrand*) <cat temp.Rep*$genome.loop.inward | awk '{if($4>1000)print $0}') <cat temp.Rep*$genome.loop.outward | awk '{if($4>5000)print $0}') > frag_loop.$genome.cis &
  $lib/merge_sorted_frag_loop.pl <(cat temp.Rep*$genome.loop.trans) > frag_loop.$genome.trans &
  wait
done
```
### frag_loop.G1.cis and frag_loop.G1.trans, frag_loop.G2.cis and frag_loop.G2.trans will be used to run [HiCorr](https://github.com/JinLabBioinfo/HiCorr)
### Step6: Run HiCorr
```
$Hicorr_path/HiCorr DPNII frag_loop.G1.cis frag_loop.G1.trans G1 hg19 
cat `ls HiCorr_output/* | grep -v p_val` | awk '{sum+=$3}END{print sum/2}' # check reads within 2Mb
```
