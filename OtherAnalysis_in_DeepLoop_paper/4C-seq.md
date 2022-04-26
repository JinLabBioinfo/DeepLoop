### The 4C-seq was analysed using [pipe4C](https://github.com/deLaatLab/pipe4C)
### Install pipe4C, download fastq and prepare the VP.info.
### Run pipe4C
```
Rscript pipe4C.R –-vpFile [path to vpFile] --fqFolder [path to folder containing the FASTQ files] –-outFolder [path to output folder] --cores 8 --wig --plot --genomePlot
```
The output contains wig.gz and bam files. The wig.gz can be visualized in Genome browser, bam files are used to assign reads on each heterozygous SNP.

### bam to SNP-allele_reads to bigWig
Assume your bam file is $name.bam
```
samtools view $name.bam | cut -f10 >temp
bedtools bamtobed -i $name.bam -cigar | paste - temp | awk '{OFS="\t";print $1,$2,$3,$4,$8,$7}' | python lib/get_seq_consider_cigar.py | awk '{OFS="\t";print $1,$2,$3,$4,$7}' |  bedtools intersect -wa -wb -a - -b <(cat GSE63525_GM12878_SNPs.reformat.bed | awk '{if($1=="'$chr'")print}') | python lib/split.py > $name.unsplitable

cat maternal.reads | cut -f6-9 | sort | uniq -c | awk '{print $2 "\t" $3 "\t" $4 "\t" $5 "\t" $1}' > $name.Mtmp
cat paternal.reads | cut -f6-9 | sort | uniq -c | awk '{print $2 "\t" $3 "\t" $4 "\t" $5 "\t" $1}' > $name.Ptmp
bedtools intersect -wa -wb -f 1 -a  <(cat GSE63525_GM12878_SNPs.reformat.bed | awk '{print $1 "\t" $2 "\t" $3}') -b <(cat $name.Mtmp | awk '{print $1 "\t" $2 "\t" $3 "\t" $5}') | bedtools intersect -wa -wb -f 1 -a - -b <(cat $name.Ptmp | awk '{print $1 "\t" $2 "\t" $3 "\t" $5}') | cut -f1-3,7,11 | awk '{print $1 "\t" $2-1 "\t" $3 "\t"$4 "\t" $5 "\t" ($5/($4+$5))-0.5}' > $name.p_fraction
~/software/bedGraphToBigWig $name.p_fraction ~/zshanshan/UCSC_interact/hg19.chrom.sizes $name.p_fraction.bw
```
