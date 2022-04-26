### We highly recommend processing Hi-C data from fastq or bam files as the examples showed in [Fastq-to-FragmentContact.Allele_resolved_Hi-C_example.md](https://github.com/shanshan950/prepossessing/blob/master/documents/Fastq-to-FragmentContact.Allele_resolved_Hi-C_example.md) and [Fastq-to-FragmentContact.Tissue_example.md](https://github.com/shanshan950/prepossessing/blob/master/documents/Fastq-to-FragmentContact.Tissue_example.md) 
### This example is to show the processed reads pair data from sn-m3C-seq, accessoion number [GSE130711](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE130711) (PMID: 31501549)

### Step1 Download processed data from 
```
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE130nnn/GSE130711/suppl/GSE130711_RAW.tar
```
#### The downloaded processed reads pair data have two formats, one with strand information, the other without strand information. 
### Step2 Map read pair to fragment pairs
#### files without strand columns, for each file run:
```
cat $file | gunzip | $lib/reads_2_cis_frag_loop.no_strand.pl $lib/hg19.DPNII.frag.bed 1 ${file/_indexed_contacts.txt.gz/}.cis &
cat $file | gunzip | $lib/reads_2_trans_frag_loop.no_strand.pl $lib/hg19.DPNII.frag.bed 1 ${file/_indexed_contacts.txt.gz/}.trans - &
wait
$lib/summary_sorted_trans_frag_loop.pl ${file/_indexed_contacts.txt.gz/}.trans | cut -f1-3 > frag_loop.${file/_indexed_contacts.txt.gz/}.trans &
cat ${file/_indexed_contacts.txt.gz/}.cis | $lib/summary_sorted_frag_loop.pl $lib/hg19.DPNII.frag.bed > temp.${file/_indexed_contacts.txt.gz/}.cis &
wait
$lib/resort_by_frag_id.pl $lib/hg19.DPNII.frag.bed temp.${file/_indexed_contacts.txt.gz/}.cis
mv temp.${file/_indexed_contacts.txt.gz/}.cis frag_loop.${file/_indexed_contacts.txt.gz/}.cis

```
#### files with strand columns, for each file run:
```
name=${file/_indexed_contacts.txt.gz/}
cat $file | gunzip | $lib/reads_2_cis_frag_loop.snm3Cseq.pl $lib/hg19.DPNII.frag.bed 1 $name.loop.inward $name.loop.outward $name.loop.samestrand summary.frag_loop.read_count $name -  &
cat $file | gunzip | $lib/reads_2_trans_frag_loop.snm3Cseq.pl $lib/hg19.DPNII.frag.bed 1 $name.trans - &
wait
cat $name.loop.inward | $lib/summary_sorted_frag_loop.pl $lib/hg19.DPNII.frag.bed > temp.$name.loop.inward &
cat $name.loop.outward | $lib/summary_sorted_frag_loop.pl $lib/hg19.DPNII.frag.bed > temp.$name.loop.outward &
cat $name.loop.samestrand | $lib/summary_sorted_frag_loop.pl $lib/hg19.DPNII.frag.bed > temp.$name.loop.samestrand &
wait
$lib/resort_by_frag_id.pl $lib/hg19.DPNII.frag.bed temp.$name.loop.inward &
$lib/resort_by_frag_id.pl $lib/hg19.DPNII.frag.bed temp.$name.loop.outward &
$lib/resort_by_frag_id.pl $lib/hg19.DPNII.frag.bed temp.$name.loop.samestrand &
wait
cat temp.$name.loop.inward  |  awk '{if($4>1000)print $0}' > frag_loop.$name.inward
cat temp.$name.loop.outward | awk '{if($4>5000)print $0}' > frag_loop.$name.outward
$lib/merge_sorted_frag_loop.pl frag_loop.$name.inward frag_loop.$name.outward temp.$name.loop.samestrand > frag_loop.$name.cis &
$lib/summary_sorted_trans_frag_loop.pl $name.trans | cut -f1-3 > frag_loop.$name.trans &
wait
rm -rf $name.loop.inward $name.loop.outward $name.loop.samestrand temp.$name.loop.inward temp.$name.loop.outward temp.$name.loop.samestrand frag_loop.$name.inward frag_loop.$name.outward $name.trans
```
### Step3 merge fragment pairs from the same cell type 
$lib/merge_sorted_frag_loop.pl <cellType1.cisfile_lis> > frag_loop.cellType1.cis
$lib/merge_sorted_frag_loop.pl <cellType1.transfile_lis> > frag_loop.cellType1.trans
### frag_loop.cellType1.cis and frag_loop.cellType1.trans will be used to run [HiCorr](https://github.com/JinLabBioinfo/HiCorr)
### Check details in [FragmentContact-to-HiCorr-DeepLoop.example.md](https://github.com/shanshan950/prepossessing/blob/master/documents/FragmentContact-to-HiCorr-DeepLoop.example.md) 
