#!/usr/bin/perl
#
#  Author: Preslav Nakov
#  
#  Description: Generates the expected gold output for the devel set.
#
#  Last modified: September 20, 2014
#
#

use warnings;
use strict;
use utf8;

use constant INPUT_FILE          => 'CQA-QL-devel.xml';
use constant OUTPUT_FILE_GOLD    => 'CQA-QL-devel-gold.txt';
use constant OUTPUT_FILE_GOLD_YN => 'CQA-QL-devel-gold-yn.txt';
use constant OUTPUT_FILE_INPUT   => 'CQA-QL-devel-input.xml';


################
###   MAIN   ###
################

### 1. Open the files
open INPUT, INPUT_FILE or die;
open OUTPUT_GOLD, '>' . OUTPUT_FILE_GOLD or die;
open OUTPUT_GOLD_YN, '>' . OUTPUT_FILE_GOLD_YN or die;
open OUTPUT_INPUT, '>' . OUTPUT_FILE_INPUT or die;

binmode(INPUT, ":utf8");
binmode(OUTPUT_GOLD, ":utf8");
binmode(OUTPUT_GOLD_YN, ":utf8");
binmode(OUTPUT_INPUT, ":utf8");

### 2. Process the files
while (<INPUT>) {

	### 2.1. Look for comments
	### <Comment CID="Q2601_C1" CUSERID="U3098" CGOLD="Potential" CGOLD_YN="Not Applicable">
	if (/<Comment CID=\"(Q[0-9]+\_C[0-9]+)\" CUSERID=\"U[0-9]+\" CGOLD="([^\"]+)"/) {
		my ($cid, $label) = ($1, $2);
		print OUTPUT_GOLD "$cid\t$label\n";

		s/ CGOLD=\"[^\"]+\"/ CGOLD="\?\"/;
		s/ CGOLD_YN=\"[^\"]+\"//;
	}

	### 2.2. Look for questions
	### <Question QID="Q2900" QCATEGORY="Qatar Living Lounge" QDATE="2010-05-13 14:22:31" QUSERID="U37" QTYPE="YES_NO" QGOLD_YN="Yes">
	elsif (/<Question QID=\"(Q[0-9]+)\" QCATEGORY=\"[^\"]+\" QDATE=\"[^\"]+\" QUSERID=\"U[0-9]+\" QTYPE=\"([^\"]+)\" QGOLD_YN=\"([^\"]+)\">/) {
		my ($qid, $qtype, $label) = ($1, $2, $3);
		if ($qtype eq 'YES_NO') {
			print OUTPUT_GOLD_YN "$qid\t$label\n";
			s/ QTYPE=\"YES_NO\" QGOLD_YN=\"[^\"]+\"/ QTYPE=\"YES_NO\" QGOLD_YN=\"\?\"/;
		}
		else {
			s/ QTYPE=\"GENERAL\" QGOLD_YN=\"[^\"]+\"/ QTYPE=\"GENERAL\"/;
		}
	}

	### 2.3. Output the cleansed input with question marks
	print OUTPUT_INPUT;
}

### 3. Close the files
close INPUT or die;
close OUTPUT_GOLD or die;
close OUTPUT_GOLD_YN or die;
close OUTPUT_INPUT or die;
