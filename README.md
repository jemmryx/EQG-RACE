Data and code for paper "[EQG-RACE: Examination-Type Question Generation](https://arxiv.org/abs/2012.06106)" at AAAI2021.

## Pure Dataset

EQG-RACE.tar.gz

each sample contains
+ sent:splited paragraph
+ question
+ answer
+ max_sent: key sentence in paragraph with max rough score
+ rouge: max rough score
+ tag: key sentence tag

## Processed Dataset
race.tar.gz

contains the processed train/dev/test data and corresponding adjacency matrix

## Code 

The unified model (without using pretrained elmo/bert) is uploaded (flaten version).

## Cite

If you extend or use this work, please cite the [paper](https://arxiv.org/abs/2012.06106) where it was introduced:

```
@misc{jia2020eqgrace,
      title={EQG-RACE: Examination-Type Question Generation}, 
      author={Xin Jia and Wenjie Zhou and Xu Sun and Yunfang Wu},
      year={2020},
      eprint={2012.06106},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

```
