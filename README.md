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

If you extend or use this work, please cite the [paper](https://www.semanticscholar.org/paper/EQG-RACE%3A-Examination-Type-Question-Generation-Jia-Zhou/f84b531135acc19191310537065a804c00814cdd) where it was introduced:

```
@inproceedings{Jia2021EQGRACEEQ,
  title={EQG-RACE: Examination-Type Question Generation},
  author={Xin Jia and Wenjie Zhou and Xu Sun and Yunfang Wu},
  booktitle={AAAI},
  year={2021}
}

```
