# SimCLR
A PyTorch Implementation Of SimCLR<br>
### Introduction: 
SimCLR, short for "Simple Contrastive Learning of Visual Representations," is a powerful self-supervised learning framework for learning high-quality image representations without requiring manual labels.It leverages contrastive learning, where the model is trained to pull together similar images and push apart dissimilar ones in a learned feature space.
<p align="center">
  <img src="https://github.com/Spijkervet/SimCLR/blob/master/media/architecture.png?raw=true" width="500"/>
</p>
[Link to paper](https://arxiv.org/pdf/2002.05709.pdf)
### Folder Structure:
#!/bin/bash

#File: tree-md

tree=$(tree -tf --noreport -I '*~' --charset ascii $1 |
       sed -e 's/| \+/  /g' -e 's/[|`]-\+/ */g' -e 's:\(* \)\(\(.*/\)\([^/]\+\)\):\1[\4](\2):g')

printf "# Project tree\n\n${tree}"


