# WaveNetText

Inspired by WaveNet, we use dilated convolutions in a latent variable generative model for text. 

### Dilated convolutions

From WaveNet: 

*"A dilated convolution is a convolution where the filter is applied over an area larger than its length by skipping input values with a certain step. It is equivalent to a convolution with a larger filter derived from the original filter by dilating it with zeros, but is significantly more efficient."*

### Model details

An example graphical model with 3 layers and sequences of length 6 is shown below, although to avoid clutter we exclude the latent variable from the figure. In reality, there is a latent node **z** which is connected to all of the **h** nodes.

<img src="_paper/figures/graphical_model.png">

The form of the hidden states is shown below, where * indicates a convolution:

<img src="_paper/figures/hid_state.png" height="60">

The dashed lines indicate that:

<img src="_paper/figures/prob_obs.png" height="100">

The experiments are trained using Wikipedia data, with a vocabulary of 120,000 words and sentences with a maximum length of 30 words. As per the original WaveNet paper, the generative model uses stacks of dilations in the pattern 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, ...

For the variational distribution, we have tried using:

* A simple feedforward network, taking the concatenation of the word embeddings **x**<sub>1</sub> ... **x**<sub>L</sub> as input.
* A WaveNet style network, where the final hidden layer is passed through a feedforward network to produce the mean and variance of the variational distribution.

### Example generations

Below, the lines `gen x sampled` are constructed by sampling a word from the distribution defined by **y**<sub>l</sub> at each time step, whereas the lines `gen x argmax` are constructed by taking the mode of that distribution at each time step.

The first 5 sentences sample **z** from the prior p(**z**), and the last 5 sentences sample **z** from the variational distribution q(**z**|**x**), where **x** is the sentence indicated by `true x`.

```
gen x sampled: the village has an approximate population of 70 . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> prescott . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> .
 gen x argmax: <UNK> is a village in the administrative district of gmina <UNK> , within <UNK> county , west pomeranian voivodeship , in north-western poland . <EOS> . <EOS> . <EOS> .
----------
gen x sampled: his ship disappeared . <EOS> bien british reportedly followed a four-star black and winning five <UNK> to the united states , specifically in the roman empire circuit . <EOS> .
 gen x argmax: <UNK> <UNK> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS>
----------
gen x sampled: cnn hosted the annual meeting ( commonly known as personal executive ) created . opera coverage can also be expected to expansion . <EOS> . <EOS> <EOS> 2010. <EOS> <EOS>
 gen x argmax: the school has been a member of the national association of the australian schoolboys team . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> .
----------
gen x sampled: diplomatic relations . <EOS> . <EOS> . <EOS> . <EOS> . guyana 1992 , 1949 . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> .
 gen x argmax: <UNK> <UNK> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS>
----------
gen x sampled: a means classical word for `` <UNK> <UNK> '' than them <UNK> ( <UNK> ) are full-length during world temperature . <EOS> naha india . <EOS> . <EOS> . <EOS>
 gen x argmax: <UNK> <UNK> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS>
----------
==========
==========
       true x: simple gifts . <EOS>
gen x sampled: hans <UNK> . <EOS> . <EOS> . santa gertrudis , <UNK> south manila south west . <EOS> . <EOS> . <EOS> . <UNK> <EOS> . <EOS> . <EOS> . <EOS>
 gen x argmax: <UNK> <UNK> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS>
----------
       true x: new age revival . <EOS>
gen x sampled: guy charles roberts ( born february 21 , 1976 in glasgow ) is a dutch federal judge . <EOS> . she runs her latter term in australia . <EOS> .
 gen x argmax: <UNK> <UNK> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS>
----------
       true x: full welsh breakfast . <EOS>
gen x sampled: the world is also home to the north west side of the world . <EOS> . <EOS> . <EOS> , he has been related to the <UNK> cricket . <EOS>
 gen x argmax: the <UNK> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS>
----------
       true x: the italian colonial ship `` eritrea '' was constructed in the castellammare shipyards . construction started in 1935 and she was commissioned in 1937 . <EOS>
gen x sampled: hms negev ( <UNK> ) . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> <UNK> . <EOS> , <UNK> was first <EOS> , spanish . <EOS> <EOS> . <EOS>
 gen x argmax: the <UNK> was a french automobile manufacturer of the royal navy . <EOS> . <EOS> . <EOS> , he was interred in the <UNK> river . <EOS> . <EOS> .
----------
       true x: morris , darryl . <EOS>
gen x sampled: <UNK> competed in the 2003 winter olympics in seoul . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> .
 gen x argmax: <UNK> <UNK> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS> . <EOS>
```

### Current problems

* Whilst the model can produce coherent sentences, the KL divergence from the variational distribution q(**z**|**x**) to the prior p(**z**) tends towards 0, which means that the model ignores the latent variable. This means that the range of sentences produced is not diverse.
  * As well as using balanced generative and recognition models (i.e. with 3 stacks of 1, 2, 4, 8, 16 dilations in both models), we have tried using only a single stack of dilated convolutions in the generative model with 3 stacks in the recognition model, but this does not seem to help with increasing KL[q(**z**|**x**) || p(**z**)].
