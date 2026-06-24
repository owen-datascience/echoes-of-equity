# VII. Conclusion

Voice-based trustworthy-intent classification on the publicly available TIS Corpus [1] has plateaued at approximately 71% accuracy, and the published Random Forest baseline misclassifies South Asian speakers at a 5-percentage-point higher rate than White speakers. This bias propagates into any downstream voice-AI that surfaces a trust score, with direct social cost in applications from customer-service routing to recruitment screening, and motivated the present study.

We made three contributions on this corpus.

1. **Faithful reproduction.** Under a joint $(\text{intent} \times \text{ethnicity})$ stratified $80/20$ split, our Random Forest reaches $73.4 \pm 1.0\%$ accuracy and our Logistic Regression reaches $71.9 \pm 0.0\%$, recovering the source paper's published $71\%$ and $69\%$ within $\pm 2$ percentage points and validating the comparison ground.

2. **Deep-model accuracy and fairness lift.** A three-layer ANN reaches $76.5 \pm 2.5\%$ accuracy and a 1D-CNN reaches $75.4 \pm 0.9\%$ — both clearly above the published baseline. More importantly, both deep models more than halve the per-ethnicity accuracy gap (RF $12.1$pp / LR $11.3$pp $\to$ CNN $5.0$pp), recovering the source paper's published $5$pp gap *without any fairness-specific machinery*. The biggest beneficiary is the South Asian group, whose mean accuracy rises by approximately $8$ percentage points absolute (RF $66.6\%$ $\to$ CNN $74.9\%$ / DANN $75.2\%$). The non-linear capacity of a modestly sized neural network is, on this dataset, sufficient to remove most of the demographic accuracy gap on its own.

3. **Adversarial reproducibility.** DANN-Trust, our domain-adversarial network with a gradient-reversal layer, matches the simpler deep models on mean accuracy ($75.0\%$) and mean fairness gap ($5.03$pp). Its distinct contribution is *variance reduction*: across five random initialisations, the fairness-gap standard deviation drops from CNN's $2.87$pp to **$0.89$pp**, and the accuracy standard deviation drops from ANN's $2.53\%$ to **$0.88\%$**. The adversarial constraint trades a fraction of a point of peak accuracy for fairness behaviour that is predictable across retrains.

In one sentence: across five random seeds, deep models lift accuracy from $73\%$ (RF baseline) to $75$–$76\%$ and shrink the cross-ethnicity gap from $12$pp to approximately $5$pp, matching the source paper's published gap; among deep models, DANN-Trust uniquely delivers this fairness improvement with roughly one-third the run-to-run variance.

These results matter because voice-based AI is increasingly deployed in customer-service routing, fraud detection, recruitment screening and even courtroom evidence. A model that systematically misjudges Black or South Asian speakers as "less trustworthy" causes direct harm. Naively dropping demographic columns is insufficient because the model can re-learn demographic shortcuts from acoustic correlates. Our findings show that depth and non-linearity alone close most of the per-ethnicity accuracy gap and that adversarial training adds the further property of *predictable* fairness behaviour across retrains. The second property is what makes a model deployable in regulated settings where retraining must produce reliable results.

Five concrete directions extend this work:

* **Listener-perception loop.** Pair every utterance with crowd ratings of *perceived* trust to learn the production–perception alignment, addressing a limitation acknowledged by the source paper.
* **Cross-corpus generalisation.** Re-evaluate DANN-Trust on a second, non-overlapping voice-trust corpus (for example, a Mandarin or Latino voice-trust dataset, once available) to confirm that the variance-reduction property is not corpus-specific.
* **End-to-end raw-audio models.** Replace the hand-crafted 60-feature input with a self-supervised audio encoder (wav2vec2 or HuBERT) and retrain the DANN heads on top; the GRL transfers without modification.
* **More demographic axes.** Add native language / accent and socioeconomic proxies as further adversarial heads; the joint loss generalises to any finite number of protected attributes by simply adding one $-\lambda \cdot \mathcal{L}_{\text{dom}}$ term per attribute.
* **Deployment-time fairness audit.** Release a small Python package that any voice-AI team can run on its own held-out data to measure $\Delta_{\text{eth}}$, $\Delta_{\text{age}}$ and $\Delta_{\text{sex}}$ on the model of its choice, lowering the cost of routine fairness checks.

Code, trained checkpoints, per-seed metrics and the joint-stratification protocol are publicly released to support reproducible voice-AI fairness research.
