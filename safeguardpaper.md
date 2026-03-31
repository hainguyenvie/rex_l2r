## SafeGround: Know When to Trust GUI Grounding Models

## via Uncertainty Calibration

```
Qingni Wang* 1 2 Yue Fan* 2 Xin Eric Wang^1
```
## Abstract

```
Graphical User Interface (GUI) grounding aims
to translate natural language instructions into
executable screen coordinates, enabling auto-
mated GUI interaction. Nevertheless, incorrect
grounding can result in costly, hard-to-reverse
actions (e.g., erroneous payment approvals), rais-
ing concerns about model reliability. In this pa-
per, we introduce SAFEGROUND, an uncertainty-
aware framework for GUI grounding models
that enables risk-aware predictions through cal-
ibrations before testing. SAFEGROUND lever-
ages a distribution-aware uncertainty quantifica-
tion method to capture the spatial dispersion of
stochastic samples from outputs of any given
model. Then, through the calibration process,
SAFEGROUND derives a test-time decision thresh-
old with statistically guaranteed false discovery
rate (FDR) control. We apply SAFEGROUND on
multiple GUI grounding models for the challeng-
ing ScreenSpot-Pro benchmark. Experimental
results show that our uncertainty measure con-
sistently outperforms existing baselines in distin-
guishing correct from incorrect predictions, while
the calibrated threshold reliably enables rigorous
risk control and potentials of substantial system-
level accuracy improvements. Across multiple
GUI grounding models, SAFEGROUND improves
system-level accuracy by up to 5.38% percentage
points over Gemini-only inference.
```
## 1. Introduction

```
Graphical User Interface (GUI) grounding is a critical
component for autonomous GUI agents, enabling vision-
language models (VLMs) to translate natural language in-
```
*Equal contribution (^1) University of California, Santa Barbara
(^2) University of California, Santa Cruz. Correspondence to: Xin
Eric Wang <ericxwang@ucsb.edu>.
Preprint. February 4, 2026.
Query (𝐪): Remove the utensils from my shopping cart.
Due to uncertainty,
I abstain from
immediate
prediction.
Consulting external
knowledge ...
Click (232, 733)
User
Existing GUI grounding model
SafeGround
(ours)
Identify as a highenforcing user-specified -risk action, risk-level: 𝜶.
Incorrect grounding
Click (235, 834 )
User’s
Screen
Correct grounding
Figure 1. While existing models may commit costly errors on
hard-to-undo actions (e.g., checkout), SAFEGROUND detects high
uncertainty and defers the decision via cascading. This mechanism
explicitly limits the risk of erroneous actions to a user-specified
tolerance.
structions into executable screen coordinates (Nguyen et al.,
2025; Cheng et al., 2024). Recent advances have substan-
tially improved grounding accuracy across diverse GUI
environments, making it increasingly feasible to deploy
such agents in real-world applications (Fan et al., 2025;
Hong et al., 2024). However, in practical GUI interactions,
a single incorrect grounding can trigger costly and hard-
to-reverse actions, including erroneous payment approvals
or irreversible system configurations (Zhang et al., 2025).
Despite these risks, existing GUI grounding models typi-
cally output only point predictions, offering no indication of
when a prediction is unreliable or should be deferred (Gaw-
likowski et al., 2022; Hu et al., 2023) as shown in Figure 1.
The aforementioned limitation of existing GUI grounding
models motivates the incorporation of uncertainty quantifi-
cation (UQ) to enable safer decision-making. However,
existing UQ techniques are poorly suited for GUI ground-
ing and remain largely underexplored in this setting (Zhang
et al., 2025). In particular, prior approaches suffer from
several key limitations. (1) Uncertainty derived from model
probabilities or logits (Hendrycks & Gimpel, 2017) assumes
access to internal model states, making it infeasible for

# arXiv:2602.02419v2 [cs.AI] 3 Feb 2026


```
Click on the “Copy Path” button.
```
```
User’s
screen
```
```
Sure.try multiple Let me
times.
```
```
HighLow Entropy Concentration
```
```
Ranked Region Scores
```
```
High Uncertainty Warning!
```
```
Reliable Prediction!
```
```
Low-Uncertainty Scenario
```
```
Calibrate a uncertainty threshold 𝜏 based
on user-specified risk level 𝛼....
```
Execute:uncertainty (^) ≤𝜏
Abstainuncertainty or cascade: >𝜏
Step 2: Selective Prediction Calibration
(342, 1576),(100, 820),^
(1890, 433),(756, 1294),
(1678, 1822)(1210, 98),^
...
Step 1: Uncertainty Quantification
High Entropy
Low Concentration
Ranked Region Scores
High-Uncertainty Scenario
Andto be I nowant more the than risk 30%.level 𝛼
User
query
GUI Model
Low Variance
High Variance
Decision Making
Local GUI grounding
model
Safe(ours)Ground
External resource
Figure 2. Overview of SAFEGROUND. Given a GUI input, the model performs multiple stochastic grounding samples to estimate
predictive uncertainty. An uncertainty thresholdτis calibrated on a held-out set under a user-specified risk level (i.e, the maximum
error rate). At test time, predictions with uncertainty≤ τare executed directly, while high-uncertainty cases are abstained or cascaded.
Low-uncertainty cases exhibit concentrated region scores, low entropy, and low variance, whereas high-uncertainty cases show dispersed
predictions and trigger safety-aware deferral.
black-box vision-language models commonly used in GUI
agents (Ye et al., 2024; Wang et al., 2025b). (2) Verbal-
ized self-assessment (Kadavath et al., 2022) relies on strong
instruction-following behavior and often fails when models
do not explicitly reason about confidence. (3) Approaches
that estimate uncertainty using ground-truth regions, such
as Zhang et al. (2025), require annotation and cannot be
applied at inference time. (4) Existing methods focus on
producing uncertainty scores alone, without specifying how
predictions should be acted upon at deployment time (e.g.,
whether to accept, defer, or abstain) despite this decision
being critical in high-stakes GUI interactions (Geifman &
El-Yaniv, 2017; Wang et al., 2025c). Collectively, these lim-
itations expose a clear gap between existing UQ approaches
and the practical requirements of GUI grounding, where
uncertainty must be reliable under limited model access and
without test-time supervision (Lin et al., 2023).
To address these challenges, we introduce SAFEGROUND,
an uncertainty-aware framework that enables risk-aware pre-
dictions for existing state-of-the-art GUI grounding models,
without requiring access to model internals. Concretely,
as shown in Figure 2, SAFEGROUND first quantifies the
predictive uncertainty of grounding outputs from the spa-
tial distribution of multiple stochastic grounding samples
from the same model. Then, given the model outputs with
estimated uncertainty, we adopt a Learn Then Test (LTT)
calibration paradigm to select a decision threshold that rig-
orously controls the false discovery rate (FDR) of accepted
grounding predictions. This calibration procedure provides
finite-sample guarantees: with high probability, the pro-
portion of incorrect predictions among all accepted actions
does not exceed a user-specified risk levelα. At infer-
ence time, SAFEGROUND enables a principled selective
prediction mechanism. Predictions deemed reliable under
the calibrated threshold are executed directly, while high-
uncertainty cases are abstained from or deferred to stronger
models for further processing. Furthermore, with the selec-
tive prediction, we realized the cascading inference, where
even when the primary model’s base accuracy is limited, we
can further leverage external resource to aid the prediction,
achieving strong system-level accurarcy.
We evaluate SAFEGROUND on the challenging ScreenSpot-
Pro benchmark across multiple state-of-the-art GUI ground-
ing models. Experimental results demonstrate that our pro-
posed uncertainty measure consistently outperforms existing
baselines in distinguishing correct from incorrect predic-
tions. Especially, SAFEGROUND achieves reliable FDR
control in practice and significantly improves overall sys-
tem accuracy through selective deferral, validating its ef-
fectiveness for high-stakes GUI interaction scenarios. Em-
pirically, SAFEGROUND demonstrates clear system-level
accuracy gains across different risk levels. For instance, on
ScreenSpot-Pro, uncertainty-aware cascading with Holo1.5-
7B achieves 58.66% accuracy at risk level 0.34, improving
over Gemini-only inference by 5.38% points. Our contribu-
tions can be summarized as follows:

- We propose SAFEGROUND, the first framework for
    uncertainty-aware selective GUI grounding with finite-
    sample risk guarantees via calibration.
- We introduce distribution-aware uncertainty quantifica-
    tion that leverage the spatial dispersion and concentra-
    tion of stochastic grounding predictions.
- We demonstrate that SAFEGROUND with uncertainty-
    calibrated selective prediction enables reliable FDR


```
control and improves system-level accuracy in cascad-
ing inference on the ScreenSpot-Pro benchmark.
```
## 2. Related Work

2.1. GUI Grounding

GUI grounding maps natural language instructions to ac-
tionable interface elements or click locations in graphical
user interfaces (Nguyen et al., 2025; Fan et al., 2025). Most
existing GUI grounding methods formulate the problem as
a text-based coordinate prediction task, where models gen-
erate point locations conditioned on the input screenshot
and instruction (Chen et al., 2023; Wang et al., 2024a; Qin
et al., 2025a). Recently, motivated by how humans interact
with digital interfaces, GUI-Actor introduces an attention-
based formulation that aggregates spatial evidence into a
single grounding decision (Wu et al., 2025b). These meth-
ods have achieved strong empirical accuracy across diverse
GUI environments. However, most existing approaches pro-
duce deterministic point predictions and do not explicitly
model predictive uncertainty, limiting their ability to assess
decision reliability or defer actions under high uncertainty.

2.2. Uncertainty Estimation

Uncertainty estimation is widely used to support reliable
decision making in AI systems by quantifying the confi-
dence of model predictions (Liu et al., 2025). In large
language models, uncertainty has also been derived from
probabilistic measures, semantic entropy, or verbalized self-
reports (Hou et al., 2025; Wang et al., 2024b; Xu et al.,
2025; Kuhn et al., 2023b). In GUI grounding, uncertainty
estimation remains largely underexplored. Existing GUI
grounding approaches typically rely on probabilistic uncer-
tainty or verbalized uncertainty, both of which have been
shown to be systematically miscalibrated, exhibiting a mis-
match between predicted confidence and actual grounding
accuracy (Zhang et al., 2025). This misalignment moti-
vates uncertainty estimation methods that rely solely on
model outputs while providing more reliable signals for
downstream decision-making, as considered in our work.

2.3. Learn then Test Calibration

Learn Then Test (LTT) is a post-hoc calibration paradigm
that separates model learning from statistical risk con-
trol (Angelopoulos et al., 2022). Given a fixed predictive
model, LTT frames decision making as a hypothesis testing
problem over a low-dimensional decision space, and uses
held-out calibration data to identify parameters that satisfy
user-specified risk constraints with finite-sample guarantees.
Split conformal prediction (SCP) (Angelopoulos & Bates,
2022) follows this principle by leveraging data splitting and
concentration-based confidence bounds to perform valid risk

```
estimation. Prior work builds on this paradigm to enable
reliable decision making in large foundation models (Jung
et al., 2025; Wang et al., 2025a;c; Wang et al.). Our ap-
proach also builds on the LTT paradigm and extends it to
GUI grounding through uncertainty-based calibration of
spatial action decisions for the first time.
```
## 3. Methodology

```
3.1. Problem Formulation and Notations
Let the GUI grounding model be a functionf :X ×T →
R^2 , which takes a UI screenshotx∈Xand a user instruc-
tionq ∈ T as input. Given an input pair(x,q), the model
predicts a coordinateˆ = (ˆy u, ˆ)v ∈ R^2 on the screen. Al-
though the model produces a single point prediction, the
ground truth for a target UI element is typically provided
as a spatial region on the screen, denoted byB∗⊂ R^2. A
predicted coordinate is considered correct if and only if it
falls within the ground-truth region, which we conclude as
an admission functionA : R^2 ×P(R^2 ) → { 0 , 1 }with 1
indicating a correct prediction:
```
### A

### 

```
ˆy,B∗
```
### 

### =

### (

```
1 , if ˆy ∈ B∗,
0 , otherwise.
```
```
In current coordinate-based GUI grounding models, predic-
tions are deterministic and are not accompanied by explicit
uncertainty or confidence estimates, which leaves the trust-
worthiness of model outputs largely uncharacterized, and
may cause users to place unwarranted trust in incorrect pre-
dictions, without any indication of potential failure.
```
```
3.2. Method Overview
```
```
To address this issue, we propose SAFEGROUND, an
uncertainty-aware GUI grounding framework that can be in-
tegrated with diverse state-of-the-art GUI grounding models
without requiring access to internal model states, as illus-
trated in Figure 2. SAFEGROUND introduce a user-specified
risk levelα∈ (0, 1)that quantifies the maximum tolerable
proportion of incorrect predictions, serving as a high-level
control signal for how conservatively the system should
behave. The risk levelαis then translated into an uncer-
tainty thresholdτthrough a calibration procedure. Specif-
ically, the GUI grounding model’s predictive uncertainty,
U
```
### 

```
ˆ(MLG)y
```
### 

```
∈ Rfor a predictionˆy(MLG), is estimated by
SAFEGROUND through sampling multiple additional predic-
tions from the GUI grounding model given the same input.
The larger values of such uncertainty score indicate lower re-
liability. A predictionˆyis correct ifU(ˆ)y≤ τand rejected
otherwise, in which case it is deferred to a stronger model.
The thresholdτis chosen such that, among all admitted
predictions, the fraction of incorrect ones, measured by the
admission function A(ˆy,B∗), is controlled below α.
```

3.3. Uncertainty Quantification

We first quantify model uncertainty by analyzing the distri-
butional properties of the ranked region scores. Then, three
complementary uncertainty measures are introduced, where
they are designed to capture complementary failure modes
of GUI grounding: local ambiguity among competing tar-
gets, global dispersion of belief across regions, and lack of
dominant spatial concentration.

Sampling-Based Spatial Distribution Construction To
move beyond deterministic point predictions and capture
the output distribution of GUI grounding models, we em-
ploy a Monte Carlo (Gal & Ghahramani) sampling strat-
egy followed by spatial aggregation, drawing inspiration
from attention-based aggregation mechanisms in (Wu et al.,
2025b). Specifically, for each input(x,q), we performK
stochastic forward passes of the grounding model, generat-
ing a set of coordinatesS ={ˆy(i)}Ki=1, where ˆy(i)∈ R^2.

These sampled coordinates are then projected onto a dis-
cretized screen grid to estimate a normalized local density
mapP, which empirically characterizes the spatial distri-
bution of the model’s predictions using only sampled out-
puts from the model. Intuitively, high density in a local-
ized area indicates model consistency and thus low uncer-
tainty. To establish object-level representations, we aggre-
gate connected high-density patches inPinto disjoint re-
gionsR = {Rm}Mm=1through density-based clustering.
Each regionRmis scored by its average probability density,
denoted asSm, serving as a proxy for the likelihood that the
region corresponds to the intended UI element. Regions are
further ranked such thatS(1)≥ S(2)≥···≥ S(M). More
implementation details are provided in the Appendix B.3.

Uncertainty Measurement 1. Top-Candidate Ambiguity
(TA). To measure the distinctiveness of a certain predic-
tion from a GUI grounding model, we compute the margin
between the two leading candidates. A vanishing margin
indicates that the model is uncertain between multiple plau-
sible targets (e.g., two identical exit buttons), therefore, we
propose the uncertanty score measured by top-candidate
ambiguity:

### UTA=

### (

### 1 −

```
S(1)−S(2)
S(1)+ε ,^ if M ≥^2
max(0. 1 , 1 − S(1)), otherwise
```
### (1)

whereεensures numerical stability. HighUTAsignifies
localized confusion at the decision boundary.

Uncertainty Measurement 2. Informational Dispersion
(IE). We assess global uncertainty using the entropy of
the region score distribution. To ensure a valid probabilistic
interpretation, we induce a categorical distribution over the

```
M regions:
ˆpi=
```
```
S(i)
PM
j=1S(j)
```
### , (2)

```
and then we define the uncertainty score based on informa-
tion dispersion as the normalized entropy:
```
### UIE=−

### 1

```
logM
```
### XM

```
i=
```
```
ˆpilog(ˆip + ε). (3)
```
```
Such measurement captures the dispersion of probability
mass across regions; a highUIEindicates that the model’s
confidence is fragmented, failing to converge on a single
consistent hypothesis.
```
```
Uncertainty Measurement 3. Concentration Deficit (CD).
While entropy assesses global disorder, we explicitly quan-
tify the lack of focus with another uncertainty scoreUCDby
examining the quadratic concentration of the distribution:
```
### UCD= 1−

### XM

```
i=
```
```
ˆ^2 ip (4)
```
```
Unlike entropy,UCDis more sensitive to the dominance of
the top candidates. Higher values ofUCDindicate a highly
fragmented distribution, suggesting that the model lacks a
clear spatial focus and distributes confidence across multiple
interface regions.
```
```
Combined Uncertainty Score. Each uncertainty score
captures a distinct aspect of predictive dispersion, and no sin-
gle measurement is universally dominant across all models
and scenarios. To obtain a unified and deployment-friendly
uncertainty signal, we aggregate these three scores into a
single one via a fixed weighted combination:
```
```
UCOM(ˆ) =y wCD·UCD+wIE·UIE+wTA·UTA. (5)
```
```
We adopt a single set of weights across all models to pre-
serve a plug-and-play interface without model-specific tun-
ing.
```
```
3.4. Uncertainty Calibration for Selective Prediction
Although the proposed uncertainty measures capture pre-
dictive uncertainty, they cannot fully distinguish between
correct and incorrect predictions. To enable user-specified
deployment, we further introduce a selective prediction
mechanism by calibrating a statistically rigorous decision
thresholdτon the uncertainty score, such that, among all
accepted predictions, the proportion of incorrect predictions
does not exceed a desired level α.
Following prior SCP-based frameworks, we hold out a cali-
bration set ofNdata points:Dcal={(xi,qi,Bi∗)}Ni=1. For
```

each calibration input pair(xi,qi), we produceˆ(iyMLG)and

quantify its uncertainty scoreui= U

### 

```
ˆ(iyMLG)
```
### 

. Given a

candidate thresholdτ, we obtain the number of accepted
predictions

### PN

i^1 {ui≤ τ}, and the number of incorrect
predictions

### PN

```
i^1 {ui ≤ τ,A(ˆy
```
(MLG)
i ,B
∗
i) = 0}. We
then compute the false discovery rate (FDR) onDcalunder
threshold τ :

```
FDRcal(τ) =
```
### PN

```
i^1 {ui≤ τ,A(ˆy
```
```
(MLG)
i ,B
∗
i) = 0}
PN
i^1 {ui≤ τ}
```
### (6)

To provide finite-sample FDR guarantees for the accepted
samples at test time, we first introduce an auxiliary lemma.

Lemma 3.1 (Clopper–Pearson interval (Clopper & Pearson,
1934)). LetX ∼ Bin(n,p)be the number of successes in
ni.i.d. Bernoulli trials with success probabilityp. For any
δ ∈ (0, 1), define the Clopper-Pearson confidence interval

```
h
pL(X), pU(X)
```
```
i
=
```
```
h
Beta−^1
```
```
δ
2 ; X, n− X + 1
```
### 

### ,

```
Beta−^1
```
### 

```
1 −δ 2 ; X + 1, n− X
```
```
i,
```
```
(7)
```
whereBeta−^1 (q;a,b)denotes theq-quantile from a beta
distribution with shape parametersaandb. Then the inter-
val has (at least) nominal coverage:

```
P(p∈ [pL(X),pU(X)])≥ 1 − δ. (8)
```
In our setting,X =

### PN

```
i^1 {ui≤ τ,A(ˆy
```
(MLG)
i ,B
∗
i) = 0}
andn =

### PN

i^1 {ui≤ τ}. Since we focus on controlling
the upper tail of the system FDRR(τ)(thereby constrain-
ing test-time FDR), based on Lemma 3.1, we construct
a high-probability upper confidence bound,FDRˆ

upper
1 −δ (τ),
forR(τ), using its empirical estimate from the calibration
data:

```
FDRˆ upper 1 −δ (τ) = Beta(1− δ; X + 1, n− X)
= sup{R : Pr(Bin(n,R)≤ X)≥ δ}
```
### , (9)

whereFDRˆ

```
upper
1 −δ guarantees
```
```
Pr
```
### 

```
R(τ)≤FDRˆ
```
```
upper
1 −δ (τ)
```
### 

```
≥ 1 − δ. (10)
```
Essentially,FDRˆ

upper
1 −δ (τ)can be interpreted as the largest
plausible value that the system FDR could take, given that an
extremely smallFDRcal(τ)is observed on the calibration
set at significance levelδ. If the true system FDR were to
exceed this bound, then observingFDRcal(τ)in a single
realization would be statistically impossible at the levelδ.
A formal proof of Eq. (10) is provided in Appendix A.

```
To rigorously constrain test-time FDR, we calibrateτsuch
thatFDRˆ
```
```
upper
1 −δ (τ) does not exceed the risk level α:
```
```
ˆ = supτ {τ :FDRˆ
```
```
upper
1 −δ (τ)≤ α}^ (11)
The choice ofˆτmaximizes the acceptance of model predic-
tions (or minimizes the abstention rate), while maintaining
marginal FDR control. For a test sample(xtest,qtest,B∗test)
with the model predictionˆtesty(MLG)and estimated uncertainty
scoreutest= U
```
### 

```
ˆy(testMLG)
```
### 

```
, by applying the calibrated de-
cision threshold ˆ , we establish the following guaranteeτ
```
```
Pr
```
### 

```
Pr
```
### 

### A

### 

```
ˆ(testyMLG),Btest∗
```
### 

```
= 0| utest≤ ˆτ
```
### 

```
≤ α
```
### 

```
≥ 1 −δ.
(12)
```
```
Cascading Inference. At inference time, for each test
input(xtest,qtest), we first estimate the model uncertainty
utest, and then perform selective prediction and escalating:
```
- Ifutest≤ ˆτ, we define the sample as “safe” and accept
    the prediction of the primary model.
- Ifutest> ˆτ, we flag the sample as “risky” and escalate
    the input to a stronger model to enhance performance.

## 4. Experiment

```
4.1. Experimental Settings
```
```
Models and Dataset We conduct our experiments over
6 GUI-grounding models, including Holo1.5 (Company,
2025), GUI-Actor (Wu et al., 2025a), UI-TARS-1.5 (Qin
et al., 2025b), GTA1 (Yang et al., 2025): Holo1.5-3B,
Holo1.5-7B, GUI-Actor-2VL-7B, GUI-Actor-2.5VL-7B,
UI-TARS-1.5-7B and GTA1-7B. To assess reliability under
high-stakes scenarios, we conduct all experiments on the
challenging ScreenSpot-Pro (Li et al., 2025) benchmark.
Additional dataset details are provided in the Appendix B.1.
```
```
Evaluation Metrics To comprehensively evaluate both
the discriminative ability of UQ methods and the reliability
and effectiveness of SAFEGROUND, we adopt four comple-
mentary metrics: Area Under Receiver Operating Charac-
teristic (AUROC), Area Under Accuracy-Rejection Curve
(AUARC), FDR, and power (Lin et al., 2024; Wang et al.,
2025c). AUROC measures the ability of uncertainty es-
timates to distinguish correct from incorrect predictions,
while AUARC evaluates whether prediction accuracy im-
proves as high-uncertainty samples are progressively re-
jected. FDR quantifies the proportion of incorrect pre-
dictions among the accepted samples. Power measures
the proportion of correct samples that are retained after
uncertainty-based selection, relative to the total number of
correct samples. More details about the metrics can be
found in Appendix B.2.
```

Hyperparameters For uncertainty estimation, we sample
each input 10 times with the decoding temperature set to 1.
to compute the corresponding UQ score. The most likely
generationˆiy(MLG)is obtained by uniformly sampling one
output from the generated candidates. Specifically, when
computing UQ scores, we partition the input into patches
with a patch size of 14 to obtain region-level scoresSi
for uncertainty estimation. We repeat the random calibra-
tion–test split 100 times and report the mean and standard
deviation (mean±std) over all runs. All confidence bounds
are constructed at a significance level ofδ = 0. 05. For the
combined uncertainty scoreUCOM, we use a fixed weight-
ing scheme(wCD,wIE,wTA) = (0. 6 , 0. 2 , 0 .2)across all
models.

4.2. Evaluation of Uncertainty Estimation

Following prior work (Kuhn et al., 2023a; Band et al., 2022),
we evaluate the quality of uncertainty estimates using AU-
ROC and AUARC, which measure the discriminative ability
of uncertainty scores and their effectiveness for selective
prediction, respectively. We compare our distribution-aware
uncertainty with the probabilistic confidence (PC) baseline,
defined as one minus the average token probability (Pouget
et al., 2016).

Table 2 reports AUROC results across six GUI grounding
models. When PC is available, our method consistently
achieves higher AUROC. , and on Holo1.5-7B from 0.
to 0.7526. For models where PC is not directly applicable
(e.g., GUI-Actor variants), our method still attains strong
AUROC values (up to 0.8155), demonstrating robust er-
ror discrimination under limited model access. Overall,
these results suggest that modeling the spatial distribution of
grounding predictions yields more informative uncertainty
signals than token-level confidence alone.

We further evaluate uncertainty quality using AUARC,
which captures accuracy gains as high uncertainty predic-
tions are progressively rejected. As shown in Table 3,
our method consistently outperforms baselines across mod-
els. For example, on Holo1.5-3B, AUARC improves from
0.6444 to 0.6576 compared to PC. These results indicate
that our uncertainty estimates are particularly effective for
guiding selective prediction decisions.

4.3. Selective Prediction with FDR Guarantees

While AUROC and AUARC evaluate the quality of un-
certainty estimates, reliable deployment further requires
translating these scores into principled decision rules with
explicit risk guarantees. We therefore study selective predic-
tion under false discovery rate (FDR) control.

```
Table 1. System-level accuracy (%) of uncertainty-calibrated cas-
cading under different risk levels. “–” indicates infeasible risk
levels. Parentheses show∆over the corresponding model baseline
(no cascading). All reported accuracies are computed on the test
split, with a test ratio of 0.8.
Risk Level
Model 0.34 0.38 0.42 0.46 0.
Gemini-only 53.
Holo1.5-7B 52.
(+SAFEGROUND) 58.66 (+ 6.25) 57.87 (+ 5.46) 55.73 (+ 3.32) 53.20 (+ 0.79) 52.41 (+ 0.00)
Holo1.5-3B 45.
(+SAFEGROUND) 53.44 (+ 7.99) 52.73 (+ 7.28) 52.02 (+ 6.57) 49.25 (+ 3.80) 47.35 (+ 1.90)
UI-TARS-1.5-7B 41.
(+SAFEGROUND) 53.68 (+12.10) 54.70 (+13.12) 53.04 (+11.46) 50.43 (+ 8.85) 47.91 (+ 6.33)
GUI-Actor-2.5VL-7B45.
(+SAFEGROUND) 55.18 (+ 9.49) 54.86 (+ 9.17) 53.60 (+ 7.91) 51.38 (+ 5.69) 49.17 (+ 3.48)
GUI-Actor-2VL-7B 40.
(+SAFEGROUND) 55.18 (+14.39) 53.28 (+12.49) 53.99 (+13.20) 52.96 (+12.17) 50.67 (+9.88)
GTA1-7B 46.
(+SAFEGROUND) – – – 53.12 (+ 6.24) 49.96 (+ 3.08)
```
```
Table 2. AUROC comparison of uncertainty quantification meth-
ods across different models. The best results for each model are
highlighted in bold. PC is the Probabilistic Confidence baseline.
```
```
Model Uncertainty Score
PC UCOM(Ours)
Holo1.5-3B 0.7576 0.
Holo1.5-7B 0.6983 0.
GUI-Actor-2.5VL-7B - 0.
UI-TARS-1.5-7B 0.7844 0.
GUI-Actor-2VL-7B - 0.
GTA1-7B 0.6114 0.
```
```
FDR Control Guarantee For each uncertainty method
and risk level, we calibrate a decision threshold on the cal-
ibration set using the Clopper–Pearson upper confidence
bound (Clopper & Pearson, 1934), ensuring that the test-
time FDR does not exceed the specified risk level with high
probability. Figure 3 illustrates the empirical FDR on the
test set across various user-specified risk levels (α). Notably,
the evaluated risk levels start from a minimum attainable
value. This arises because the intrinsic limitations of the
base model and the imperfect discriminative power of un-
certainty estimates may cause some incorrect predictions
to receive relatively low uncertainty scores, making them
inseparable from correct ones by thresholding. As a result,
very stringent FDR requirements may be infeasible to sat-
isfy, as no decision threshold can meet the risk constraint
under such conditions (Wang et al., 2025a). Importantly,
this does not undermine the safety guarantee, as the cali-
bration stage explicitly determines whether a user-specified
risk level is achievable prior to deployment, providing a
principled fail-safe mechanism for high-stakes interactions.
The results in Figure 3 show that for all tested models (e.g.,
Holo1.5, UI-TARS), the actual FDR is consistently bounded
below the theoretical upper bound. This empirically verifies
```

```
Figure 3. Test-time FDR (mean±std) on the ScreenSpot-Pro dataset under different risk levels.
```
Figure 4. Test-time power (mean) of ourUCOMand PC baseline on
the ScreenSpot-Pro dataset under different risk levels.

Table 3. AUARC comparison of uncertainty quantification meth-
ods across different models. The best results for each model are
highlighted in bold.

```
Model Uncertainty Score
Random PC UCOM(Ours)
Holo1.5-3B 0.4706 0.6444 0.
Holo1.5-7B 0.5345 0.6686 0.
GUI-Actor-2.5VL-7B 0.4662 – 0.
GUI-Actor-2VL-7B 0.4130 – 0.
UI-TARS-1.5-7B 0.4231 0.6222 0.
GTA1-7B 0.4769 0.5521 0.
```
that SAFEGROUND provides rigorous safety guarantees,
ensuring that, with high probability, the error rate among
accepted predictions is controlled at the specified level.

Power Comparison In addition to FDR, we report power
to further characterize the effectiveness of selective predic-
tion. Higher power indicates that the uncertainty estimates
more precisely identify truly risky cases, allowing the sys-
tem to retain a larger set of reliable predictions without
violating the target FDR. Figure 4 compares the power of
our methodUCOMversus the PC baseline under identical
risk levels. Across the evaluated models,UCOMdemon-
strates superior robustness, particularly at strict risk levels
(e.g., 0.38) where PC often fails to yield valid predictions.
Notably, the minimum attainable risk level at which PC can
satisfy the FDR constraint is consistently higher than that of
UCOM, indicating a narrower feasible operating range for PC.
UCOMconsistently outperforms PC, retaining a significantly

```
larger volume of correct responses. These results indicate
thatUCOMis systematically less conservative than PC: it
accepts a larger fraction of correct predictions while still
satisfying the same FDR constraint.
```
```
4.4. Cascading Inference
Finally, we study the system-level benefits of uncertainty-
aware decision making in a cascaded inference setting.
Given that powerful external models (e.g., Gemini) often
incur latency and financial costs, our goal is to improve
system accuracy by selectively invoking stronger models
when the uncertainty of the base model exceeds a calibrated
threshold. Specifically, we fix the calibration split ratio to
0.2 and use the remaining 80% of the data as the test set
to evaluate the cascaded system. At test time, predictions
with uncertainty scores below or equal to the threshold are
handled by the primary local grounding model, while high-
uncertainty cases are deferred to the stronger expert model,
Gemini-3-pro (Team et al., 2023).
Table 1 reports the accuracy of uncertainty-aware Gemini
cascading under different risk levels. Across a wide range
of feasible risk levels, the proposed approach consistently
improves system accuracy over both Gemini-only infer-
ence and the base models, demonstrating the effectiveness
of uncertainty-aware cascading. At relatively small risk
levels, uncertainty-aware cascading yields substantial ac-
curacy gains. For instance, with Holo1.5-7B at risk level
0. 34 , the system achieves 58 .66%accuracy, outperforming
Gemini-only inference by 5 .38%. As the risk level increases,
the improvement gradually diminishes, since fewer high-
uncertainty samples are deferred to Gemini, and the system
behavior approaches that of the base model. The effect is
more pronounced for models such as Holo1.5-3B and UI-
TARS-1.5-7B, where uncertainty-aware cascading improves
accuracy by more than7%to13%over the base models at
relatively small risk levels. We also report the cascading
rate in Figure 5, i.e., the fraction of test samples deferred to
Gemini. As the risk level increases, the cascading rate con-
sistently decreases across all models, indicating that fewer
uncertain cases are escalated to the expert model. This re-
flects the inherent trade-off between accuracy and expert
```

Figure 5. Cascading rate (fraction of test samples deferred to
Gemini) across different risk levels.

Figure 6. Effect of sampling
sizeKon uncertainty estima-
tion quality for UI-TARS-1.5-
7B.

```
Figure 7. Test-time FDR results
of various calibration test split
ratios.
```
invocation cost in uncertainty-aware cascading.

4.5. Sensitivity Analyses

Sampling Efficiency We investigate the trade-off between
computational cost and estimation quality by varying the
sample countKand measuring the resulting AUROC and
AUARC. As shown in Figure 6, increasing the sample size
fromK = 5toK = 10yields a improvement for both
metrics, indicating that the proposed uncertainty estimates
are already effective with a small number of samples. In
contrast, further increasingKfrom 10 to 15 leads to only
marginal changes. Based on this trade-off between perfor-
mance and computational cost, we setK = 10as the default
sampling size in all experiments.

Ablation of Uncertainty Components We analyze the
contribution of individual uncertainty components,UTA,
UIE, andUCD, across different GUI grounding models. As
shown in Table 4, the most informative uncertainty cue is
model-dependent. On GTA1,UTAis the strongest single
signal, whereas for GUI-Actor-2VL and Holo1.5,UCDis
more effective, and UTAalone is insufficient.

Across all models, no single component consistently dom-
inates.UCOMachieves stable performance in all settings,

```
Table 4. Ablation study of uncertainty components on GTA1, GUI-
Actor-2VL, and Holo1.5 models. Best results within each model
block are highlighted in bold.
```
```
Model Uncertainty AUROC AUARC
```
```
GTA
```
```
UTA 0.6228 0.
UIE 0.5916 0.
UCD 0.5917 0.
UCOM 0.6344 0.
w/o UTA 0.5917 0.
```
```
GUI-Actor-2VL-7B
```
```
UTA 0.4844 0.
UIE 0.7731 0.
UCD 0.7894 0.
UCOM 0.8155 0.
w/o UCD 0.7987 0.
```
```
Holo1.5-7B
```
```
UTA 0.6296 0.
UIE 0.7380 0.
UCD 0.7529 0.
UCOM 0.7526 0.
w/o UCD 0.7303 0.
```
```
and removing the dominant component for a given model
leads to a clear drop in both AUROC and AUARC. This
indicates that combining complementary cues yields a more
robust, model-agnostic uncertainty estimate for selective
prediction. Additional robustness analyses with respect to
the uncertainty weighting are provided in the Appendix F.
```
```
Sensitivity to Calibration-Test Split Ratio We further
study the sensitivity of our method to the calibration–test
split ratio when using the combined uncertainty measure
UCOM. Specifically, we vary the proportion of data allocated
to the calibration set while keeping the target risk level
fixed, and evaluate the resulting empirical FDR on the test
set. As shown in Figure 7, across a wide range of split ratios,
the empirical FDR achieved by all three models remains
consistently below the target upper bound. These results
suggest that our approach does not rely on a carefully tuned
split ratio and can be applied robustly in practical settings.
```
## 5. Conclusion

```
We presented SAFEGROUND, an uncertainty-aware frame-
work that enables reliable and risk controlled GUI grounding
under limited model access. By modeling spatial uncer-
tainty from stochastic grounding samples, SAFEGROUND
captures distributional signals that go beyond point predic-
tions and provide effective discrimination between correct
and incorrect predictions. Based on uncertainty estimation,
we further calibrate decision thresholds with finite-sample
guarantees, supporting deployment-time decision making
in high-stakes GUI interactions. Extensive experiments
demonstrate that SAFEGROUND achieves accurate uncer-
tainty discrimination, rigorous FDR control, and improved
system-level performance through selective prediction and
cascading inference. We hope this work provides a princi-
pled way for deploying GUI agents with safety guarantees.
```

## Impact Statement

This paper introduces SAFEGROUND, a framework that sig-
nificantly enhances the reliability and safety of autonomous
GUI agents. By providing the first principled method for un-
certainty quantification in GUI grounding with finite-sample
statistical guarantees, our work addresses a critical bottle-
neck in the real-world deployment of visual agents, the risk
of high-stakes, irreversible errors (e.g., erroneous financial
transactions). Beyond improving individual model reliabil-
ity, the proposed selective deferral mechanism demonstrates
that local models, when combined with uncertainty-aware
cascading to powerful external experts, can achieve superior
system-level accuracy with substantially reduced compu-
tational costs. This research provides a foundational step
toward trustworthy human-AI interaction in digital envi-
ronments, ensuring that automated systems “know when
they don’t know” and make conservative decisions under
ambiguous conditions.

## References

Angelopoulos, A. N. and Bates, S. A gentle introduction
to conformal prediction and distribution-free uncertainty
quantification, 2022. URLhttps://arxiv.org/
abs/2107.07511.

Angelopoulos, A. N., Bates, S., Candes, E. J., Jordan,`
M. I., and Lei, L. Learn then test: Calibrating pre-
dictive algorithms to achieve risk control, 2022. URL
https://arxiv.org/abs/2110.01052.

Band, N., Rudner, T. G. J., Feng, Q., Filos, A., Nado,
Z., Dusenberry, M. W., Jerfel, G., Tran, D., and Gal,
Y. Benchmarking bayesian deep learning on diabetic
retinopathy detection tasks, 2022. URLhttps://
arxiv.org/abs/2211.12717.

Chen, K., Zhang, Z., Zeng, W., Zhang, R., Zhu, F., and
Zhao, R. Shikra: Unleashing multimodal llm’s referential
dialogue magic, 2023. URLhttps://arxiv.org/
abs/2306.15195.

Cheng, K., Sun, Q., Chu, Y., Xu, F., YanTao, L., Zhang,
J., and Wu, Z. Seeclick: Harnessing gui grounding for
advanced visual gui agents. In Proceedings of the 62nd
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pp. 9313–9332,
2024.

Clopper, C. J. and Pearson, E. S. The use of confidence
or fiducial limits illustrated in the case of the binomial.
Biometrika, 1934.

Company, H. Holo1.5 - open foundation models
for computer use agents, 2025. URL https://

```
huggingface.co/collections/Hcompany/
holo15-68c1a5736e8583a309d23d9b.
```
```
Fan, Y., Zhao, H., Zhang, R., Shen, Y., Wang, X. E.,
and Wu, G. GUI-bee: Align GUI action grounding
to novel environments via autonomous exploration. In
Christodoulopoulos, C., Chakraborty, T., Rose, C., and
Peng, V. (eds.), Proceedings of the 2025 Conference
on Empirical Methods in Natural Language Processing,
Suzhou, China, November 2025. Association for Compu-
tational Linguistics.
```
```
Gal, Y. and Ghahramani, Z. Dropout as a bayesian approxi-
mation: Representing model uncertainty in deep learning.
In Proceedings of The 33rd International Conference on
Machine Learning.
```
```
Gawlikowski, J., Tassi, C. R. N., Ali, M., Lee, J., Humt,
M., Feng, J., Kruspe, A., Triebel, R., Jung, P., Roscher,
R., Shahzad, M., Yang, W., Bamler, R., and Zhu, X. X.
A survey of uncertainty in deep neural networks, 2022.
URL https://arxiv.org/abs/2107.03342.
```
```
Geifman, Y. and El-Yaniv, R. Selective classification for
deep neural networks. Advances in neural information
processing systems, 30, 2017.
```
```
Hendrycks, D. and Gimpel, K. A baseline for detecting
misclassified and out-of-distribution examples in neu-
ral networks. In International Conference on Learning
Representations, 2017. URLhttps://openreview.
net/forum?id=Hkg4TI9xl.
```
```
Hong, W., Wang, W., Lv, Q., Xu, J., Yu, W., Ji, J., Wang,
Y., Wang, Z., Dong, Y., Ding, M., and Tang, J. Cogagent:
A visual language model for gui agents. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), pp. 14281–14290, June
2024.
```
```
Hou, B., Zhang, Y., Andreas, J., and Chang, S. A probabilis-
tic framework for llm hallucination detection via belief
tree propagation, 2025. URLhttps://arxiv.org/
abs/2406.06950.
```
```
Hu, M., Zhang, Z., Zhao, S., Huang, M., and Wu, B.
Uncertainty in natural language processing: Sources,
quantification, and applications, 2023. URLhttps:
//arxiv.org/abs/2306.04459.
```
```
Jung, J., Brahman, F., and Choi, Y. Trust or escalate: LLM
judges with provable guarantees for human agreement.
In The Thirteenth International Conference on Learning
Representations, 2025. URLhttps://openreview.
net/forum?id=UHPnqSTBPO.
```

Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain,
D., Perez, E., Schiefer, N., Hatfield-Dodds, Z., DasSarma,
N., Tran-Johnson, E., Johnston, S., El-Showk, S., Jones,
A., Elhage, N., Hume, T., Chen, A., Bai, Y., Bowman,
S., Fort, S., Ganguli, D., Hernandez, D., Jacobson, J.,
Kernion, J., Kravec, S., Lovitt, L., Ndousse, K., Olsson,
C., Ringer, S., Amodei, D., Brown, T., Clark, J., Joseph,
N., Mann, B., McCandlish, S., Olah, C., and Kaplan, J.
Language models (mostly) know what they know, 2022.
URL https://arxiv.org/abs/2207.05221.

Kuhn, L., Gal, Y., and Farquhar, S. Semantic uncer-
tainty: Linguistic invariances for uncertainty estima-
tion in natural language generation. In The Eleventh
International Conference on Learning Representations,
2023a. URLhttps://openreview.net/forum?
id=VD-AYtP0dve.

Kuhn, L., Gal, Y., and Farquhar, S. Semantic uncer-
tainty: Linguistic invariances for uncertainty estimation
in natural language generation, 2023b. URLhttps:
//arxiv.org/abs/2302.09664.

Li, K., Meng, Z., Lin, H., Luo, Z., Tian, Y., Ma, J., Huang,
Z., and Chua, T.-S. Screenspot-pro: Gui grounding for
professional high-resolution computer use, 2025. URL
https://arxiv.org/abs/2504.07981.

Lin, Z., Trivedi, S., and Sun, J. Generating with confidence:
Uncertainty quantification for black-box large language
models. arXiv preprint arXiv:2305.19187, 2023.

Lin, Z., Trivedi, S., and Sun, J. Generating with confi-
dence: Uncertainty quantification for black-box large
language models, 2024. URLhttps://arxiv.org/
abs/2305.19187.

Liu, X., Chen, T., Da, L., Chen, C., Lin, Z., and Wei, H.
Uncertainty quantification and confidence calibration in
large language models: A survey, 2025. URLhttps:
//arxiv.org/abs/2503.15850.

Nguyen, D., Chen, J., Wang, Y., Wu, G., Park, N., Hu, Z.,
Lyu, H., Wu, J., Aponte, R., Xia, Y., Li, X., Shi, J., Chen,
H., Lai, V. D., Xie, Z., Kim, S., Zhang, R., Yu, T., Tanjim,
M., Ahmed, N. K., Mathur, P., Yoon, S., Yao, L., Kveton,
B., Kil, J., Nguyen, T. H., Bui, T., Zhou, T., Rossi, R. A.,
and Dernoncourt, F. Gui agents: A survey, 2025. URL
https://arxiv.org/abs/2412.13501.

Pouget, A., Drugowitsch, J., and Kepecs, A. Confidence
and certainty: distinct probabilistic quantities for different
goals. Nature neuroscience, 19(3):366–374, 2016.

Qin, Y., Ye, Y., Fang, J., Wang, H., Liang, S., Tian, S.,
Zhang, J., Li, J., Li, Y., Huang, S., Zhong, W., Li, K.,
Yang, J., Miao, Y., Lin, W., Liu, L., Jiang, X., Ma, Q.,

```
Li, J., Xiao, X., Cai, K., Li, C., Zheng, Y., Jin, C., Li, C.,
Zhou, X., Wang, M., Chen, H., Li, Z., Yang, H., Liu, H.,
Lin, F., Peng, T., Liu, X., and Shi, G. Ui-tars: Pioneering
automated gui interaction with native agents, 2025a. URL
https://arxiv.org/abs/2501.12326.
```
```
Qin, Y., Ye, Y., Fang, J., Wang, H., Liang, S., Tian, S.,
Zhang, J., Li, J., Li, Y., Huang, S., et al. Ui-tars: Pioneer-
ing automated gui interaction with native agents. arXiv
preprint arXiv:2501.12326, 2025b.
```
```
Team, G., Anil, R., Borgeaud, S., Alayrac, J.-B., Yu, J., Sori-
cut, R., Schalkwyk, J., Dai, A. M., Hauth, A., Millican,
K., et al. Gemini: a family of highly capable multimodal
models. arXiv preprint arXiv:2312.11805, 2023.
```
```
Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen,
K., Liu, X., Wang, J., Ge, W., Fan, Y., Dang, K., Du,
M., Ren, X., Men, R., Liu, D., Zhou, C., Zhou, J., and
Lin, J. Qwen2-vl: Enhancing vision-language model’s
perception of the world at any resolution, 2024a. URL
https://arxiv.org/abs/2409.12191.
```
```
Wang, Q., Fan, Y., and Wang, X. E. Safer: Risk-constrained
sample-then-filter in large language models, 2025a. URL
https://arxiv.org/abs/2510.10193.
```
```
Wang, Q., Geng, T., Wang, Z., Wang, T., Fu, B., and
Zheng, F. Sample then identify: A general framework
for risk control and assessment in multimodal large lan-
guage models. In The Thirteenth International Confer-
ence on Learning Representations, 2025b. URLhttps:
//openreview.net/forum?id=9WYMDgxDac.
```
```
Wang, Z., Wang, Q., Zhang, Y., Chen, T., Zhu, X., Shi, X.,
and Xu, K. SConU: Selective conformal uncertainty in
large language models. In Che, W., Nabende, J., Shutova,
E., and Pilehvar, M. T. (eds.), Proceedings of the 63rd
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), July.
```
```
Wang, Z., Duan, J., Yuan, C., Chen, Q., Chen, T., Zhang,
Y., Wang, R., Shi, X., and Xu, K. Word-sequence en-
tropy: Towards uncertainty estimation in free-form medi-
cal question answering applications and beyond, 2024b.
URL https://arxiv.org/abs/2402.14259.
```
```
Wang, Z., Duan, J., Wang, Q., Zhu, X., Chen, T., Shi,
X., and Xu, K. Coin: Uncertainty-guarding selective
question answering for foundation models with provable
risk guarantees, 2025c. URLhttps://arxiv.org/
abs/2506.20178.
```
```
Wu, Q., Cheng, K., Yang, R., Zhang, C., Yang, J., Jiang,
H., Mu, J., Peng, B., Qiao, B., Tan, R., Qin, S., Liden,
L., Lin, Q., Zhang, H., Zhang, T., Zhang, J., Zhang, D.,
and Gao, J. Gui-actor: Coordinate-free visual grounding
```

```
for gui agents, 2025a. URLhttps://arxiv.org/
abs/2506.03143.
```
Wu, Q., Cheng, K., Yang, R., Zhang, C., Yang, J., Jiang,
H., Mu, J., Peng, B., Qiao, B., Tan, R., et al. Gui-actor:
Coordinate-free visual grounding for gui agents. arXiv
preprint arXiv:2506.03143, 2025b.

Xu, Z., Song, T., and Lee, Y.-C. Confronting verbalized
uncertainty: Understanding how llm’s verbalized uncer-
tainty influences users in ai-assisted decision-making. Int.
J. Hum.-Comput. Stud., 197(C), March 2025. ISSN 1071-

5819. doi: 10.1016/j.ijhcs.2025.103455. URLhttps:
//doi.org/10.1016/j.ijhcs.2025.103455.

Yang, Y., Li, D., Dai, Y., Yang, Y., Luo, Z., Zhao, Z.,
Hu, Z., Huang, J., Saha, A., Chen, Z., Xu, R., Pan, L.,
Savarese, S., Xiong, C., and Li, J. Gta1: Gui test-time
scaling agent, 2025. URLhttps://arxiv.org/
abs/2507.05791.

Ye, F., Yang, M., Pang, J., Wang, L., Wong, D. F., Yilmaz, E.,
Shi, S., and Tu, Z. Benchmarking LLMs via uncertainty
quantification. In The Thirty-eight Conference on Neu-
ral Information Processing Systems Datasets and Bench-
marks Track, 2024. URLhttps://openreview.
net/forum?id=L0oSfTroNE.

Zhang, S., Fu, P., Zhang, R., Yang, J., Du, A., Xi, X., Wang,
S., Huang, Y., Qin, B., Luo, Z., and Luan, J. Hyperclick:
Advancing reliable GUI grounding via uncertainty cal-
ibration, 2025. URLhttps://openreview.net/
forum?id=pXYwksqDyE.


## Limitation

Our uncertainty estimation relies on the variability in the sampled predictions to characterize spatial ambiguity. For highly
deterministic models with limited sampling diversity, the resulting spatial distributions may be less informative. Despite
these limitations, SAFEGROUND provides a general and principled foundation for uncertainty-aware GUI grounding.

## A. Proofs

In this section, we provide a compelete proof that the upper confidence boundFDRˆ

upper
1 −δ (τ)defined in Eq.(9)satisfies the
statistical guarantee in Eq.(9). RecallFDRˆ

```
upper
1 −δ (τ) = sup{R : Pr(Bin(n,R) ≤ X) ≥ δ}, wheren =
```
### PN

i^1 {ui≤ τ}
is the number of accepted calibration samples, andX =

### PN

```
i^1 {ui≤ τ,A(ˆy
```
(MLG)
i ,B
∗
i) = 0}is the number of accepted
incorrect calibration samples. In general,Bin(n,R)denotes the random variable representing the number of successes inn
Bernoulli trials when the system success probability isR. In our setting, it corresponds to the random variable counting the
number of errors among n samples when the system FDR is R under a given threshold τ.

We define the cumulative distribution function (CDF) of the random variableRˆ(τ) =Bin(nn;R(τ)), corresponding to the error
rate over any n accepted samples when the system FDR is R(τ), as

```
CDF
```
### 

```
r | R(τ)
```
### 

```
= Pr
```
### ˆ

```
R(τ)≤ r | R(τ)
```
### 

### . (13)

By the definition ofFDRˆ

```
upper
1 −δ (τ), we have
```
### CDF

### 

### X

```
n
```
### |FDRˆ

```
upper
1 −δ (τ)
```
### 

```
= δ. (14)
```
If R(τ) >FDRˆ

```
upper
1 −δ (τ), we have CDF
```
### X

```
n| R(τ)
```
### 

```
≤ δ. Then, we have
```
```
Pr
```
### 

```
R(τ)≤FDRˆ
```
```
upper
1 −δ (τ)
```
### 

```
= 1− Pr
```
### 

```
R(τ) >FDRˆ
```
```
upper
1 −δ (τ)
```
### 

```
≥ 1 − Pr
```
### 

### CDF

### 

### X

```
n
```
```
| R(τ)
```
### 

```
≤ δ
```
### . (15)

We further the Inverse Cumulative Distribution Function (ICDF):

```
CDF−^1
```
### 

```
p| R(τ)
```
### 

```
= sup{r : CDF (r | R(τ))≤ p}. (16)
```
If CDF

### X

```
n| R(τ)
```
### 

```
≤ δ, we haveXn≤ CDF−^1
```
### 

```
δ | R(τ)
```
### 

. We then obtain

```
Pr
```
### 

```
R(τ)≤FDRˆ
```
```
upper
1 −δ (τ)
```
### 

```
≥ 1 − Pr
```
### 

### X

```
n
```
### ≤ CDF−^1

### 

```
δ | R(τ)
```
### 

### 

### . (17)

SinceXnis exactly the empirical error rate observed over thenaccepted samples in the calibration set, the probability that it

is less than or equal to CDF−^1

### 

```
δ | R(τ)
```
### 

```
does not exceed δ. Finally, we conclude
```
```
Pr
```
### 

```
R(τ)≤FDRˆ
```
```
upper
1 −δ (τ)
```
### 

```
≥ 1 − δ. (18)
```
In this way, we obtain an upper bound on the system FDR at thresholdτwith at least 1 − δconfidence. At test time, by the
exchangeability condition, we provide marginal guarantees of FDR control.

## B. Details of Experimental Settings

B.1. Dataset

ScreenSpot-Pro ScreenSpot-Pro consists of 1581 UI screenshots paired with natural language instructions that refer to
target UI elements on the screen. Each target is annotated as a spatial region rather than a single point. Compared to earlier
GUI grounding benchmarks, ScreenSpot-Pro features higher visual complexity, denser UI layouts, and more fine-grained
distinctions between neighboring elements, making it particularly suitable for studying uncertainty-aware grounding.


```
B.2. Evaluation Metrics
```
```
We evaluate uncertainty estimation quality and selective prediction performance using four complementary metrics: AUROC,
AUARC, FDR, and power. All metrics are defined with respect to the admission functionA(ˆy,B∗)∈{ 0 , 1 }introduced in
Section 3.1, which indicates whether a grounding prediction is admissible.
```
```
Area Under Receiver Operating Characteristic (AUROC) LetU(ˆy)denote an uncertainty score, where larger values
indicate higher uncertainty. AUROC measures how wellU(ˆ)y separates inadmissible predictions from admissible ones.
Formally, AUROC is the area under the receiver operating characteristic curve obtained by thresholdingU(ˆ)y to predict
whether A(ˆy,B∗) = 0. A higher AUROC indicates stronger discriminative ability of the uncertainty estimate.
```
```
Area Under Accuracy-Rejection Curve (AUARC) AUARC evaluates selective prediction behavior by measuring how
accuracy changes as predictions with high uncertainty are rejected. LetSτ={i : U(ˆiy)≤ τ}denote the set of accepted
samples under threshold τ. The accuracy at τ is defined as
```
```
Acc(τ) =
```
### 1

```
|Sτ|
```
### X

```
i∈Sτ
```
```
A(ˆiy,B∗i).
```
```
In practice,τis chosen to correspond to a target rejection rate, and AUARC is computed as the area under the curve of
Acc(τ) as a function of the rejection rate.
```
```
False Discovery Rate (FDR) Under a given uncertainty threshold τ , the false discovery rate is defined as
```
```
FDR(τ) =
```
### P

```
iI
```
### 

```
U(ˆiy)≤ τ
```
### 

### I

### 

```
A(ˆiy,B∗i) = 0
```
### 

### P

```
iI
```
### 

```
U(ˆiy)≤ τ
```
### .

```
FDR quantifies the proportion of inadmissible predictions among all accepted predictions and serves as the primary risk
metric controlled by SAFEGROUND.
```
```
Power Power measures the proportion of correct predictions retained by selective prediction under a risk constraint and is
defined as:
```
```
Power(τ) =
```
### PN

```
i=1I
```
### 

```
U(ˆiy)≤ τ
```
### 

### I

### 

```
A(ˆiy,Bi∗) = 1
```
### 

### PN

```
i=1I
```
### 

```
A(ˆiy,Bi∗) = 1
```
### .

```
Higher power indicates that more correct predictions are retained while satisfying the specified FDR constraint.
```
B.3. Spatial Region Construction
Given an input image–instruction pair(x,q), we obtain a set ofKsampled grounding predictionsS = {ˆy(i) =
(x(i),y(i))}Ki=1via stochastic decoding. To lift these point-wise samples into a spatial distribution, we discretize the
screen into a fixedH × Wgrid of patches and map each sampled coordinate to its corresponding patch. LetCu,vdenote
the number of samples falling into patch(u,v). We then normalize the resulting count map to obtain a spatial probability
distribution
Pu,v=
Cu,v
P
u′,v′Cu′,v′

### , (19)

```
which serves as an empirical estimate of the model’s predictive density over the output space.
```
```
Region Extraction To identify object-level grounding hypotheses, we first filter low-density patches using an instance-
adaptive threshold. Specifically, letPmax= maxu,vPu,v, and retain only patches satisfyingPu,v> βPmax, whereβis a
fixed ratio (set to 0. 3 in our experiments, following (Wu et al., 2025b)). We then group spatially adjacent retained patches
(using 4-connected neighborhood) into connected components. This yields a set of disjoint regionsR ={Rm}Mm=1, each
corresponding to a plausible grounding target.
```

```
Gemini
```
```
”””
You are a GUI agent that locates UI elements in screenshots.
```
```
CRITICAL RULES:
```
1. You MUST output ONLY valid JSON, nothing else
    2. Do NOT include any explanation, markdown formatting, or natural language
    3. Do NOT wrap the response in code blocks (“‘json)
    4. Coordinates must be in PIXEL values (NOT normalized to 0-1000)
    5. If you cannot find the element, output an empty list: []

```
Your response must be a valid JSON array.
”””
```
```
Figure 8. A system prompt example for Gemini-3-pro in ScreenSpot-Pro dataset.
```
Region Scoring For each region Rm, we compute a region-level score

```
Sm=
```
### 1

```
|Rm|
```
### X

```
(u,v)∈Rm
```
```
Pu,v, (20)
```
i.e., the average probability density within the region. This score reflects the relative support assigned to the region by
the sampled predictions while remaining invariant to region size. The resulting region scores{Sm}Mm=1are subsequently
normalized and used to compute the uncertainty metrics described in Section 3.3.

## C. Threshold Calibration with Finite-Sample Guarantees

This section details the threshold calibration procedure used in SafeGround to obtain finite-sample guarantees on selective
prediction risk, based on Clopper–Pearson confidence bounds, as summarized in Algorithm 1.

## D. Prompt Template

To ensure a fair and reliable evaluation of large vision–language models on GUI grounding, we adopt a strictly constrained
prompt template for Gemini in the ScreenSpot-Pro benchmark, as illustrated in Figure 8, 9.

## E. Case Study

We present qualitative examples to illustrate how the proposed uncertainty score reflects the reliability of GUI grounding
predictions in practice in Figure 10, 11, 12, 13, 14.

## F. Additional Experimental Results

Sensitivity to Sampling Temperature. We further examine the sensitivity of the proposed uncertainty measures to
the sampling temperature used during stochastic decoding. Table 5 and Table 6 report AUROC and AUARC results on
Holo1.5-3B (Company, 2025) under different temperature settings. As the temperature increases,UIEandUCDbecome
more informative, reflected by consistent gains in AUROC. In contrast, margin-based uncertainty exhibits relatively limited
sensitivity to temperature changes. UCOMshows a dependence on the sampling temperature, reflecting its ability to adapt
to changes in the diversity and dispersion of stochastic predictions, while remaining competitive across the evaluated
temperature range.

Sensitivity to Uncertainty Weighting. We examine the sensitivity of the proposed framework to the weighting scheme
used in the combined uncertainty scoreUCOM. Starting from the default setting(wCD,wIE,wTA) = (0. 6 , 0. 2 , 0 .2), we


```
Gemini
```
```
Task: Point to the UI element matching this instruction: {instruction}
```
```
Image size: {W} x {H} pixels.
```
```
Output format (JSON only, no markdown):
[{"point": [y, x], "label": "description"}]
```
```
Where:
```
- point: [y, x] coordinates in PIXELS (NOT normalized to 0-1000)
- y is vertical (0 to {H})
- x is horizontal (0 to {W})

```
If no element found, output: []
```
```
Example: [{"point": [60, 230], "label": "submit button"}]
```
```
Figure 9. A user prompt example for Gemini-3-pro in ScreenSpot-Pro dataset.
```
```
Instruction: “Use Rectangle Tool. ”
```
```
Uncertainty:
"𝑈!"": 0.
"𝑈#$": 0.
"𝑈%&": 0.
"𝑈%'(": 0.
```
```
Hit: False
```
```
Figure 10. An example of GUI grounding task using our uncertainty score.
```
evaluate several alternative weighting configurations that moderately vary the relative contributions of the three uncertainty
components, while keeping the weights normalized.

Specifically, we consider the following weighting configurations for the combined uncertainty score UCOM:

- v1: (wCD,wIE,wTA) = (0. 34 , 0. 33 , 0 .33);
- v2: (wCD,wIE,wTA) = (0. 2 , 0. 2 , 0 .6);
- v3: (wCD,wIE,wTA) = (0. 2 , 0. 6 , 0 .2);
- v4: (wCD,wIE,wTA) = (0. 5 , 0. 25 , 0 .25);


```
Instruction: “Create captions based
on the transcription.”
```
```
Uncertainty:
"𝑈!"": 0.
"𝑈#$": 0.5,
"𝑈%&": 0.1,
"𝑈%'(": 0.27,
```
```
Hit: False
```
```
Figure 11. An example of GUI grounding task using our uncertainty score.
```
```
Instruction : “Click the tool menu."
```
```
Uncertainty:
"𝑈!"": 1.
"𝑈#$": 0.
"𝑈%&": 0.8 9
"𝑈%'(": 0.
```
```
Hit: False
```
```
Figure 12. An example of GUI grounding task using our uncertainty score.
```
- v5: (wCD,wIE,wTA) = (0. 25 , 0. 25 , 0 .5);
- v6: (wCD,wIE,wTA) = (0. 25 , 0. 5 , 0 .25);
- original: (wCD,wIE,wTA) = (0. 6 , 0. 2 , 0 .2).

As shown in Figure 15, 16, 18, 19, 17, 20, across all evaluated models, both AUROC and AUARC exhibit only minor
fluctuations under different weighting schemes. These results indicate that the proposed uncertainty aggregation is robust to
moderate changes in the weighting scheme, supporting the use of a fixed, model-agnostic combination in practice.


```
Instruction : “Search report in vivado."
```
```
Uncertainty:
"𝑈!"": 1.
"𝑈#$": 0.9 7
"𝑈%&": 0.8 6
"𝑈%'(": 0.9 2
```
```
Hit: False
```
```
Figure 13. An example of GUI grounding task using our uncertainty score.
```
```
Instruction : “Expand the range of the
docstrings of the load_datafunction."
```
```
Uncertainty:
"𝑈!"": 0.
"𝑈#$": 0. 5
"𝑈%&": 0. 1
"𝑈%'(": 0. 26
```
```
Hit: True
```
```
Figure 14. An example of GUI grounding task using our uncertainty score.
```
Table 5. AUROC of different uncertainty measures on Holo1.5-3B under varying sampling temperatures.

```
Method Temp=0.3 Temp=0.5 Temp=0.7 Temp=1.
UTA 0.6258 0.6270 0.6297 0.
UIE 0.6621 0.6900 0.7329 0.
UCD 0.6689 0.7078 0.7590 0.
UCOM 0.6819 0.7218 0.7578 0.
```
Table 6. AUARC of different uncertainty measures on Holo1.5-3B under varying sampling temperatures.

```
Method Temp=0.3 Temp=0.5 Temp=0.7 Temp=1.
UTA 0.5373 0.5247 0.5186 0.
UIE 0.5182 0.5165 0.5015 0.
UCD 0.5219 0.5205 0.4977 0.
UCOM 0.5250 0.5308 0.4960 0.
```

Algorithm 1 SafeGround: Clopper–Pearson Threshold Calibration with Sampling-Based Spatial Uncertainty

```
1:Input: GUI grounding modelf; calibration setDcal={(xi,qi,B∗i)}Ni=1; sample countK; patch grid sizeH × W;
region threshold ratio β; admission function A(ˆy,B∗); risk level α; significance level δ; weights (wCD,wIE,wTA)
2: Output: calibrated uncertainty threshold ˆτ
3: for i = 1 to N do
4: (Primary prediction) Obtain ˆ(iyMLG)← f(xi,qi)
5: (Sampling) Draw K stochastic predictionsSi={ˆy
(k)
i }
K
k=1via stochastic decoding
6: (Discretized density map) Initialize count map C ∈ NH×W← 0
7: for k = 1 to K do
8: Map ˆ(iyk)to patch index (u,v) and set Cu,v← Cu,v+ 1
9: end for
10: Normalize to density Pu,v←P Cu,v
u′,v′Cu′,v′
11: (Region extraction) Pmax← maxu,vPu,v; mask Mu,v← I{Pu,v> βPmax}
12: Group 4-connected active patches inMintoMiconnected components via BFS, yielding regionsRi={Ri,m}Mmi=
13: (Region scoring) For each region Ri,m, compute Si,m←|R^1 i,m|
```
### P

```
(u,v)∈Ri,mPu,v
14: Sort scores in descending order: Si,(1)≥···≥ Si,(Mi)
15: Induce categorical distribution ˆi,jp ←
Si,(j)
PMi
ℓ=1Si,(ℓ)
16: (Uncertainty components)
```
```
17: UTA,i←
```
### (

### 1 −

```
Si,(1)−Si,(2)
Si,(1)+ε , Mi≥^2
max(0. 1 , 1 − Si,(1)), Mi= 1
18: UIE,i←−log^1 Mi
```
```
PMi
j=1ˆi,jp log(ˆi,jp + ε)
19: UCD,i← 1 −
```
```
PMi
j=1ˆp
```
```
2
i,j
20: (Combined uncertainty) ui← wCDUCD,i+ wIEUIE,i+ wTAUTA,i
21: (Error indicator) erri← I{A(ˆiy(MLG),Bi∗) = 0}
22: end for
23: Sort uncertainties ascending: u(1)≤···≤ u(N)with aligned err(1),...,err(N)
24: Initialize the selected threshold ˆ ←τ NULL
25: for t = 1 to N do
26: Set candidate threshold τ ← u(t)
27: n←
```
### PN

```
j=1I{u(j)≤ τ}^ (number of accepted samples)
28: X ←
```
### PN

```
j=1I{u(j)≤ τ ∧ err(j)= 1}^ (number of errors among accepted)
29: Compute Clopper–Pearson upper bound: UCB← BetaInv(1− δ; X + 1, n− X)
30: if UCB≤ α then
31: Update ˆ ←τ τ
32: end if
33: end for
34: if ˆ =τ NULL then
35: Return “The target risk level α is unattainable under calibration.”
36: else
37: Return ˆτ
38: end if
```

```
Figure 15. Sensitivity analysis of AUROC and AUARC to uncertainty weighting for GTA1-7B.
```
Figure 16. Sensitivity analysis of AUROC and AUARC to uncertainty weighting for GUI-Actor-2.5VL-7B.

```
Figure 17. Sensitivity analysis of AUROC and AUARC to uncertainty weighting for GUI-Actor-2VL-7B.
```

```
Figure 18. Sensitivity analysis of AUROC and AUARC to uncertainty weighting for Holo1.5-7B.
```
```
Figure 19. Sensitivity analysis of AUROC and AUARC to uncertainty weighting for Holo1.5-3B.
```
Figure 20. Sensitivity analysis of AUROC and AUARC to uncertainty weighting for UI-TARS-1.5-7B.


