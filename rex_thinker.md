# arXiv:2506.04034v1 [cs.CV] 4 Jun 2025

## Rex-Thinker: Grounded Object Referring via

## Chain-of-Thought Reasoning

## Qing Jiang

###### 1 , 2 ∗

## , Xingyu Chen

###### 3 ∗

## , Zhaoyang Zeng

###### 1

## , Junzhi Yu

###### 3

## , Lei Zhang

###### 1 , 2 †

###### 1

## International Digital Economy Academy (IDEA)

(^2) South China University of Technology

###### 3

## Peking University

#### {jiangqing, chenxingyu, leizhang}@idea.edu.cn

## https://rexthinker.github.io/

## Abstract

## Object referring aims to detect all objects in an image that match a given natu-

## ral language description. We argue that a robust object referring model should

## be grounded, meaning its predictions should be both explainable and faithful to

## the visual content. Specifically, it should satisfy two key properties:1) Verifi-

## able, by producing interpretable reasoning that justifies its predictions and clearly

## links them to visual evidence; and2) Trustworthy, by learning to abstain when

## no object in the image satisfies the given expression. However, most methods

## treat referring as a direct bounding box prediction task, offering limited inter-

## pretability and struggling to reject expressions with no matching object. In this

## work, we propose Rex-Thinker, a model that formulates object referring as an ex-

## plicit Chain-of-Thought (CoT) reasoning task. Given a referring expression, we

## first identify all candidate object instances corresponding to the referred object

## category. Rex-Thinker then performs step-by-step reasoning over each candidate

## to assess whether it matches the given expression, before making a final predic-

## tion. To support this paradigm, we construct a large-scale CoT-style referring

## dataset named HumanRef-CoT by prompting GPT-4o on the HumanRef dataset.

## Each reasoning trace follows a structured planning, action, and summarization

## format, enabling the model to learn decomposed, interpretable reasoning over ob-

## ject candidates. We then train Rex-Thinker in two stages: a cold-start supervised

## fine-tuning phase to teach the model how to perform structured reasoning in our

## defined CoT format, followed by GRPO-based reinforcement learning to further

## improve accuracy and generalization. Experiments show that our CoT-based ap-

## proach outperforms standard baselines in both precision and interpretability on

## in-domain evaluation, while also demonstrating improved ability to reject hal-

## lucinated outputs and strong generalization in out-of-domain settings. Code is

## available athttps://github.com/IDEA-Research/Rex-Thinker.

## 1 Introduction

## Object Referring, also known as Referring Expression Comprehension (REC) [45, 69, 38, 23, 73, 35,

## 68, 68, 63, 29], aims to predict the bounding boxes for objects in an image that match a given natural

## language description, which may refer to visual attributes, spatial relations, or interactions. This task

## has broad applications; however, compared to standard open-vocabulary object detection [20, 46, 47,

## 32, 65, 19, 18, 25, 46, 27, 9, 40, 70, 56], REC is significantly more challenging, as it requires both

## fine-grained visual grounding and more complicated language understanding.

```
∗
```
#### Equal contributions, work done during internship or academic visit at IDEA.

```
†
```
#### Corresponding author.

#### Preprint. Under review.


```
Question : Please detect the person sitting on the turtle.
Answer :
<think> OK, the user needs us to find the person sitting on the turtle. I need to break it down into the following steps:
```
- Step 1: Identify all the turtles in the image.
- Step 2: For each turtle found in Step 1, check if there is a person sitting on it.

```
# Step 1: Identify all the turtles in the image.
It is located on the right-hand card in the top-right section of the overall picture. It is a green turtle swimming in a pond
with lily pads.
# Step2: For each turtle found in Step 1, check if there is a person sitting on it. I see five people in this image. Let's
look at their surroundings and put a ⚠ if there is a turtle and ❌ if there is not.
```
- Person 1: A man and a woman sitting on a swing. Not sitting on a turtle. ❌
- Person 2: A man and a woman sitting on a swing. Not sitting on a turtle. ❌
- Person 3: A bearded figure wearing a red hat and red clothes. He is sitting on the green turtle. ⚠
- Person 4: A one - eyed figure in a boat, fishing. Not sitting on a turtle. ❌
- Person 5: A figure near a beehive. Not sitting on a turtle. ❌

```
# Summarize and Re-Check answer
Let’s now recheck our answer and put ✅ for the target fish and ❌ for others.
```
- Person 1,2,4,5: Not sitting on a turtle. ❌
- Person 3: Sitting on the turtle. ✅ **</think>
<answer>** ```json\n[{“bbox_2d”: [1021, 210, 1307, 420], “label”: “the person sitting on the turtle”}]``` **</answer>**

```
Person 5 Person 3
```
```
Person 4
Person 2
Person 1
```
### Planning

### Ac)on

### Summariza)on

### Figure 1: An example of Rex-Thinker for object referring with CoT reasoning of planning (task

### decomposition), action (evaluating each candidate), and summarization (final decision). Each step

### is grounded in a specific hint box (as denoted in the left image), enabling interpretable predictions.

### Benefiting from the strong language comprehension capabilities of large language models (LLMs),

### multimodal large language models (MLLMs) have demonstrated impressive performance on this

### task. There are mainly two paradigms: one treats bounding box coordinates as text tokens and

### predicts them directly [5, 66, 74, 54, 71, 72, 3, 57, 7, 39], while the other adopts a retrieval-based

### strategy [36, 22, 21], where the model is given a set of candidate boxes and predicts the box indices

### that match the expression. Although both approaches have shown promising results, they remain

### fundamentally implicit, lacking interpretable reasoning steps that reveal how the model arrives at its

### final prediction. Furthermore, these models are prone to hallucination [21], often producing outputs

### for objects that do not exist in the image, thereby limiting their reliability in real-world applications.

### We argue that a robust referring system should begrounded, i.e., its predictions must be both ex-

### plainable and tightly linked to visual evidence. This requires two essential properties:1) Verifiable,

### by providing an explicit reasoning process that allows its decisions to be examined and traced to

### specific image regions; and2) Trustworthy, by minimizing hallucinated outputs and learning to

### reject when no object in the image satisfies the given description. To meet these criteria, we draw

### inspiration from how humans naturally approach referring expressions. For example, when asked

### to locate “the person wearing a blue shirt”, humans would typically first identify all people in the

### image, then examine each one to determine whether it matches the described attribute. This step-by-

### step approach reflects a grounded reasoning process, i.e., first localizing relevant object candidates,

### and then carefully verifying each one against the expression.

### Motivated by this observation, we propose Rex-Thinker, an MLLM that performs object referring

### through explicit Chain-of-Thought (CoT) reasoning. Specifically, given an image and a referring

### expression, we first use an open-vocabulary object detector [32] to extract all candidate object boxes

### corresponding to the referred category. These candidate boxes, along with the image and the expres-

### sion, are then passed into the model for step-by-step reasoning. Rex-Thinker follows a structured

### CoT framework consisting of three key stages as shown in Figure 1:1) Planning, where the model

### decomposes the referring expression into subgoals;2) Action, where the model examines each can-

### didate box to determine whether it satisfies its current subgoal;3) Summarization, where it aggre-

### gates the intermediate decisions to produce the final prediction. Following DeepSeek-R1 [14], we

### instruct the model to place its reasoning steps within a<think>...</think>block and to output the

### final prediction inside a<answer>...</answer>block. This structured reasoning process not only

### improves interpretability, but also enables transparent and verifiable predictions, as each reasoning

### step is grounded in a specific candidate region in the image.

### To support this CoT framework, we construct a CoT-style referring dataset named HumanRef-CoT,

### containing 90,824 samples generated by prompting GPT-4o [16] on the HumanRef [21] dataset.

### Each example is annotated with a structured reasoning trace following the planning, action, and

### summarization paradigm, enabling explicit supervision for step-by-step reasoning. We train our

### model in two stages: a cold-start supervised fine-tuning phase to teach the model how to perform

### structured reasoning, followed by reinforcement learning (RL) based on Group Relative Policy


### Optimization (GRPO) [48] to further improve accuracy and generalization. Experiments demon-

### strate that our CoT-based approach consistently outperforms direct coordinate prediction baselines.

### On the in-domain HumanRef benchmark, our model achieves state-of-the-art results with higher

### detection accuracy and significantly fewer hallucinated outputs, especially on rejection cases. In

### out-of-domain evaluations on RefCOCOg [38], the model trained only on HumanRef-CoT shows

### strong zero-shot generalization. Further fine-tuning with GRPO on RefCOCOg yields additional

### performance gains while preserving the model’s ability to perform grounded CoT reasoning across

### arbitrary object categories. To summarize, our contributions are threefold:

- We formulate the grounded object referring task as aplanning–action–summarizationproblem,

### leveraging Chain-of-Thought reasoning to build a verifiable and trustworthy system.

- We introduce HumanRef-CoT, the first dataset for grounded object referring with step-by-step

### reasoning annotations, enabling the supervised training of model interpretability.

- We propose Rex-Thinker, a grounded object referring model trained via cold-start SFT and

### GRPO-based reinforcement learning. Rex-Thinker achieves SOTA performance on the Human-

### Ref benchmark and demonstrates strong generalization on out-of-domain scenes and objects.

## 2 Related Work

### MLLM-based Object Referring Methods.Recent progress in multimodal large language models

### (MLLMs) [43, 3, 57, 7, 2, 24, 11, 1, 53, 26, 28, 31, 62, 76, 4, 50, 15, 4] has led to strong performance

### in referring expression comprehension. Existing approaches typically follow two paradigms. One

### line of work treats bounding box coordinates as textual tokens [6] and directly generates them during

### decoding [5, 66, 54, 71, 74]. The other line formulates the task as retrieval [22, 36, 21], where

### a detector proposes candidate regions and the model selects the best-matching box indices based

### on the input expression. This decouples localization from semantic understanding and simplifies

### learning.

### While both paradigms achieve strong results on standard benchmarks such as RefCOCO/+/g [38,

### 69], they face key limitations: a lack of interpretability and an inability to abstain when no object in

### the image matches the expression [21]. To address this, we introduce a Chain-of-Thought reasoning

### framework that enables step-by-step evaluation over candidate boxes. This improves interpretability,

### reduces hallucinations, and grounds the model’s predictions in the input image.

### Reasoning-based LLMs and MLLMs.Recent work in large language models [17, 14, 51, 41, 58,

### 59, 10, 42] has demonstrated that reasoning ability can be significantly enhanced through Chain-

### of-Thought (CoT) training or reinforcement learning-based post-training. OpenAI o1 [17] model

### demonstrates that inference-time scaling can greatly enhance performance on complex tasks like

### math and coding. DeepSeek-R1 [14] introduces GRPO [48] as a post-training method to improve

### reasoning without requiring costly critic models.

### In the multimodal domain, efforts such as LLaVA-CoT [60] and LlamaV-o1 [52] aim to enhance

### reasoning by constructing CoT-style data or employing multi-step curriculum learning, without re-

### lying on reinforcement learning. More recently, inspired by DeepSeek-R1 [14], a growing number

### of works adopt GRPO-based post-training to endow MLLMs with reasoning capabilities. GRPO

### has been successfully applied to enhance multimodal reasoning across a wide range of domains,

### including mathematical problem solving [64, 44, 75, 12, 55], video understanding [13, 30], and per-

### ception tasks [33, 34, 37, 49, 67] such as object detection, segmentation, and referring expression

### comprehension. Following the DeepSeek-R1 paradigm, we first fine-tune Rex-Thinker on structured

### CoT data to teach the model how to perform grounded object reasoning. GRPO is then applied in a

### second stage to further improve accuracy and generalization.

## 3 Chain-of-Thought Reasoning Referring Data

### High-quality supervision is critical for teaching the model to reason explicitly. To this end, we

### develop a data engine that generates structured referring annotations aligned with our Chain-of-

### Thought formulation. In this section, we introduce the design principles of our CoT reasoning

### structure and present the data construction pipeline that transforms existing REC annotations into

### step-by-step reasoning traces suitable for supervised training.


```
Set-of-marks
Image
```
```
the person between the two adults
```
```
4
```
```
[Person4, Person 3, Person 1, Person 2]
```
```
Answer: Number of Person:
```
```
Position Order:
```
```
Question Referring:
```
```
Meta Information
```
```
1.Planning:
OK, the user needs us to detect the person between the two adults. To accomplish this task, I need to break
it down into the following steps:
```
- Step 1: Identify the two adults in the image.
- Step 2: Determine the person located between the two adults.

```
2.Action:
```
```
3.Summarization:
# Summarize and Re-Check answer
Let’s now recheck our answer and put ✅ for the target people and ❌ for others:
```
- Person 1: Located between the two adults. → ✅
- Person 2: One of the adults, not the person between the adults. → ❌
- Person 3: Located between the two adults. → ✅
- Person 4: One of the adults, not the person between the adults. → ❌

```
# Step 1: Identify the two adults in the image.
I see four people in this image. Let's look at their age - related attributes and put a ⚠ if they are adults
and ❌ if they are not.
```
- Person 1: A young girl with long hair, not an adult. → ❌
- Person 2: A man with sunglasses on his head, appears to be an adult. → ⚠
- Person 3: A young boy with short hair, not an adult. → ❌
- Person 4: A woman with her hair tied back, appears to be an adult. → ⚠

```
So Person 2 and Person 4 are the two adults.
```
```
# Step 2: Determine the person located between the two adults.
The order from left to right is [person 4, person 3, person 1, person 2].
Between Person 4 (the woman on the left - hand side) and Person 2 (the man on the right - hand side),
there are two people: Person 3 and Person 1.
```
```
GPT-4o
```
```
SystemPrompt(Partial)
Given the image and the description below, output a detailed
analysis of how you found the object(s) matching the
description. You should conduct an analysis of what you see
and how each object relates to the provided description and
put a ❌or ✅at the end.... Example:
```
```
Person 1, Person 3
```
```
Example1, ...
```
### Figure 2: Overview of the proposed CoT reasoning referring data engine. We prompt GPT-4o to

### generate a three-step CoT reasoning process, including planning, action, and summarization.

### 3.1 CoT Formulation

### The core idea behind our CoT formulation for REC is to transform the task into a structured,

### grounded reasoning process over a set of candidate objects. Rather than directly predicting the

### referred object, the model evaluates each candidate in sequence, guided by input box hints that

### localize specific regions in the image. We decompose this CoT process into three key stages:

- Planning:The model analyzes the complexity of the referring expression and determines how

### many reasoning steps are needed. For simple expressions, it may plan a single step to directly

### match an attribute such as color or size. For more complex expressions, the model generates a

### multi-step plan, where each step focuses on resolving a specific sub-aspect.

- Action:Based on the reasoning plan, the model checks whether each candidate region, grounded

### via its input box hint, satisfies the current subgoal. This makes the reasoning clear and directly

### tied to specific regions in the image.

- Summarization:Finally, the model reviews the reasoning results across all steps and determines

### which objects best match the overall expression and outputs the final prediction.

### This structured CoT process improves both interpretability and verifiability. Each candidate is eval-

### uated corresponding to the input box hints, allowing every reasoning step to be explicitly grounded

### to a specific region of the image. This makes the model’s decisions transparent and easy to trace.

### Additionally, breaking complex expressions into sub-tasks enables step-by-step reasoning, which

### enhances accuracy and reflects how humans typically process such tasks.

### 3.2 Data Engine Pipeline

### Building on the structured CoT formulation, we develop a data engine that leverages GPT-4o [16]

### to generate high-quality CoT annotations tailored to the referring task.

### 3.2.1 Data Acquisition

### We construct our CoT dataset based on HumanRef [21], a recently proposed dataset specifically

### designed for REC in human-centric scenarios. Unlike prior REC datasets such as RefCOCO/+/g [38,

### 69], HumanRef emphasizes multi-instance referring expressions, where a single expression may

### refer to multiple target persons. It also categorizes expressions into six distinct subsets: attribute,

### position, interaction, reasoning, celebrity recognition, and rejection. Since the HumanRef dataset

### provides all person boxes in an image, it can be directly used in our CoT annotation pipeline.

### 3.2.2 GPT-4o Annotation

### To generate high-quality CoT annotations, we employ in-context prompting with GPT-4o [16] as

### shown in Figure 2. Given an image and the bounding boxes of all persons within it, we apply the Set-

### of-Mark [61] strategy: each individual is labeled with an indexed visual marker, where ground-truth

### targets are marked in green and others in red. This design grounds the answer and guides GPT-4o

### to reason along the correct path. The prompt includes three key components: 1) meta-information

### such as the referring question, the number of people, their left-to-right spatial order, and the correct

### answer; 2) a system prompt specifying the desired planning–action–summarization structure; and


```
Qwen2.5-VL
```
```
Open-Vocabulary
Object Detector
```
```
System
Box Hint Prompt
```
```
“person” Text Tokenizer
```
```
“Find person
+ holding letter H”
```
```
Box Hint {“person”: [[101, 10, 140, 200],[122, 15, 162, 205], [152, 16, 186, 210] ...}
System Prompt “ The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are
enclosed within <think> </think> and <answer> </answer> tags, respectively. ”
```
## SFT ColdStart RL Post Training

- Person 1: He is holding a red letter C. → ❌
- Person 2: She is holding a yellow le>er H→ ✅
- Person 3: He is holding a brown letter U→ ❌
- Person 4: She is holding a letter R→ ❌
- Person 5: He is holding a red letter C. → ❌
- Person 6: She is holding a blue letter H→ ✅

```
<think>
```
```
</think><answer>
```
```
</answer>
```
```
[{"bbox_2d": [101, 10, 140, 200], "label": ”..."},
{"bbox_2d": [122, 15, 162, 205], "label": ”..."}]
```
- Person 1: He is holding a red letter H. → ✅
- Person 2: She is holding a yellow letter C→ ❌
- Person 3: He is holding a brown letter U→ ❌
- Person 4: She is holding a letter R→ ❌
- Person 5: He is holding a red letter C. → ❌
- Person 6: She is holding a blue letter H→ ✅

```
<think>
```
```
</think><answer>
```
```
</answer>
```
```
[{"bbox_2d": [152, 16, 186, 210], "label": ”..."},
{"bbox_2d": [122, 15, 162, 205], "label": ”..."}]
```
```
model
```
```
F1 Reward: 1.
```
```
competition 1 competition 2 competition N- 1 competition N
```
```
...
```
```
F1 Reward: 0.5 ... F1 Reward: 0.0 F1 Reward: 0.
```
```
GRPO Loss
```
```
Update
```
```
<think> ... </think> <answer> ... </answer>
```
```
<think> ... </think> <answer> ... </answer>
```
```
Per Token CE Loss
```
###### GT:

### Figure 3: Overview of the Rex-Thinker architecture and our two-stage training methods

#### <image>. A conversation between User and Assistant. The user asks a question, and the Assistant solves it.

#### The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.

#### The reasoning process and answer are enclosed within<think> </think>and<answer> </answer>tags,

#### respectively, i.e.,<think>reasoning process here</think> <answer>answer here</answer>. Hint: Object

#### and its coordinates in this image: Box Hint. User: Locate Referring. Assistant:

### Table 1: Prompt Template for Rex-Thinker. Box Hint and Referring will be replaced with the input

### candidate boxes and the referring expression, respectively.

### 3) several in-context examples written by humans to illustrate the expected reasoning format. In

### essence, we provide GPT-4o with both the referring expression and its ground-truth answer, and

### prompt it to generate step-by-step reasoning in our CoT format. To ensure annotation quality, we

### retain only examples where GPT-4o’s final prediction matches the ground-truth label.

### We construct a total of 90,824 high-quality CoT annotations based on the HumanRef dataset, which

### we refer to as HumanRef-CoT. This diverse and large-scale dataset serves as the foundation for both

### our initial cold-start SFT and GRPO-based post-training.

## 4 Method

### To leverage the CoT-style referring data, we present Rex-Thinker, a retrieval-based model that per-

### forms object referring through explicit Chain-of-Thought reasoning.

### 4.1 Retrieval-based Object Referring

### To support explicit Chain-of-Thought (CoT) reasoning, we reformulate referring expression com-

### prehension as a retrieval-based task. As shown in Figure 3, rather than directly regressing bounding

### boxes, we first use an open-vocabulary detector [32] to extract a set of candidate object boxes cor-

### responding to the referred object category. These candidate boxes serve asbox hintsto guide both

### the reasoning path and final decision of the model. This retrieval-based formulation brings two

### key advantages. First, during the reasoning phase, the model evaluates each candidate region in

### the order they appear in the input box hints (e.g., “Person 1” corresponds to the first input box).

### This alignment ensures that each step in the CoT trace is explicitly grounded to a specific region in

### the image, making the reasoning process interpretable and visually verifiable. Second, during the

### prediction phase, the model can directly select from the input box hints when producing the final

### output, thereby easing the challenge of precise coordinate regression.

### We build Rex-Thinker on top of Qwen2.5-VL-7B [3], preserving its original architecture and using

### JSON-format bounding box coordinates as the final output. The model input includes the image, the

### box hint, the referring expression, and a system prompt that guides the reasoning process. We adopt

### a similar input prompt format in DeepSeek-R1 [14] as shown in Table 1.

### 4.2 Training

### Following DeepSeek-R1 [14], we adopt a two-stage training strategy consisting of supervised fine-

### tuning for cold start and GRPO-based reinforcement learning for post-training.


### 4.2.1 SFT Cold Start

### We begin by fine-tuning Rex-Thinker on the HumanRef-CoT dataset to instill the ability to perform

### structured reasoning following our defined planning, action, and summarization format. We apply

### cross-entropy loss at the token level to both the reasoning trace and the final answer, providing strong

### supervision across the entire generation process. This stage teaches the model how to reason step-

### by-step in a CoT manner and also how to utilize the provided box hints to guide its final predictions.

### 4.2.2 GRPO Post Training

### While SFT teaches the model to follow our grounded CoT format, its strict token-level supervision

### may constrain the model to explore alternative reasoning traces and generalize beyond the training

### data. To enhance generalization beyond the limitations of supervised learning, we employ GRPO-

### based [48] reinforcement learning for post-training. GRPO optimizes model performance by 1)

### sampling multiple candidate responses for each question and 2) selectively reinforcing responses

### that achieve higher task-level rewards.

### In our setting, given an image and a referring expression(I,x), the model generates a group of

### Gcomplete responseso 1 ,o 2 ,...,oGfrom the current modelπθ. Each response contains a full

### reasoning trace and a final predicted bounding box set. For eachoi, we compute a scalar rewardri

### (detailed in Section 4.2.3), and normalize these rewards to estimate group-relative advantages:

### Ai= (ri−mean(r 1 ,...,rG))/std(r 1 ,...,rG). (1)

### Define the token-level advantage estimatesAˆi,t=Ai, and the importance ratio at each decoding

### step as follows,

### ρi,t=

### πθ(oi,t|(I,x),x,oi,<t)

### πθold(oi,t|(I,x),x,oi,<t)

### , (2)

### whereπθoldis the model before the current update. Then, the GRPO objective is given as follows,

### JGRPO(θ) =

### 1

### G

### XG

###### i=

### 1

### |oi|

###### |oi|

### X

###### t=

### h

### min

### 

### ρi,tAˆi,t,clip(ρi,t, 1 −ε,1 +ε)Aˆi,t

### 

### −βDKL[πθ∥πref]

### i

### (3)

### DKL[πθ∥πref] =

### πθ(oi,t|(I,x),x,oi,<t)

### πref(oi,t|(I,x),x,oi,<t)

### −log

### πθ(oi,t|(I,x),x,oi,<t)

### πref(oi,t|(I,x),x,oi,<t)

### − 1 , (4)

### whereεis a hyperparameter controlling the clipping range,πrefis the model fixed after SFT stage,

### andβis the KL penalty coefficient.

### We argue that this formulation is suited to policy exploration in our reasoning-driven task. Given

### that the model is already capable of producing structured reasoning traces after SFT, GRPO allows it

### to freely explore different reasoning paths. In each iteration, the model generates diverse reasoning

### strategies that may lead to different predicted object sets. The reward function then guides the model

### to reinforce reasoning paths that yield accurate predictions.

### 4.2.3 Reward Modeling

### Accuracy Reward.We use the F1 score to jointly evaluate the precision and recall of the model’s

### predictions. Given a set of predicted boxesBˆand the ground-truth setB

###### ∗

### , since box hints are

### provided as input, we define a match only when a predicted box exactly overlaps with a ground-

### truth box (i.e., IoU = 1), which encourages the model to select final outputs directly from the box

### hints. LetM=Bˆ∩B∗denote the set of matched box pairs under this criterion. We compute

### precision, recall, and the F1 reward as:

### Precision=

### |M|

### |Bˆ|

### , Recall=

### |M|

### |B∗|

### , r

###### F

### =

### 2 ·Precision·Recall

### Precision+Recall

### . (5)

### Format Reward.To encourage interpretable and well-structured output, we define a format reward

### r

###### fmt

### that equals 1 if the output follows the required structure: the reasoning must be enclosed in

### <think>...</think>and the final result in<answer>...</answer>, and 0 otherwise.

### The total reward is a weighted combination of the accuracy and format rewards, i.e.,ri=λ·r

###### F

### i +

### (1−λ)·r

###### fmt

### i , whereλ= 0.^9 to emphasize correct detection while still enforcing output structure.


```
Method Attribute Position Interaction Reasoning Celebrity Average Rejection
R P DF1 R P DF1 R P DF1 R P DF1 R P DF1 R P DF1 Score
DINOX [46] 59.5 28.8 20.9 78.8 28.1 17.6 67.3 28.5 18.9 76.2 32.1 22.2 94.1 48.0 37.0 75.2 33.1 23.3 36.
InternVL-2.5-8B [8] 23.5 39.0 27.1 23.0 28.0 24.3 27.8 40.1 31.3 17.5 22.8 18.9 57.4 59.3 58.0 29.8 37.8 31.9 54.
Ferret-7B [66] 27.9 44.4 30.4 30.2 36.2 29.8 30.8 41.8 31.2 19.7 33.7 22.8 63.2 60.0 57.5 34.4 43.2 34.3 2.
Groma-7B [36] 67.5 47.8 38.6 63.2 43.1 37.2 66.6 48.1 40.6 59.1 41.4 34.8 73.2 63.3 59.1 65.9 48.7 42.1 0.
ChatRex-7B [22] 44.3 78.0 51.8 48.0 66.7 52.5 49.6 74.8 56.5 36.6 65.1 42.8 73.7 76.5 74.2 50.4 72.2 55.6 0.
Qwen2.5-VL-7B [3] 49.1 71.3 54.4 50.2 61.7 52.8 48.2 66.3 53.2 34.6 61.2 40.3 80.3 81.9 80.1 52.5 68.5 56.2 7.
DeepSeek-VL2-small [57] 52.3 78.0 57.7 56.4 66.1 58.1 55.4 75.7 60.7 46.6 61.7 50.1 85.9 74.3 70.7 59.3 71.2 59.5 3.
Molmo-7B-D [11] 82.7 86.4 76.3 78.0 80.6 72.4 69.9 77.7 66.1 72.1 80.4 65.5 85.9 87.5 82.9 77.7 82.5 72.6 68.
RexSeek-7B [21] 87.2 86.8 81.5 86.1 86.3 83.8 84.8 84.6 80.7 87.8 84.7 81.5 83.4 86.5 84.2 85.9 85.8 82.3 54.
Rex-Thinker-Plain 83.0 88.7 81.4 82.5 83.9 81.3 80.1 85.6 80.2 80.5 82.2 77.3 86.7 88.7 86.8 82.6 85.8 81.4 53.
Rex-Thinker-CoT 86.6 87.7 82.7 86.5 87.0 84.3 79.6 81.7 77.2 85.7 83.8 80.3 87.6 89.5 87.2 85.2 85.9 82.3 67.
Rex-Thinker-GRPO 88.5 88.7 84.1 87.2 87.1 84.6 81.5 83.5 79.1 87.7 85.4 82.3 88.0 89.3 87.2 86.6 86.8 83.5 68.
```
### Table 2: In-domain evaluation results on the HumanRef benchmark. R, P, and DF1 represent Recall,

### Precision, and DensityF1. Theblodand underlinefonts indicate the best and second numbers.

## 5 Experiments

### In this section, we evaluate the effectiveness of our CoT-based reasoning approach for object refer-

### ring. We first introduce the experimental setup, then present in-domain results on the HumanRef

### benchmark, followed by out-of-domain evaluation on the RefCOCOg benchmark. Lastly, we con-

### duct ablation studies to analyze key design choices.

### 5.1 Experimental Setup

### Model Setting.We use Qwen2.5-VL-7B-Instruct as our base model. Qwen2.5-VL outputs absolute

### bounding box coordinates rather than quantized tokens, which provides better localization accuracy

### for detection tasks. We adopt this native decoding format for final bounding box predictions.

### SFT Training.We fine-tune the model on the full HumanRef-CoT dataset using supervised learn-

### ing. We use a learning rate of 2e-5, weight decay of 0.01, and cosine decay scheduling. The max-

### imum generation length is set to 2048 tokens. During SFT, the vision encoder and MLP projector

### are frozen, and we update only the LLM parameters. For each training instance, we use all person

### bounding boxes in the image as box hints.

### GRPO Training.After SFT, we apply GRPO for reward-driven post-training. We continue training

### on HumanRef-CoT, but randomly shuffle the box hint order in each training data to create novel input

### configurations. This leads the model to explore different reasoning paths than those seen during SFT.

### During this phase, we train only the LLM. We use a learning rate of 1e-6, 8 rollout samples per input,

### a batch size of 8, and gradient accumulation steps of 2. The KL penalty coefficientβis set to 0.04,

### the sampling temperature to 1.0, and the output length remains 2048 tokens.

### Evaluation Protocol.For in-domain evaluation, we evaluate our model on the HumanRef bench-

### mark, which consists of six subsets: attribute, position, interaction, reasoning, celebrity recognition,

### and rejection. Following [21], we report Recall (R), Precision (P), and DensityF1 (DF1) scores av-

### eraged over IoU thresholds from 0.5 to 0.95. For the rejection subset, we report the rejection score,

### defined as the proportion of 1,000 images where the model correctly outputs no bounding box when

### the object described by the referring expression is not present in the image. For out-of-domain eval-

### uation, we evaluate our model on the RefCOCOg dataset and report accuracy at an IoU threshold of

### 0.5. We compare three variants: 1) Rex-Thinker-Plain, which is trained on HumanRef-CoT using

### SFT only on the final detection outputs, without reasoning supervision; 2) Rex-Thinker-CoT, which

### is trained with SFT on both the reasoning process and the final answer; and 3) Rex-Thinker-GRPO,

### which is initialized from Rex-Thinker-CoT and further optimized with GRPO training.

### 5.2 In-domain Evaluation Results

### We begin by evaluating in-domain performance on the HumanRef benchmark to assess referring

### accuracy within the person domain. As shown in Table 2, Rex-Thinker-CoT, trained with struc-

### tured CoT supervision, consistently outperforms Rex-Thinker-Plain across most evaluation subsets.

### Specifically, it achieves average improvements of +2.6 Recall, +0.1 Precision, and +0.9 DensityF1,

### confirming that step-by-step reasoning leads to more accurate and well-grounded predictions. Most

### notably, the CoT-trained model shows a remarkable 13.8 point improvement in terms of Rejection

### Score on the rejection subset, indicating substantially reduced hallucination rates and enhanced abil-

### ity to appropriately abstain from predictions when no valid target exists, which is a critical capability

### for real-world applications requiring high reliability.

### Additional performance gains are realized through GRPO-based reinforcement learning. Rex-

### Thinker-GRPO demonstrates consistent improvements over Rex-Thinker-CoT, achieving gains of


###### Question : Please detect the fish of manta ray.

###### Answer :

###### <think> OK, the user needs us to detect the manta ray. In order to accomplish this task, I need to break it down into

###### the following steps:

- Step 1: Identify each fish-like creature based on their appearance and determine if they are a manta ray. Let's start

###### our analysis:

###### # Step 1: Identify each fish-like creature based on their appearance and determine if it is a manta ray.

###### I see four creatures in this image. Let’s look at them one by one and put a ⚠ if it is a manta ray and ❌ if it is not.

- Fish 1: It is an orange and white striped fish, which is a clown fish. ❌
- Fish 2: It is a brown fish with spiky features, which is a puffer fish. ❌
- Fish 3: It has a large, wing-like structure on its back and a flat body shape, which are characteristic features of a

###### manta fay. ⚠

- Fish 4: It is a blue-colored fish with a long tail, which is a humpback whale (not a fish). ❌

###### # Summarize and Re-Check answer

###### Let’s now recheck our answer and put ✅ for the target fish and ❌ for others.

- Fish 1: It is a clown fish, not a manta ray. ❌
- Fish 2: It is a puffer fish, not a manta ray. ❌
- Fish 3: It is a manta ray. ✅
- Fish 4: It is a humpback whale, not a fish. ❌ **</think>**

###### <answer> ```json\n[{“bbox_2d”: [83, 894, 501, 1240], “label”: “fish of manta ray”}]``` </answer>

```
Fish 1
```
```
Fish 2
```
```
Fish 3
```
```
Fish 4
```
### Figure 4: The out-of-domain result. We use Rex-Thinker-GPRO trained on HumanRef-CoT to infer

### an unseen category (i.e., fish), resulting in a strong generalization. Boxes in the image denote hints.

###### Model

###### RefCOCOg

###### val test

###### RexSeek-7B [21] 84.0 84.

###### Grounding DINO [32] 86.1 87.

###### QwenVL-2.5-7B [3] 87.2 87.

###### ChatRex-7B [22] 89.8 90.

###### Rex-Thinker-CoT 81.2 80.

###### Rex-Thinker-GRPO 83.2 83.

###### Rex-Thinker-GRPO∗ 89.2 88.

### Table 3: Out-of-domain eval-

### uation results on RefCOCOg.

###### ∗

### Fine-tuned on RefCOCOg

### using GRPO.

```
Question: Please detect the person holding a small ball
```
```
Answer:
<think>
There is one person holding a small ball in the image.
This person is standing on the left side of the group.
</think>
<answer>
[100, 174, 369, 821],[290, 51, 538, 775],[625, 97, 839,
761]
</answer>
```
### Figure 5: Predictions from a model that was trained with GRPO

### only, without CoT-based supervised fine-tuning as cold-start ini-

### tialization. Boxes in the image denote answers.

### +1.4 Recall, +0.9 Precision, and +1.2 DensityF1. These results demonstrate the effectiveness of

### reward-based optimization in improving both the model’s reasoning process and prediction accu-

### racy. While supervised CoT training provides strong guidance and teaches the model how to reason

### step by step, it may constrain the model to follow fixed patterns and limit its ability to explore

### more optimal reasoning strategies. In contrast, GRPO enables dynamic exploration of alternative

### reasoning strategies that better optimize for task-level objectives, leading to improved performance.

### Among the six subsets in HumanRef benchmark, Rex-Thinker-GRPO achieves leading metrics in

### most categories. Notably, in theReasoningsubsets, it outperforms Rex-Thinker-Plain by a +5 Den-

### sityF1, attributed to its CoT capabilities for reasoning tasks. The only subset with relatively weaker

### performance isInteractionsubset, please refer to Section 6 for analysis.

### 5.3 Out-of-domain Evaluation Results

### To evaluate the generalization of Rex-Thinker to unseen object categories, we conduct experiments

### on the out-of-domain RefCOCOg dataset. We first adopt a zero-shot setting: the model is trained

### only on HumanRef-CoT and directly evaluated on RefCOCOg. Given a referring expression and

### its original COCO category label, we use Grounding DINO [32] to detect all instances of the target

### object category and use the detected results as box hints to Rex-Thinker.

### As shown in Table 3, the CoT-trained model already performs competitively without any task-

### specific tuning. Further gains are achieved by applying GRPO for post training, demonstrating that

### reward-driven training enhances the model’s ability to generalize beyond the training domain. Inter-

### estingly, we find that Rex-Thinker maintains its structured CoT behavior even on novel categories.

### As illustrated in Figure 4, Rex-Thinker-GRPO successfully generalizes to detect the fish of manta

### ray (with “fish” bounding boxes as hints) while adhering to itsplanning-action-summarizationrea-

### soning paradigm. Notably, the model demonstrates self-correction ability: when provided with an


```
With
Box Hint
```
```
Attribute Position Interaction Reasoning Celebrity Average Rejection
R P DF1 R P DF1 R P DF1 R P DF1 R P DF1 R P DF1 Score
No 66.4 74.3 67.2 69.3 71.9 69.5 65.2 72.1 66.4 63.6 67.5 62.2 82.4 84.6 82.7 69.4 74.1 69.6 71.
Yes 83.0 88.7 81.4 82.5 83.9 81.3 80.1 85.6 80.2 80.5 82.2 77.3 86.7 88.7 86.8 82.6 85.8 81.4 53.
```
### Table 4: Ablation study on the retrieval-based design of our model. We compare performance with

### and without box hints to assess their impact on referring accuracy.

```
With
Cold Start
```
```
Attribute Position Interaction Reasoning Celebrity Average Rejection
R P DF1 R P DF1 R P DF1 R P DF1 R P DF1 R P DF1 Score
No 81.4 85.8 78.1 80.2 80.2 77.5 79.6 82.6 78.0 77.6 75.0 70.6 87.3 86.5 84.8 81.2 82.0 77.8 66.
Yes 88.5 88.7 84.1 87.2 87.1 84.6 81.5 83.5 79.1 87.7 85.4 82.3 88.0 89.3 87.2 86.6 86.8 83.5 68.
```
### Table 5: Ablation on the impact of CoT-based cold start on final performance after GRPO training.

### incorrect hint label (e.g., a whale was incorrectly labeled as a "fish" in hint boxes), Rex-Thinker

### rectifies the error through logical reasoning and explicitly rejects the misclassification.

### To further explore the upper bound of the model, we fine-tune Rex-Thinker-CoT using GRPO di-

### rectly on RefCOCOg. This leads to additional performance improvements, achieving results com-

### parable to state-of-the-art referring models. The experiment results highlight the adaptability of our

### reasoning paradigm across domains and the effectiveness of reward-based optimization in extending

### CoT reasoning to unseen categories.

### 5.4 Ablations

### Effect of Retrieval-based Referring.Our approach adopts a retrieval-based formulation of object

### referring by providing the model with candidate object boxes as box hints. This design serves two

### key purposes: first, it allows the model to reason over each candidate region individually, aligning

### each step of the reasoning process with a specific image region and thereby ensuring grounded,

### interpretable outputs; second, it enables the model to reference these box hints when producing the

### final prediction, reducing the difficulty of direct coordinate regression.

### To evaluate the impact of this retrieval-based design on referring accuracy, we conduct an ablation

### study by fine-tuning Qwen2.5-VL-7B on HumanRef-CoT with and without box hints. In this exper-

### iment, we do not include CoT supervision, as CoT reasoning inherently depends on the presence of

### box hints. As shown in Table 6, incorporating box hints as input leads to substantial performance

### improvements across all major metrics, with average increases of 13.2, 11.7, and 10.8 points in

### Recall, Precision, and Density F1, respectively. While the model without box hints shows higher

### performance on the rejection subset, we attribute this phenomenon primarily to its over-rejection

### behavior. By analyzing the full test set, we observe that the no-hint model incorrectly abstains from

### prediction on 189 samples across the five non-rejection subsets, compared to only 134 for the box-

### hint variant. These results indicate that box hints facilitate more accurate predictions by reducing

### the difficulty of direct coordinate regression.

### Impact of CoT-based Cold Start on GRPO.In Rex-Thinker, we adopt a two-stage training strat-

### egy where the model is first supervised using CoT-annotated data, followed by GRPO-based rein-

### forcement learning. To assess the importance of this CoT-based initialization, we compare GRPO

### training with and without the cold-start SFT stage.

### As shown in Table 5, the model with CoT-based SFT achieves higher final performance than the

### direct GRPO model, indicating that the initial exposure to structured reasoning patterns provides

### a more effective starting point for reward-driven learning. Furthermore, as illustrated in Fig-

### ure 5, models trained without CoT supervision tend to generate unstructured or incoherent rea-

### soning traces, lacking the verifiable and trustworthy qualities we aim to promote. In contrast, CoT-

### pretrained models produce well-formed thinking steps aligned with our planning, action, and sum-

### marization framework.

## 6 Conclusion

### We have presented Rex-Thinker, a novel framework that has reformulated the object referring prob-

### lem as an explicit Chain-of-Thought reasoning process to achieve grounded and interpretable predic-

### tions. Unlike conventional approaches that have treated referring as direct bounding box prediction,

### our model has first detected candidate objects and then performed step-by-step verification against

### the referring expression through structured planning-action-summarization reasoning. To support

### this paradigm, we have constructed HumanRef-CoT, a large-scale dataset with reasoning traces that

### have enabled learning decomposed and interpretable reasoning patterns. Through a two-stage train-


### ing approach combining SFT and GRPO-based RL, Rex-Thinker has demonstrated superior perfor-

### mance over prior works in both referring accuracy and rejection.

### LimitationAs shown in Table 2, our model has exhibited relatively weaker performance in the in-

### teraction subset. This limitation has arisen because the CoT reasoning process must simultaneously

### model relationships and interactions among multiple objects. Errors in this complex inference chain

### have propagated, leading to misleading final responses. Please refer to the Appendix for further

### limitation analysis.

## References

#### [1] Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Devendra Chaplot, Jessica Chudnovsky, Saurabh

#### Garg, Theophile Gervet, Soham Ghosh, Amélie Héliou, Paul Jacob, et al. Pixtral 12b.arXiv preprint

#### arXiv:2410.07073, 2024.

#### [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc,

#### Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for

#### few-shot learning.NeurIPS, 35:23716–23736, 2022.

#### [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie

#### Wang, Jun Tang, et al. Qwen2.5-VL technical report.arXiv preprint arXiv:2502.13923, 2025.

#### [4] Guo Chen, Zhiqi Li, Shihao Wang, Jindong Jiang, Yicheng Liu, Lidong Lu, De-An Huang, Wonmin

#### Byeon, Matthieu Le, Tuomas Rintamaki, et al. Eagle 2.5: Boosting long-context post-training for frontier

#### vision-language models.arXiv preprint arXiv:2504.15271, 2025.

#### [5] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing

#### multimodal llm’s referential dialogue magic.arXiv preprint arXiv:2306.15195, 2023.

#### [6] Ting Chen, Saurabh Saxena, Lala Li, David J Fleet, and Geoffrey Hinton. Pix2seq: A language modeling

#### framework for object detection.arXiv preprint arXiv:2109.10852, 2021.

#### [7] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye,

#### Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models

#### with model, data, and test-time scaling.arXiv preprint arXiv:2412.05271, 2024.

#### [8] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye,

#### Hao Tian, Zhaoyang Liu, Lixin Gu, Xuehui Wang, Qingyun Li, Yimin Ren, Zixuan Chen, Jiapeng Luo,

#### Jiahao Wang, Tan Jiang, Bo Wang, Conghui He, Botian Shi, Xingcheng Zhang, Han Lv, Yi Wang, Wenqi

#### Shao, Pei Chu, Zhongying Tu, Tong He, Zhiyong Wu, Huipeng Deng, Jiaye Ge, Kai Chen, Kaipeng

#### Zhang, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and

#### Wenhai Wang. Expanding performance boundaries of open-source multimodal models with model, data,

#### and test-time scaling, 2025.

#### [9] Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, and Ying Shan. Yolo-world: Real-

#### time open-vocabulary object detection. InProceedings of the IEEE/CVF Conference on Computer Vision

#### and Pattern Recognition, pages 16901–16911, 2024.

#### [10] Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V Le,

#### Sergey Levine, and Yi Ma. Sft memorizes, rl generalizes: A comparative study of foundation model

#### post-training.arXiv preprint arXiv:2501.17161, 2025.

#### [11] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza

#### Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open

#### data for state-of-the-art multimodal models.arXiv preprint arXiv:2409.17146, 2024.

#### [12] Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. Openvlthinker: An

#### early exploration to complex vision-language reasoning via iterative self-improvement. arXiv preprint

#### arXiv:2503.17352, 2025.

#### [13] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Benyou Wang, and

#### Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms.arXiv preprint arXiv:2503.21776, 2025.

#### [14] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong

#### Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement

#### learning.arXiv preprint arXiv:2501.12948, 2025.


#### [15] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu

#### Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report.arXiv preprint arXiv:2505.07062, 2025.

#### [16] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow,

#### Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card.arXiv preprint arXiv:2410.21276,

#### 2024.

#### [17] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Hel-

#### yar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint

#### arXiv:2412.16720, 2024.

#### [18] Qing Jiang, Feng Li, Tianhe Ren, Shilong Liu, Zhaoyang Zeng, Kent Yu, and Lei Zhang. T-rex: Counting

#### by visual prompting.arXiv preprint arXiv:2311.13596, 2023.

#### [19] Qing Jiang, Feng Li, Zhaoyang Zeng, Tianhe Ren, Shilong Liu, and Lei Zhang. T-rex2: Towards generic

#### object detection via text-visual prompt synergy. InEuropean Conference on Computer Vision, pages

#### 38–57. Springer, 2024.

#### [20] Qing Jiang, Feng Li, Zhaoyang Zeng, Tianhe Ren, Shilong Liu, and Lei Zhang. T-rex2: Towards generic

#### object detection via text-visual prompt synergy. InEuropean Conference on Computer Vision, pages

#### 38–57. Springer, 2025.

#### [21] Qing Jiang, Lin Wu, Zhaoyang Zeng, Tianhe Ren, Yuda Xiong, Yihao Chen, Qin Liu, and Lei Zhang.

#### Referring to any person, 2025.

#### [22] Qing Jiang, Yuqin Yang, Yuda Xiong, Yihao Chen, Zhaoyang Zeng, Tianhe Ren, Lei Zhang, et al. Chatrex:

#### Taming multimodal llm for joint perception and understanding.arXiv preprint arXiv:2411.18363, 2024.

#### [23] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects

#### in photographs of natural scenes. InProceedings of the 2014 conference on empirical methods in natural

#### language processing (EMNLP), pages 787–798, 2014.

#### [24] Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang, Zhiqi Shen, Bowen Qu, Xinyao Niu, Guoyin Wang,

#### Bei Chen, and Junnan Li. Aria: An open multimodal native mixture-of-experts model.arXiv preprint

#### arXiv:2410.05993, 2024.

#### [25] Feng Li, Qing Jiang, Hao Zhang, Tianhe Ren, Shilong Liu, Xueyan Zou, Huaizhe Xu, Hongyang Li, Jian-

#### wei Yang, Chunyuan Li, et al. Visual in-context prompting. InProceedings of the IEEE/CVF Conference

#### on Computer Vision and Pattern Recognition, pages 12861–12871, 2024.

#### [26] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li.

#### Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models.arXiv preprint

#### arXiv:2407.07895, 2024.

#### [27] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan

#### Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded language-image pre-training. InPro-

#### ceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10965–10975,

#### 2022.

#### [28] Zhiqi Li, Guo Chen, Shilong Liu, Shihao Wang, Vibashan VS, Yishen Ji, Shiyi Lan, Hao Zhang, Yilin

#### Zhao, Subhashree Radhakrishnan, et al. Eagle 2: Building post-training data strategies from scratch for

#### frontier vision-language models.arXiv preprint arXiv:2501.14818, 2025.

#### [29] Yue Liao, Si Liu, Guanbin Li, Fei Wang, Yanjie Chen, Chen Qian, and Bo Li. A real-time cross-modality

#### correlation filtering method for referring expression comprehension. InProceedings of the IEEE/CVF

#### Conference on Computer Vision and Pattern Recognition, pages 10880–10889, 2020.

#### [30] Zhenyi Liao, Qingsong Xie, Yanhao Zhang, Zijian Kong, Haonan Lu, Zhenyu Yang, and Zhijie Deng.

#### Improved visual-spatial reasoning via r1-zero-like training.arXiv preprint arXiv:2504.00883, 2025.

#### [31] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. InNeurIPS, 2023.

#### [32] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li,

#### Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set

#### object detection. InEuropean Conference on Computer Vision, pages 38–55. Springer, 2024.

#### [33] Yuqi Liu, Bohao Peng, Zhisheng Zhong, Zihao Yue, Fanbin Lu, Bei Yu, and Jiaya Jia. Seg-zero:

#### Reasoning-chain guided segmentation via cognitive reinforcement. arXiv preprint arXiv:2503.06520,

#### 2025.


#### [34] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi

#### Wang. Visual-rft: Visual reinforcement fine-tuning.arXiv preprint arXiv:2503.01785, 2025.

#### [35] Gen Luo, Yiyi Zhou, Xiaoshuai Sun, Liujuan Cao, Chenglin Wu, Cheng Deng, and Rongrong Ji. Multi-

#### task collaborative network for joint referring expression comprehension and segmentation. InProceedings

#### of the IEEE/CVF Conference on computer vision and pattern recognition, pages 10034–10043, 2020.

#### [36] Chuofan Ma, Yi Jiang, Jiannan Wu, Zehuan Yuan, and Xiaojuan Qi. Groma: Localized visual tokenization

#### for grounding multimodal large language models.arXiv preprint arXiv:2404.13013, 2024.

#### [37] Xinyu Ma, Ziyang Ding, Zhicong Luo, Chi Chen, Zonghao Guo, Derek F Wong, Xiaoyi Feng, and

#### Maosong Sun. Deepperception: Advancing r1-like cognitive visual perception in mllms for knowledge-

#### intensive visual grounding.arXiv preprint arXiv:2503.12797, 2025.

#### [38] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L. Yuille, and Kevin Murphy.

#### Generation and comprehension of unambiguous object descriptions. InCVPR, pages 11–20, 2016.

#### [39] Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter,

#### Dhruti Shah, Xianzhi Du, Futang Peng, Floris Weers, Anton Belyi, Haotian Zhang, Karanjeet Singh,

#### Doug Kang, Ankur Jain, Hongyu Hè, Max Schwarzer, Tom Gunter, Xiang Kong, Aonan Zhang, Jianyu

#### Wang, Chong Wang, Nan Du, Tao Lei, Sam Wiseman, Guoli Yin, Mark Lee, Zirui Wang, Ruoming Pang,

#### Peter Grasch, Alexander Toshev, and Yinfei Yang. MM1: methods, analysis & insights from multimodal

#### LLM pre-training.arXiv: 2403.09611, 2024.

#### [40] Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Doso-

#### vitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, et al. Simple open-

#### vocabulary object detection. InEuropean conference on computer vision, pages 728–755. Springer, 2022.

#### [41] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke

#### Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling,

#### 2025.

#### [42] OpenAI, :, Ahmed El-Kishky, Alexander Wei, Andre Saraiva, Borys Minaiev, Daniel Selsam, David Do-

#### han, Francis Song, Hunter Lightman, Ignasi Clavera, Jakub Pachocki, Jerry Tworek, Lorenz Kuhn, Lukasz

#### Kaiser, Mark Chen, Max Schwarzer, Mostafa Rohaninejad, Nat McAleese, o3 contributors, Oleg Mürk,

#### Rhythm Garg, Rui Shu, Szymon Sidor, Vineet Kosaraju, and Wenda Zhou. Competitive programming

#### with large reasoning models, 2025.

#### [43] OpenAI. Gpt-4v(ision) system card.https://cdn.openai.com/papers/GPTV_System_Card.pdf,

#### 2023.

#### [44] Yi Peng, Xiaokun Wang, Yichen Wei, Jiangbo Pei, Weijie Qiu, Ai Jian, Yunzhuo Hao, Jiachun Pan,

#### Tianyidan Xie, Li Ge, et al. Skywork r1v: Pioneering multimodal reasoning with chain-of-thought.arXiv

#### preprint arXiv:2504.05599, 2025.

#### [45] Yanyuan Qiao, Chaorui Deng, and Qi Wu. Referring expression comprehension: A survey of methods

#### and datasets.IEEE Transactions on Multimedia, 23:4426–4440, 2020.

#### [46] Tianhe Ren, Yihao Chen, Qing Jiang, Zhaoyang Zeng, Yuda Xiong, Wenlong Liu, Zhengyu Ma, Junyi

#### Shen, Yuan Gao, Xiaoke Jiang, et al. Dino-x: A unified vision model for open-world object detection and

#### understanding.arXiv preprint arXiv:2411.14347, 2024.

#### [47] Tianhe Ren, Qing Jiang, Shilong Liu, Zhaoyang Zeng, Wenlong Liu, Han Gao, Hongjie Huang, Zhengyu

#### Ma, Xiaoke Jiang, Yihao Chen, et al. Grounding dino 1.5: Advance the" edge" of open-set object detec-

#### tion.arXiv preprint arXiv:2405.10300, 2024.

#### [48] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan

#### Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in

#### open language models, 2024.

#### [49] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang,

#### Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language

#### model.arXiv preprint arXiv:2504.07615, 2025.

#### [50] Andreas Steiner, André Susano Pinto, Michael Tschannen, Daniel Keysers, Xiao Wang, Yonatan Bitton,

#### Alexey Gritsenko, Matthias Minderer, Anthony Sherbondy, Shangbang Long, et al. Paligemma 2: A

#### family of versatile vlms for transfer.arXiv preprint arXiv:2412.03555, 2024.


#### [51] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao,

#### Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv

#### preprint arXiv:2501.12599, 2025.

#### [52] Omkar Thawakar, Dinura Dissanayake, Ketan More, Ritesh Thawkar, Ahmed Heakl, Noor Ahsan, Yuhao

#### Li, Mohammed Zumri, Jean Lahoud, Rao Muhammad Anwer, et al. Llamav-o1: Rethinking step-by-step

#### visual reasoning in llms.arXiv preprint arXiv:2501.06186, 2025.

#### [53] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin

#### Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any

#### resolution.arXiv preprint arXiv:2409.12191, 2024.

#### [54] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang,

#### Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models.arXiv preprint

#### arXiv:2311.03079, 2023.

#### [55] Yichen Wei, Yi Peng, Xiaokun Wang, Weijie Qiu, Wei Shen, Tianyidan Xie, Jiangbo Pei, Jianhao Zhang,

#### Yunzhuo Hao, Xuchen Song, et al. Skywork r1v2: Multimodal hybrid reinforcement learning for reason-

#### ing.arXiv preprint arXiv:2504.16656, 2025.

#### [56] Size Wu, Wenwei Zhang, Sheng Jin, Wentao Liu, and Chen Change Loy. Aligning bag of regions for

#### open-vocabulary object detection. InProceedings of the IEEE/CVF conference on computer vision and

#### pattern recognition, pages 15254–15264, 2023.

#### [57] Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, Yiyang Ma,

#### Chengyue Wu, Bingxuan Wang, et al. Deepseek-vl2: Mixture-of-experts vision-language models for

#### advanced multimodal understanding.arXiv preprint arXiv:2412.10302, 2024.

#### [58] Violet Xiang, Charlie Snell, Kanishk Gandhi, Alon Albalak, Anikait Singh, Chase Blagden, Duy Phung,

#### Rafael Rafailov, Nathan Lile, Dakota Mahan, Louis Castricato, Jan-Philipp Franken, Nick Haber, and

#### Chelsea Finn. Towards system 2 reasoning in llms: Learning how to think with meta chain-of-thought,

#### 2025.

#### [59] Wei Xiong, Hanning Zhang, Chenlu Ye, Lichang Chen, Nan Jiang, and Tong Zhang. Self-rewarding

#### correction for mathematical reasoning, 2025.

#### [60] Guowei Xu, Peng Jin, Li Hao, Yibing Song, Lichao Sun, and Li Yuan. Llava-o1: Let vision language

#### models reason step-by-step.arXiv preprint arXiv:2411.10440, 2024.

#### [61] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting

#### unleashes extraordinary visual grounding in gpt-4v.arXiv preprint arXiv:2310.11441, 2023.

#### [62] Jie Yang, ZENG Wang, Sheng Jin, Lumin Xu, Wentao Liu, Chen Qian, and Ruimao Zhang. Kptllm:

#### Unveiling the power of large language model for keypoint comprehension. InThe Thirty-eighth Annual

#### Conference on Neural Information Processing Systems, 2025.

#### [63] Sibei Yang, Guanbin Li, and Yizhou Yu. Dynamic graph attention for referring expression comprehension.

#### InProceedings of the IEEE/CVF international conference on computer vision, pages 4644–4653, 2019.

#### [64] Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin,

#### Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through

#### cross-modal formalization.arXiv preprint arXiv:2503.10615, 2025.

#### [65] Lewei Yao, Jianhua Han, Youpeng Wen, Xiaodan Liang, Dan Xu, Wei Zhang, Zhenguo Li, Chunjing

#### Xu, and Hang Xu. Detclip: Dictionary-enriched visual-concept paralleled pre-training for open-world

#### detection.Advances in Neural Information Processing Systems, 35:9125–9138, 2022.

#### [66] Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-

#### Fu Chang, and Yinfei Yang. Ferret: Refer and ground anything anywhere at any granularity. arXiv

#### preprint arXiv:2310.07704, 2023.

#### [67] En Yu, Kangheng Lin, Liang Zhao, Jisheng Yin, Yana Wei, Yuang Peng, Haoran Wei, Jianjian Sun,

#### Chunrui Han, Zheng Ge, et al. Perception-r1: Pioneering perception policy with reinforcement learning.

#### arXiv preprint arXiv:2504.07954, 2025.

#### [68] Licheng Yu, Zhe Lin, Xiaohui Shen, Jimei Yang, Xin Lu, Mohit Bansal, and Tamara L Berg. Mattnet:

#### Modular attention network for referring expression comprehension. InProceedings of the IEEE confer-

#### ence on computer vision and pattern recognition, pages 1307–1315, 2018.


#### [69] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C. Berg, and Tamara L. Berg. Modeling context in

#### referring expressions. InECCV, volume 9906, pages 69–85, 2016.

#### [70] Alireza Zareian, Kevin Dela Rosa, Derek Hao Hu, and Shih-Fu Chang. Open-vocabulary object detection

#### using captions. InProceedings of the IEEE/CVF conference on computer vision and pattern recognition,

#### pages 14393–14402, 2021.

#### [71] Yufei Zhan, Yousong Zhu, Zhiyang Chen, Fan Yang, Ming Tang, and Jinqiao Wang. Griffon: Spelling out

#### all object locations at any granularity with large language models. InEuropean Conference on Computer

#### Vision, pages 405–422. Springer, 2025.

#### [72] Yufei Zhan, Yousong Zhu, Hongyin Zhao, Fan Yang, Ming Tang, and Jinqiao Wang. Griffon v2: Advanc-

#### ing multimodal perception with high-resolution scaling and visual-language co-referring.arXiv preprint

#### arXiv:2403.09333, 2024.

#### [73] Chao Zhang, Weiming Li, Wanli Ouyang, Qiang Wang, Woo-Shik Kim, and Sunghoon Hong. Referring

#### expression comprehension with semantic visual relationship and word mapping. InProceedings of the

#### 27th ACM International Conference on Multimedia, pages 1258–1266, 2019.

#### [74] Haotian Zhang, Haoxuan You, Philipp Dufter, Bowen Zhang, Chen Chen, Hong-You Chen, Tsu-Jui Fu,

#### William Yang Wang, Shih-Fu Chang, Zhe Gan, et al. Ferret-v2: An improved baseline for referring and

#### grounding with large language models.arXiv preprint arXiv:2404.07973, 2024.

#### [75] Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao.

#### R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy

#### optimization.arXiv preprint arXiv:2503.12937, 2025.

#### [76] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, Hao Tian,

#### Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source

#### multimodal models.arXiv preprint arXiv:2504.10479, 2025.


## A Appendix

### A.1 More Details on Constructing HumanRef-CoT

### A.1.1 Prompt for GPT-4o

### To annotate HumanRef-CoT dataset using GPT-4o, we designed a two-part prompting strategy that

### addresses the diverse reasoning requirements across different subsets. This strategy consists of a

### unified system promptand a set ofsubset-specific in-context examples.

### The system prompt is shared across all subsets and instructs the model on how to interpret the input,

### which includes an image, a referring expression, and candidate bounding boxes. It also defines

### the expected format of the response, including the use of structured reasoning and answer tags. In

### addition to the system prompt, each of the six subsets in HumanRef-CoT namely attribute, position,

### interaction, reasoning, celebrity recognition, and rejection, is paired with a collection of in-context

### examples. These examples are carefully curated to reflect the specific annotation challenges and

### reasoning patterns required for each subset. They guide GPT-4o in producing chain-of-thought

### (CoT) rationales that are consistent with human annotations in both style and logic.

### In the following sections, we first present the shared system prompt. Then, for each subset, we

### provide the corresponding in-context examples and visualization results.

### Unified System Prompt.The system prompt instructs the model to perform detailed visual reason-

### ing based on either positional or attribute-based referring expressions. It emphasizes step-by-step

### analysis, beginning with predefined reasoning steps (first attributes, then orientation), and requires

### the model to explicitly evaluate each candidate object. Special symbols are also used to denote

### matching, non-matching, and reference entities during analysis.

###### Given the image and the description below, output a detailed analysis of how you found the object(s) matching the description about

###### position or attribute.

###### The position description is something like “the third man to the left of the boy wearing a red shirt”. Then you first need to find the

###### reference person, i.e. the boy wearing a red shirt, determine for each person of what you see and whether fulfill this reference condition,

###### and put a ⚠ if he is the reference person or object, and ❌if he doesn't. If this question is about attribute, you should provide a

###### description of what you see and how each object relates to the provided description, and put a ❌or ✅at the end. You first need to do

###### your analysis in a tuple like (analysis)(analysis), and then output your answer in a tuple like (answer)(answer). Note that in your analysis,

###### you need to start by listing your action steps. The action steps must be about attribute first, then orientation. And your first step doesn't

###### need to be to find all the people in the diagram, since I've already provided you with all of them. For example:

###### ```

###### To find the woman in red dress, I will need to excuse the following steps:

- Step1: Find all woman
- Step2: From the person in step1, I need to find all the person wearing red dress

###### ```

###### Each step you initially planned must be strictly enforced, and you cannot omit a step or modify the execution of each step.

###### Note that I will tell you which objects are the ground truth that fit this description, you need to use the answer I give as a reference. But

###### you can't refer to the answer I gave in your answer, pretend that you are thinking about it yourself. I will highlight all such objects as

###### marks in the diagram. Each mark has a circle and a number. The number represents the serial number of the object, and the color of the

###### circle represents whether or not the current object matches the description, with green circles representing objects that match the

###### description, and red circles representing objects that do not match the description. Note that you can't mention this mark in your answer,

###### this mark is just for you to go and specify the corresponding person with the corresponding serial number, you can say person 1, but not

###### mark 1.

###### Your final answer must be consistent with the analysis, e.g. if you say in the analysis that Person 1 and Person 2 satisfy the condition,

###### then you must also say in your final answer that it is Person 1 and Person 2. In the rare case that I give a reference answer that is

###### incorrect, you need to trust your own judgment.

###### Additionally I'll calculate the xy coordinates of each mark and then sort them in order from left to right to give you a positional reference

###### if the question is about position. But you can’t mention this order, pretend that you figure out the order all by yourself. Every useful

###### information will be provided in METAINFO

###### INCONTEXT EXAMPLES

### Figure 6: The system prompt used to instruct GPT-4o on visual reasoning for HumanRef-CoT. It

### specifies output format, reasoning steps, symbol conventions, and the expected alignment between

### intermediate analysis and final answers.

### Subset-Specific In-Context Examples.After the system prompt, we provide in-context examples

### to guide the model toward producing outputs aligned with our CoT structure. These examples help


### reinforce consistent reasoning patterns. HumanRef-CoT includes six subsets: attribute, position,

### interaction, reasoning, celebrity recognition, and rejection. Each subset uses its own set of in-context

### examples tailored to its specific reasoning needs.

### We show the in-context prompts used for each subset, along with representative outputs generated

### by GPT-4o.

###### Subset

###### attribute position interaction reasoning celebrity rejection

###### -

###### inner

###### position

###### outer

###### position

###### inner

###### interaction

###### outer

###### interaction

###### inner positon

###### reasoning

###### attribute

###### reasoning

- -

###### Prompt Figure 29 Figure 31 Figure 33 Figure 35 Figure 37 Figure 39 Figure 41 Figure 43 Figure 45

###### Example Figure 30 Figure 32 Figure 34 Figure 36 Figure 38 Figure 40 Figure 42 Figure 44 Figure 46

### Table 6: Ablation study on the retrieval-based design of our model. We compare performance with

### and without box hints to assess their impact on referring accuracy.

### A.1.2 Evaluate GPT-4o on HumanRef

### Since we use GPT-4o to annotate HumanRef-CoT, a natural question is how well GPT-4o performs

### directly on the HumanRef benchmark when prompted in a similar style. To investigate this, we

### adopt a setup similar to the annotation phase, using the same SoM-style prompt and a set of visual

### marks (with all marks shown in red). However, we remove any hint indicating which objects are

### correct. We then evaluate GPT-4o on the HumanRef-Benchmark without prompting with ground-

### truth answers. As shown in Table 7, GPT-4o achieves an average DF1 score of 53.2 without any hint

### supervision. This result suggests that while GPT-4o can be used to generate annotations when given

### the correct answer as reference, its standalone performance without answer supervision remains

### limited.

```
Method
```
```
Attribute Position Interaction Reasoning Celebrity Average Rejection
R P DF1 R P DF1 R P DF1 R P DF1 R P DF1 R P DF1 Score
GPT-4o-CoT 50.2 56.2 50.9 56.1 56.8 55.1 52.8 56.8 53.2 53.3 52.9 51.1 54.9 54.3 53.2 54.3 55.2 53.2 14.
Rex-Thinker-GRPO 88.5 88.7 84.1 87.2 87.1 84.6 81.5 83.5 79.1 87.7 85.4 82.3 88.0 89.3 87.2 86.6 86.8 83.5 68.
```
### Table 7: Evaluation of GPT-4o on the HumanRef-Benchmark test set using SoM-style prompts

### without answer hints. The model achieves 53.2 average DF1 score, indicating limited standalone

### performance.

### A.2 Experiment Details

### A.2.1 CoT SFT Settings

### Table 8 summarizes the full training hyperparameters and computational cost used during the CoT

### SFT stage. These settings were applied in the cold-start phase without prior instruction tuning.

#### batch size 4 maximum gradient norm 1 precision bf

#### gradient accumulation 4 learning rate scheduler cosine epochs 2

#### learning rate 2e-5 max length 2048 times 10.1h

#### optimizer AdamW deepspeed zero3 GPU 8xA

#### warm up ratio 0.03 weight decay 0.01 trainable module LLM

### Table 8: Training settings and cost statistics for CoT SFT.

### A.2.2 GRPO Settings

### We provide the training configurations used during the GRPO stage in Table 10. We did not run

### full GRPO training on the entire HumanRef-CoT dataset. Instead, training was terminated when the

### reward signal plateaued, indicating convergence.

### A.2.3 GRPO Training Analysis

### We analyze the training logs of the GRPO stage. As shown in Figure 7, we visualize the changes in

### both reward signals and completion length throughout training.

### Thanks to the cold-start CoT initialization, the model achieves a reasonably high accuracy reward

### at the beginning of GRPO training. At the same time, the format reward is nearly saturated from

### the start, indicating that the model has already learned to follow the correct output structure after

### CoT supervision. Meanwhile, the completion length remains stable at around 560 tokens throughout

### training. We attribute this to the model having already acquired the basic reasoning skills required


### batch size 8 num of rollout 8 precision bf

### gradient accumulation 2 β 0.04 epochs 0.

### learning rate 1e-6 temperature 1.0 times 112h

### optimizer AdamW deepspeed zero3 GPU 8xA

### warm up ratio 0.03 weight decay 0.01 trainable module LLM

### Table 9: Hyperparameters used during the GRPO training stage.

```
(a) (b) (c)
```
### Figure 7: GRPO training curves showing accuracy reward, format reward, and completion length

### over time.

### for the referring task during the CoT fine-tuning phase, resulting in consistent output lengths with

### minimal fluctuation.

### A.3 Limitations and Broader Impacts

### A.3.1 Inference Speed

### While the CoT-based design improves both interpretability and performance, it also introduces addi-

### tional computational overhead at inference time. To quantify this, we randomly selected 100 images

### from the HumanRef-Benchmark test set and compared the average inference time per image between

### RexThinker-Plain and RexThinker-GRPO.

### All experiments were conducted using the vLLM framework on a single NVIDIA A100 GPU. As

### shown in Table 3, RexThinker-GRPO exhibits slower inference due to its longer CoT-style outputs.

### This observation aligns with the general principle of test-time computation, where improved inter-

### pretability and accuracy often come at the cost of slower response time.

### model Rex-Thinker-Plain Rex-Thinker-GRPO

### average inference time 1.13s 6.68s

### Table 10: Comparison of average inference time between RexThinker-Plain and RexThinker-GRPO.

### The CoT design in GRPO leads to slower inference.

### A.3.2 Inconsistent Reasoning

### We observe occasional inconsistencies between the model’s reasoning process and its final output.

### For example, as shown in Figure 8, the model identifies nine candidate objects in the reasoning

### phase but only includes eight in the final predicted coordinates.

### We attribute such inconsistencies to the lack of explicit supervision enforcing alignment between

### the reasoning chain and the final answer during GRPO training. While most predictions remain

### consistent, these rare cases highlight a potential gap in our current framework. One possible solution

### is to introduce a consistency reward, which evaluates whether the number of objects summarized in

### the reasoning matches the number of bounding boxes in the final answer. This direction remains

### open for future exploration.

### A.3.3 Broader Impacts

### Referring models enable more flexible and natural interaction with visual scenes compared to tradi-

### tional object detection, as they can understand complex, context-dependent language. This makes

### them valuable in applications such as surveillance, smart cities, and smart homes.

### However, these models also raise concerns. In particular, they may expose privacy risks when

### deployed in sensitive environments and can inherit biases from training data or user input. Although


```
Question : Please detect all persons in the background.
Answer :
<think> OK, the user needs us to detect all persons in the background. In order to accomplish this
task, I need to break it down into the following steps:
```
- Step 1: Determine the position of each person relative to the foreground-background distinction.
Let's start our analysis:

```
# Step 1: Determine the position of each person relative to the foreground-background distinction
I see 18 people in this image. Let's analyze their positions one by one and put a ✅ if they are in
the background and ❌ if they are not.
...
```
- Person 9: She is in the foreground, sitting and holding a cat. ❌
- Person 10: He is in the background, partially visible behind Person 7. ✅
- Person 11: He is in the background, partially visible behind Person 5. ✅
- Person 12: He is in the background, partially visible behind Person 5. ✅
- Person 13: He is in the background, partially visible behind Person 5. ✅
- Person 14: He is in the background, partially visible behind Person 7. ✅
- Person 15: He is in the background, partially visible behind Person 3. ✅
- Person 16: He is in the background, partially visible behind Person 5. ✅
- Person 17: He is in the background, partially visible behind Person 6. ✅
- Person 18: He is in the background, partially visible behind Person 4. ✅

```
# Summarize and Re-Check answer
Let’s now recheck our answer and put ✅ for the target person and ❌ for others
```
- Person 1-9: In the foreground → ❌
- Person 10-18: In the background → ✅ **</think>
<answer>** ```json[{"bbox_2d": [916, 300, 952, 440], "label": "all persons in the background"},
{"bbox_2d": [730, 296, 769, 403], "label": "all persons in the background"}, {"bbox_2d": [769,
307, 790, 401], "label": "all persons in the background"}, {"bbox_2d": [939, 300, 963, 418],
"label": "all persons in the background"}, {"bbox_2d": [125, 289, 153, 341], "label": "all persons
in the background"}, {"bbox_2d": [678, 366, 704, 403], "label": "all persons in the
background"}, {"bbox_2d": [371, 361, 396, 400], "label": "all persons in the background"},
{"bbox_2d": [147, 308, 159, 338], "label": "all persons in the background"}]``` **</answer>**

```
Answer
```
```
Box hint
```
```
Answer
```
### Figure 8: Example of reasoning–answer mismatch. The number of predicted objects differs between

### reasoning and the final output.

```
Question : Please detect the youngest one.
Answer :
<think> OK, the user needs us to detect the youngest one. To accomplish this task, I need to break it down
into the following steps:
```
- Step 1: Determine the age-related attributes of each person.
- Step 2: Compare the age-related attributes to find the youngest one.

```
# Step 1: Determine the age-related attributes of each person
```
- Person 1: A young girl with light-colored hair, wearing a white outfit. She appears to be a child. ⚠
- Person 2: A woman with long brown hair, wearing a light-colored top and white pants. She is an adult. ❌
- Person 3: A man with short hair, wearing a beige sweater and brown pants. He is an adult. ❌
- Person 4: A young boy with light-colored hair, wearing a patterned shirt. He appears to be a child. ⚠
# Step 2: Compare the age-related attributes to find the youngest person
Among the four people, Person 1 and Person 4 are children, while Person 2 and Person 3 are adults.
Comparing Person 1 and Person 4, they both seem to be of similar young-age, but based on visual cues
like height and overall appearance, Person 4 appears to be slightly younger.

```
# Summarize and Re-Check answer
Let’s now recheck our answer and put ✅ for the target person and ❌ for others
```
- Person 1: She is a young girl but not the youngest. ❌
- Person 2: She is an adult. ❌
- Person 3: He is an adult. ❌
- Person 4: He is the youngest one. ✅ **</think>
<answer>** ```json[{"bbox_2d": [298, 94, 597, 410], "label": "the youngest one"}] ```</ **answer>**

```
Boxhint
```
```
Answer
```
### Figure 9: Attribute referring example.

### CoT reasoning improves interpretability, ensuring its consistency and robustness remains an open

### challenge. Future work should address these risks to ensure safe and responsible deployment.

### A.4 Visualization Results

### In this section, we present qualitative results of the RexThinker model across different scenarios, as

### shown in Figures 9–28. All visualizations are generated by the RexThinker-GRPO model, which is

### trained solely on the HumanRef-CoT dataset.


```
Question : Please detect the parents.
Answer :
<think> OK, the user needs us to detect the parents. In order to accomplish this task, I need to break it
down into the following steps:
```
- Step 1: Determine the gender and age-related attributes of each person to identify adults.
- Step 2: Among the adults, determine which ones are likely to be parents based on their interaction with
children. Let's start our analysis:

```
# Step 1: Determine the gender and age-related attributes of each person to identify adults
```
- Person 1: A young girl with long hair, wearing white clothes. She appears to be a child. ❌
- Person 2: A woman with long hair, wearing a light-colored top and white pants. She has an adult-like
appearance. ⚠
- Person 3: A man with short hair, wearing a beige sweater and brown pants. He has an adult-like
appearance. ⚠
- Person 4: A young boy with short hair, wearing a checkered shirt. He appears to be a child. ❌
# Step 2: Among the adults, determine which ones are likely to be parents based on their interaction with
children
Person 2 is sitting close to the children and has a nurturing-like posture with her arm around the girl
(Person 1). Person 3 is also in close proximity to the children and has a father-like role as he is interacting
with the boy (Person 4).

```
# Summarize and Re-Check answer
Let’s now recheck our answer and put ✅ for the target person and ❌ for others
```
- Person 1: She is a child, not a parent. ❌
- Person 2: She is an adult and interacting with children in a parental way. ✅
- Person 3: He is an adult and interacting with children in a parental way. ✅
- Person 4: He is a child, not a parent. ❌ **</think>
<answer>** ```json[{"bbox_2d": [545, 98, 1093, 740], "label": "parents"}, {"bbox_2d": [62, 95, 619, 757],
"label": "parents"}]```</ **answer>**

**Box hint**

```
Answer
```
### Figure 10: Attribute referring example.

```
Question : Please detect apple with disease, and tell me what kind of disease it may be..
Answer :
<think> OK, the user needs us to detect the apple with disease and tell what kind of disease it may be. In
order to accomplish this task, I need to break it down into the following steps:
```
- Step 1: Examine each apple to determine if it has any visible signs of disease.
- Step 2: If an apple is found to have a disease, identify the type of the disease. Let's start our analysis:

```
# Step 1: Examine each apple to determine if it has any visible signs of disease
I see five apples in this image. Let’s look at them one-by-one and put a ⚠if there are signs of disease
and ❌if there are none.
```
- Apple 1: It appears to be a healthy green apple with no visible signs of disease. ❌
- Apple 2: It is also a healthy-looking green apple with no obvious signs of disease. ❌
- Apple 3: This apple looks like a healthy green apple with no clear signs of disease. ❌
- Apple 4: Another healthy green apple with no visible signs of disease. ❌
- Apple 5: This apple has a brownish spot on its surface, which indicates that it has a disease. ⚠

```
# Step 2: Identify the type of the disease
The brownish spot on Apple 5 suggests that it may have a fungal or bacterial infection, which are
common types of diseases for apples.
```
```
# Summarize and Re-Check answer
Let’s now recheck our answer and put ✅for the target apple and ❌for others:
```
- Apple 1: It is a healthy apple with no signs of disease. ❌
- Apple 2: It is a healthy apple with no signs of disease. ❌
- Apple 3: It is a healthy apple with no signs of disease. ❌
- Apple 4: It is a healthy apple with no signs of disease. ❌
- Apple 5: It has a brownish spot, indicating a possible fungal or bacterial infection. ✅ **</think>
<answer** [{"bbox_2d": [429, 261, 570, 395], "label": "apple with disease"}]``` **</answer>**

```
Answer
```
```
Boxhint
```
### Figure 11: Attribute referring example.


```
Question : Please detect ripe tomato..
Answer :
<think> OK, the user needs us to detect the ripe tomato. In order to accomplish this task, I need to break
it down into the following steps:
```
- Step 1: Determine the ripeness of each tomato based on its color. Let's start our analysis:

```
# Step 1: Determine the ripeness of each tomato based on its color
I see seven tomatoes in this image, let’s look at their colors and put a ⚠ if it is a ripe tomato and ❌if it
is not.
```
- Tomato 1: It has a deep red color, indicating it is a ripe tomato. ⚠
- Tomato 2: It also has a deep red color, indicating it is a ripe tomato. ⚠
- Tomato 3: It has an orange-brown color, which is not the typical red color of a fully-ripe tomato. ❌
- Tomato 4: It is green in color, so it is not a ripe tomato. ❌
- Tomato 5: It is green in color, so it is not a ripe tomato. ❌
- Tomato 6: It is green in color, so it is not a ripe tomato. ❌
- Tomato 7: It is green in color, so it is not a ripe tomato. ❌

```
# Summarize and Re-Check answer
Let’s now recheck our answer and put ✅for the target tomato and ❌for others
```
- Tomato 1: It is a ripe tomato. ✅
- Tomato 2: It is a ripe tomato. ✅
- Tomato 3: It is not a ripe tomato. ❌
- Tomato 4: It is not a ripe tomato. ❌
- Tomato 5: It is not a ripe tomato. ❌
- Tomato 6: It is not a ripe tomato. ❌
- Tomato 7: It is not a ripe tomato. ❌ **</think>
<answer** [{"bbox_2d": [235, 358, 565, 657], "label": "ripe tomato"}, {"bbox_2d": [481, 455, 744, 726],
"label": "ripe tomato"}]``` **</answer>**

```
Answer
```
**Boxhint**

### Figure 12: Attribute referring example.

```
Question : Please detect damaged container..
Answer :
<think> OK, the user needs us to detect the damaged container. In order to accomplish this task, I need to
break it down into the following steps:
```
- Step 1: Examine each of the 31 containers to determine if they have any visible signs of damage.

```
# Step 1: Step 1: Examine each of the 31 containers to determine if they have any visible signs of
damage.
```
- Container 1: It appears to be in good condition with no visible signs of damage. ❌
- Container 2: There is no visible sign of damage on this container. ❌
- Container 3: This container has a large hole in its side, indicating significant damage. ⚠
- Container 4: No visible signs of damage. ❌
...
- Container 31: No visible signs of damage. ❌

```
# Summarize and Re-Check answer
Let’s now recheck our answer and put ✅for the target container and ❌for others
```
- Container 1: No visible signs of damage → ❌
- Container 2: No visible signs of damage → ❌
- Container 3: Has a large hole, is damaged → ✅
- Container 4: No visible signs of damage → ❌
...
- Container 30: No visible signs of damage → ❌
- Container 31: No visible signs of damage → ❌
**</think>
<answer** [{"bbox_2d": [288, 183, 699, 459], "label": "damaged container"}]``` **</answer>**

```
Answer
```
**Boxhint**

### Figure 13: Attribute referring example.


```
Question : Please detect athletes with an even number of number plates..
Answer :
<think> OK, the user needs us to detect athletes with an even-numbered number plates. To accomplish
this task, I need to break it down into the following steps:
```
- Step 1: Identify the number plate on each athlete.
- Step 2: Determine if the number plate is even.

```
# Step 1: Identify the number plate on each athlete
I see six athletes in this image. Let's look at the number plate on each of them:
```
- Athlete 1: His number plate reads "530".
- Athlete 2: His number plate reads "660".
- Athlete 3: His number plate reads "485".

```
# Step 2: Determine if the number plate is even
```
- Athlete 1: The number plate "530" is even. ✅
- Athlete 2: The number plate "660" is even. ✅
- Athlete 3: The number plate "485" is odd. ❌
- Athlete 4: The number plate "604" is even. ✅
- Athlete 5: The number plate "267" is odd. ❌
- Athlete 6: Since there is no visible number plate, we cannot determine if it is even or odd, but based on
the requirement of having an even-numbered plate, we can consider it as not meeting the condition. ❌

```
# Summarize and Re-Check answer
Let’s now recheck our answer and put ✅for the target athletes and ❌for others:
```
- Athlete 1: Has an even-numbered number plate. ✅
- Athlete 2: Has an even-numbered number plate. ✅
- Athlete 3: Has an odd-numbered number plate. ❌
**</think>
<answer** [{"bbox_2d": [221, 148, 372, 616], "label": "Athletes with an even number of number
plates"}...]``` **</answer>**

```
Answer
```
**Boxhint**

- Athlete 4: His number plate reads "604".
- Athlete 5: His number plate reads "267".
- Athlete 6: There is no visible number plate on this athlete.
    - Athlete 4: Has an even-numbered number plate. ✅
    - Athlete 5: Has an odd-numbered number plate. ❌
    - Athlete 6: No visible number plate. ❌

### Figure 14: Attribute referring example.

```
Question : Please detect the two people holding hands.
Answer :
<think> OK, the user needs us to detect two people holding hands. To accomplish this task, I need
to break it down into the following steps:
```
- Step 1: Identify pairs of people engaged in a hand-holding interaction.

```
# Step 1: Identify pairs of people engaged in a hand-holding interaction.
I see 18 people in this image. Let's look at their interactions:
```
- Person 3: Taking a photo, no hand-holding interaction. → ❌
- Person 15: Standing near others, no hand-holding interaction. → ❌
- Person 4: Holding hands with Person 6.→ ✅
- Person 6: Holding hands with Person 4. → ✅
- Person 9: Sitting with a cat, no hand-holding interaction. → ❌
- Person 17: Standing near others, no hand-holding interaction. → ❌
...

```
# Summarize and Re-Check answer
Let’s now recheck our answer and put ✅for the target people and ❌for others:
```
- Person 3: Not involved in a hand-holding interaction → ❌
- Person 15: Not involved in a hand-holding interaction → ❌
- Person 18: Not involved in a hand-holding interaction → ❌
- Person 4: Involved in a hand-holding interaction with Person 6 → ✅
- Person 6: Involved in a hand-holding interaction with Person 4 → ✅
- Person 9: Not involved in a hand-holding interaction → ❌
- Person 9: Not involved in a hand-holding interaction → ❌
- Person 17: Not involved in a hand-holding interaction → ❌
... **</think>
<answer>** ```json[{"bbox_2d": [169, 161, 286, 591], "label": "Two people holding hands"},
{"bbox_2d": [259, 127, 400, 450], "label": "Two people holding hands"}]``` **</answer>**

```
Box hint
```
```
Answer
```
### Figure 15: Interaction referring example.


### Box hint Answer

```
Question: Please detect car in a crash
```
###### Answer:

**<think>** OK, the user needs us to detect the cars that are in a crash. In order to accomplish this task, I need to break it down into thefollowing
steps:

- Step 1: Examine each of the 22 cars to determine if they are involved in a crash. Let's start our analysis:

# Step 1: Examine each of the 22 cars to determine if they are involved in a crash

- Car 1: This car is clearly damaged and is surrounded by emergency vehicles and people, indicating it is in a crash. ⚠
- Car 2: This car is also damaged and is in close proximity to other damaged cars and emergency-related activity, so it is in a crash. ⚠
- Car 3: There are no visible signs of damage or being involved in an accident-like situation. ❌
- Car 4: Appears to be undamaged and not part of any crash-related activity. ❌
- Car 5: Is damaged and is among other damaged cars, so it is in a crash. ⚠
- Car 6: There are no clear signs of being in a crash. ❌
...
- Car 22: No visible signs of being in a crash. ❌

# Summarize and Re-Check answer
Let’s now recheck our answer and put ✅for the target cars and ❌for others

- Car 1: Is in a crash → ✅
- Car 2: Is in a crash → ✅
- Car 3: Is not in a crash → ❌
- Car 4: Is not in a crash → ❌
- Car 5: Is in a crash → ✅
...
- Car 22: Is not in a crash → ❌ **</think><answer>** ```json
[{"bbox_2d": [401, 208, 553, 303], "label": "car in crash"}, {"bbox_2d": [252, 262, 364, 362], "label": "car in crash"}, {"bbox_2d": [127, 207,
349, 309], "label": "car in crash"}]
``` **</answer>**

### Figure 16: Interaction referring example.

```
Question : Please detect all people with something in their hands.
Answer :
<think> OK, the user needs us to detect all people with something in their hands. To accomplish
this task, I need to break it down into the following steps:
```
- Step 1: Check each person to see if they have something in their hands.

```
# Step 1: Check each person to see if they have something in their hands
I see 18 people in this image, let’s check what is each person holding and put a ⚠ if they have
something in their hands and ❌ if they don't.
```
- Person 1: He has nothing in his hands. ❌
- Person 3: He is holding a camera. ⚠
- Person 5: He has nothing in his hands. ❌
- Person 7: He has nothing in his hands. ❌
- Person 9: She is holding a cat. ⚠ ...

```
# Summarize and Re-Check answer
Let’s now recheck our answer and put ✅ for the target people and ❌ for others
```
- Person 1: He has nothing in his hands. ❌
- Person 3: He is holding a camera. ✅
- Person 5: He has nothing in his hands. ❌
- Person 7: He has nothing in his hands. ❌
- Person 9: She is holding a cat. ✅... **</think>
<answer>** ```json[{"bbox_2d": [999, 271, 1178, 811], "label": "all people with something in their
hands"}, {"bbox_2d": [0, 231, 178, 783], "label": "all people with something in their hands"},
{"bbox_2d": [259, 127, 400, 450], "label": "all people with something in their hands"},
{"bbox_2d": [768, 485, 956, 811], "label": "all people with something in their hands"},
{"bbox_2d": [175, 411, 468, 811], "label": "all people with something in their
hands"}]``` **</answer>**

```
Answer
```
```
Box hint
```
```
Answer
```
- Person 2: She is holding a book. ⚠
- Person 4: She has nothing in her hands. ❌
- Person 6: He is holding an ice-cream. ⚠
- Person 8: He is holding a lollipop. ⚠
- Person 2: She is holding a book. ✅
- Person 4: She has nothing in her hands. ❌
- Person 6: He is holding an ice-cream. ✅
- Person 8: He is holding a lollipop. ✅

### Figure 17: Interaction referring example.


```
Question : Please detect person holding two footballs
Answer :
<think> OK, the user needs us to detect the person holding two footballs. In order to accomplish this
task, I need to break it down into the following steps:
```
- Step 1: Determine what each person is holding.
- Step 2: Find the person holding two footballs. Let's start our analysis:

```
# Step 1: Determine what each person is holding
I see 17 people in this image, let’s look at what they are holding and put a ⚠ if it is two footballs and ❌
if it is not.
```
- Person 1: He is not holding anything. ❌
- Person 2: He is not holding anything. ❌
- Person 3: He is not holding anything. ❌
- Person 4: He is holding two footballs. ⚠

```
So Person 4 is the person holding two footballs.
```
```
# Summarize and Re-Check answer
Let’s now recheck our answer and put ✅ for the target person and ❌ for others
```
- Person 1: He is not holding anything. ❌
- Person 2: He is not holding anything. ❌
- Person 3: He is not holding anything. ❌
- Person 4: He is holding two footballs. ✅
**...
</think>
<answer** [{"bbox_2d": [652, 49, 791, 412], "label": "person holding two footballs"}]
``` **</answer>**

```
Answer
```
**Boxhint**

```
...
```
- Person 11: He is not holding anything. ❌
- Person 12: He is holding a football. ❌
...
- Person 17: He is not holding anything. ❌

### Figure 18: Interaction referring example.

```
Question : Please detect the leader.
Answer :
<think> OK, the user needs us to detect the leader. In order to accomplish this task, I
need to break it down into the following steps:
```
- Step 1: Observe the positions of all 18 cars on the track.
- Step 2: Determine which car is in the lead position among them.

```
# Step 1: Observe the positions of all 18 cars on the track
There are 18 cars visible on the track in the image. They are spread out across different
positions on the circuit, some closer to the inner-lane and some near the outer-lane.
# Step 2: Determine which car is in the lead position among them
By looking at the overall layout of the cars on the track, we can see that Car 14 is
positioned ahead of the other cars, occupying the most forward-most spot on the track
compared to the others.
```
```
# Summarize and Re-Check answer
Let’s now recheck our answer and put ✅ for the target car and ❌ for others
...
```
- Car 11: He is not in the lead position → ❌
- Car 12: He is not in the lead position → ❌
- Car 13: He is not in the lead position → ❌
- Car 14: He is in the lead position → ✅
- Car 15: He is not in the lead position → ❌
- Car 16: He is not in the lead position → ❌
- Car 17: He is not in the lead position → ❌
- Car 18: He is not in the lead position → ❌ **</think>
<answer>** ```[{"bbox_2d": [1177, 579, 1380, 624], "label": "the leader"}]``` **</answer>**

```
Answer
```
```
Box hint
```
### Figure 19: Position referring example.


```
Question : Please detect all person to the right of the person wearing a yellow tie
Answer :
<think> OK, the user needs us to detect all person to the right of the person wearing a yellow tie. In order
to accomplish this task, I need to break it down into the following steps:
```
- Step 1: Find the person wearing a yellow tie.
- Step 2: Identify all the people to the right of the person found in Step 1

```
# Step 1: Find the person wearing a yellow tie
I see seven people in this image. Let's check their ties and put a ⚠ if it is a yellow tie and ❌if it is not.
```
- Person 1: She is wearing a red-colored uniform and no visible tie. ❌
- Person 2: He is wearing a red-colored uniform and no visible tie. ❌
- Person 3: He is wearing a dark suit with a red tie. ❌
- Person 4: He is wearing a dark suit with a red tie. ❌
- Person 5: He is wearing a dark suit with a yellow tie. ⚠
- Person 6: He is wearing a dark suit with a green tie. ❌
- Person 7: She is wearing a red-colored uniform and no visible tie. ❌

```
So Person 5 is the person wearing a yellow tie.
```
```
# Step 2: Identify all the people to the right of Person 5
The order from left to right is [Person 7, Person 5, Person 3, Person 4, Person 6, Person 2, Person 1].
People to the right of Person 5 are Person 3, Person 4, Person 6, Person 2, Person 1.
```
```
# Summarize and Re-Check answer
Let’s now recheck our answer and put ✅for the target people and ❌for others
```
- Person 1: She is to the right of the person wearing a yellow tie. ✅
...
- Person 5: He is the person wearing a yellow tie (reference person). ⚠
- Person 6: He is to the right of the person wearing a yellow tie. ✅
- Person 7: She is to the left of the person wearing a yellow tie. ❌
**</think>
<answer** ```json[{"bbox_2d": [838, 323, 1028, 783], "label": "all person to the right of the person
wearing a yellow tie"}..]``` **</answer>**

```
Answer
```
**Boxhint**

### Figure 20: Position referring example.

```
Question : Please detect Hot dogs on the grill
Answer :
<think> OK, the user needs us to detect the hot dogs on the grill. In order to accomplish this task, I need
to break it down into the following steps:
```
- Step 1: Determine the position of each hot-dog in relation to the grill

```
# Step 1: Determine the position of each hot-dog in relation to the grill
Let's analyze the position of each hot-dog:
```
- Hot dog 1: It is placed on a piece of paper on the table, not on the grill. ❌
- Hot dog 2: It is also on a piece of paper on the table, not on the grill. ❌
- Hot dog 3: It is placed on the metal grill. ⚠
- Hot dog 4: It is placed on the metal grill. ⚠

```
# Summarize and Re-Check answer
Let’s now recheck our answer and put ✅for the target hot-dogs and ❌for others:
```
- Hot dog 1: Not on the grill. ❌
- Hot dog 2: Not on the grill. ❌
- Hot dog 3: On the grill. ✅
- Hot dog 4: On the grill. ✅
**</think>
<answer** ```json[{"bbox_2d": [782, 118, 1070, 451], "label": "Hot dogs on the grill"}, {"bbox_2d":
[901, 281, 1284, 550], "label": "Hot dogs on the grill"}]``` **</answer>**

```
Answer
```
**Boxhint**

### Figure 21: Position referring example.


```
Question : Please detect Geoffrey Hinton and say what he is doing.
Answer :
<think> OK, the user needs us to detect Geoffrey Hinton and say what he is doing. In order to
accomplish this task, I need to break it down into the following steps:
```
- Step 1: Identify Geoffrey Hinton among the 68 people.
- Step 2: Determine what Geoffrey Hinton is doing.

```
# Step 1: Identify Geoffrey Hinton among the 68 people
I see 68 people in the image. Based on general knowledge of what Geoffrey Hinton looks like (an
older man with gray-white hair), I need to check each person one-by-one.
```
- Person 1: He is an older man with gray-white hair, dressed in formal attire, shaking hands with
another person. He matches the general appearance of Geoffrey Hinton. ⚠
- Person 2: He is also an older man with gray-white hair, dressed in formal wear, shaking hands with
Person 1. But he is not Geoffrey Hinton. ❌
- For the remaining 66 people, by observing their appearance and position relative to the main
interaction, they do not match the expected appearance of Geoffrey Hinton. ❌
# Step 2: Determine what Geoffrey Hinton is doing
Person 1 is engaged in an interaction with another individual. He appears to be shaking hands or
exchanging something with Person 2. This action seems to be part of a formal event, likely related to
the "NOBEL PRIZE" as indicated by the text on the floor.

```
# Summarize and Re-Check answer
Let’s now recheck our answer and put ✅ for the target person and ❌ for others
```
- Person 1: He is Geoffrey Hinton and is shaking hands, participating in a formal event. ✅
- Person 2: He is not Geoffrey Hinton. ❌
- For the remaining 66 people: They are not Geoffrey Hinton. ❌ **</think>
<answer>** ``` [{"bbox_2d": [391, 43, 652, 706], "label": "Geoffrey Hinton and say what he is
doing"}]``` **</answer>**

```
Answer
```
**Box hint**

### Figure 22: Celebrity referring example.

```
Question : Please detect all non-real persons.
Answer :
<think> OK, the user needs us to detect all non-real persons. In order to accomplish this task, I
need to break it down into the following steps:
```
- Step 1: Determine which people in the image are real or not based on their appearance and
context.

```
# Step 1: Determine which people in the image are real or not based on their appearance and
context. I see 68 people in this image. Let's analyze each person one by one:
```
- Person 1: He is a man in formal attire shaking hands with another person. He appears to be a real-
life participant in the event. ❌
- Person 3: He is a man standing in the audience area. He seems to be a real-life attendee. ❌
- Person 4: It is a statue in the background. It is not a real person. ⚠
- Person 13: It is part of the podium-like structure, not a real person. ⚠
...

```
# Summarize and Re-Check answer
Let’s now recheck our answer and put ✅ for the target person and ❌ for others
```
- Person 1: He is a real person. ❌
- Person 3: He is a real person. ❌
- Person 4: It is a statue, not a real person. ✅
- Person 13: It is part of the structure, not a real person. ✅
... **</think>
<answer>** ```json[{"bbox_2d": [562, 349, 619, 433], "label": "all non-real persons"}, {"bbox_2d":
[850, 382, 908, 462], "label": "all non-real persons"}]``` **</answer>**

```
Answer
```
**Box hint**

### Figure 23: Reasoning referring example.


```
Question : Please detect the one with rich Vitamin C.
Answer :
<think> OK, the user needs us to detect the one with rich Vitamin C. In order to accomplish this task, I
need to break it down into the following steps:
```
- Step 1: Identify the fruits and vegetables that are known to be rich in Vitamin C.
- Step 2: Check each of the 46 items in the image against the identified fruits and vegetables. Let's start
our analysis:

```
# Step 1: Identify the fruits and vegetables that are known to be rich in Vitamin C
Fruits and vegetables rich in Vitamin C include citrus fruits like oranges and lemons, limes, bell peppers,
broccoli, and some leafy greens.
# Step 2: Check each of the 46 items in the image against the identified fruits and vegetables
```
- Item 1: Avocado, not rich in Vitamin C ❌
- Item 7: Beetroot, not rich in Vitamin C ❌
- Item 11: Kale, rich in Vitamin C ⚠
- Item 15: Garlic, not rich in Vitamin C ❌
- Item 21: Orange, rich in Vitamin C ⚠
...

```
# Summarize and Re-Check answer
Let’s now recheck our answer and put ✅ for the target item and ❌ for others
```
- Item 5: Lime, rich in Vitamin C → ✅
- Item 11: Kale, rich in Vitamin C → ✅
- Item 16: Lime, rich in Vitamin C → ✅
- Item 21: Orange, rich in Vitamin C → ✅
- Item 29: Orange, rich in Vitamin C → ✅
- Others: Not rich in Vitamin C → ❌ **</think>
<answer>** ```[{"bbox_2d": [599, 616, 716, 743], "label": "the one with rich Vitamin C"} ...]``` **</answer>**

```
Answer
```
**Box hint**

- Item 5: Lime, rich in Vitamin C ⚠
- Item 8: Lime, rich in Vitamin C ⚠
- Item 13: Broccoli, rich in Vitamin C ⚠
- Item 16: Lime, rich in Vitamin C ⚠
- Item 29: Orange, rich in Vitamin C
- Item 8: Lime, rich in Vitamin C → ✅
- Item 13: Broccoli, rich in Vitamin C → ✅
- Item 18: Lemon, rich in Vitamin C → ✅
- Item 24: Grapefruit, rich in Vitamin C → ✅

### Figure 24: Reasoning referring example.

```
Question : Please detect the dog sleep on the bed with a pot under its body
Answer :
<think> OK, the user needs us to detect the dog sleep on the bed with a pot under its body. To
accomplish this task, I need to break it down into the following steps:
```
- Step 1: Find all the dogs that are sleeping on the bed.
- Step 2: From the dogs found in step 1, check if there is a pot under their bodies.

```
# Step 1: Find all the dogs that are sleeping on the bed
I see three animals in this image. Let's look at their positions and actions and put a ⚠ if they are sleeping
on the bed and ❌ if not.
```
- Dog 1: A brown dog lying on the bed, appears to be sleeping. → ⚠
- Dog 2: A dark-colored dog lying on the bed, appears to be sleeping. → ⚠
- Cat 3: A cat lying on the bed, but it is not a dog. → ❌

```
So Dog 1 and Dog 2 are the dogs sleeping on the bed.
```
```
# Step 2: From the dogs found in step 1, check if there is a pot under their bodies
```
- Dog 1: There is no pot under its body. → ❌
- Dog 2: There is no pot under its body. → ❌

```
However, if we consider the overall description and the fact that the reference for the existing condition
mentions only Dog 1, we can re-evaluate. Since the reference only includes Dog 1, we focus on it. Dog 1
is a dog sleeping on the bed, but there is no pot under its body. So there is no dog that meets the exact
description as stated, but if we consider the reference-related aspect, only Dog 1 is relevant among the
sleeping dogs on the bed.
```
```
# Summarize and Re-Check answer
Let’s now recheck our answer and put ✅ for the target person and ❌ for others
```
- Dog 1: A dog sleeping on the bed but no pot under its body → ❌
- Dog 2: A dog sleeping on the bed but no pot under its body → ❌
- Cat 3: Not a dog → ❌
**</think>
<answer>There is no such thing in the image.</answer>**

```
Answer
```
**Boxhint**

### Figure 25: Rejection referring example.


```
Question : Please detect person wearing cloth that has letter A
Answer :
<think> OK, the user needs us to detect the person wearing cloth that has letter A. To accomplish this
task, I need to break it down into the following steps:
```
- Step 1: Find all the people in the image.
- Step 2: Check the clothes of each person to see if there is the letter A on them.

```
# Step 1: Find all the people in the image.
There are 8 people in the image, listed from left-to-right as person 4, person 6, person 5, person 2, person
3, person 7, person 1, person 8.
```
```
# Step 2: Check the clothes of each person to see if there is the letter A on them.
```
- Person 4: Wearing a pink dress with the word "VOL" on it, no letter A → ❌
- Person 6: Wearing an orange dress with the word "VOL" on it, no letter A → ❌
- Person 5: Wearing a blue shirt with the word "L" on it, no letter A → ❌
- Person 2: Wearing a pink dress with the word "U" on it, no letter A → ❌
- Person 3: Wearing a yellow shirt with the word "N" on it, no letter A → ❌
- Person 7: Wearing a brown shirt with the word "T" on it, no letter A → ❌
- Person 1: Wearing a blue shirt with the word "ER" on it, no letter A → ❌
- Person 8: There is a blue shirt inside the box, but no person 8 is visible in terms of wearing clothes with
a letter A, and among the people visible, none have a letter A on their clothes.

```
We also note that person 4 and person 6 are part of the group mentioned in the additional information, but
they still do not have the letter A on their clothes.
```
```
# Summarize and Re-Check answer
Let’s now recheck our answer and put ✅ for the target person and ❌ for others
```
- Person 4: no letter A on clothes → ❌
- Person 6: no letter A on clothes → ❌
- Person 5: no letter A on clothes → ❌
- Person 2: no letter A on clothes → ❌
- Person 3: no letter A on clothes → ❌
- Person 7: no letter A on clothes → ❌
- Person 1: no letter A on clothes → ❌
- Person 8: no person 8 visible wearing relevant clothes and others don't have A either → ❌ **</think>
<answer>There is no such thing in the image.</answer>**

```
Answer
```
```
Boxhint
```
### Figure 26: Rejection referring example.

```
Question : Please detect Kobe Bryant.
Answer :
<think> OK, the user needs us to detect Kobe Bryant. To accomplish this task, I need to break it down into the
following steps:
```
- Step 1: Identify each person based on their appearance and basketball-related attributes.
- Step 2: Check if any of the identified people match the identity of Kobe Bryant.

```
# Step 1: Identify each person based on their appearance and basketball-related attributes.
```
- Person 1: A man wearing a white Dallas-branded basketball jersey, holding a basketball. He has short hair and a
beard.
- Person 2: A man wearing a green Boston-branded basketball jersey with the number 0. He is in a shooting pose.
- Person 3: A man wearing a white Denver Nuggets-branded basketball jersey with the number 15. He is in a
shooting pose.
- Person 4: A man wearing a black San Antonio Spurs-branded basketball jersey. He has a muscular build.
- Person 5: A man wearing a blue and yellow basketball jersey. He has a beard and is smiling.
- Person 6: Not clearly visible in terms of distinct features related to identification.
- Person 7: Partially visible, no clear distinct features for identification.
- Person 8: A man wearing a purple and yellow hoodie, not in a basketball-playing pose.
- Person 9: A man wearing a blue Thunder-branded basketball jersey with the number 2. He has a headband.
# Step 2: Check if any of the identified people match the identity of Kobe Bryant.
Kobe Bryant is a well-known basketball player, but none of the people in the image have the exact appearance
and context that can confirm them as Kobe Bryant...

```
# Summarize and Re-Check answer
Let’s now recheck our answer and put ✅ for the target person and ❌ for others
```
- Person 1-9: Not Kobe Bryant → ❌ **</think>
<answer>There is no such thing in the image.</answer>**

**Answer**

**Box hint**

### Figure 27: Rejection referring example.


#### Question : Please detect all fish, analyze their subcategories, and tell me if they are in a normal

#### or endangered state.

#### Answer :

#### <think> OK, the user needs us to detect all fish, analyze their subcategories, and tell if they are in a

#### normal or endangered state. To accomplish this task, I need to break it down into the following steps:

- Step 1: Identify all the fish in the image.
- Step 2: Analyze the subcategories of the identified fish.
- Step 3: Determine the state (normal or endangered) of each fish.

#### # Step 1: Identify all the fish in the image

#### There are six sea-related characters in the image. Let's check which ones are fish:

- Fish 1: A clown fish with orange and white stripes. ⚠
- Fish 2: A puffer fish with a round body and spiky fins. ⚠
- Fish 3: A stingray with a flat, wing-like body. ⚠
- Fish 4: A blue whale with a long body and flippers. ⚠
- Fish 5: A jellyfish with a pink body and blue tentacles. ❌
- Fish 6: An oyster with a purple shell and an eye-like feature. ❌

#### So, Fish 1, Fish 2, Fish 3, and Fish 4 are fish.

#### # Step 2: Analyze the subcategories of the identified fish

- Fish 1: A clown fish, which is a type of fish.
- Fish 2: A puffer fish, which is a type of fish.
- Fish 3: A stingray, which is a type of fish.
- Fish 4: A blue whale, which is a type of fish.

#### # Step 3: Determine the state (normal or endangered) of each fish

- Fish 1: The clown fish is a common and widely-distributed fish, so it is in a normal state. ✅
- Fish 2: The puffer fish is also a relatively common fish, so it is in a normal state. ✅
- Fish 3: The stingray is not as common as some other fish, but still not on the endangered list, so it is

#### in a normal state. ✅

- Fish 4: The blue whale is an endangered species due to hunting and habitat loss. ❌

#### # Summarize and Re-Check answer

#### Let’s now recheck our answer and put ✅ for the target fish and ❌ for others:

- Fish 1: In a normal state. ✅
- Fish 2: In a normal state. ✅
- Fish 3: In a normal state. ✅
- Fish 4: Endangered. ❌
- Fish 5: Not a fish. ❌
- Fish 6: Not a fish. ❌ **</think>**

#### <answer> ``` [{"bbox_2d": [515, 156, 785, 348], "label": "all fish, analyze their subcategories, and

#### tell me if they are in a normal or endangered state"}, {"bbox_2d": [512, 457, 764, 705], "label": "all

#### fish, analyze their subcategories, and tell me if they are in a normal or endangered state"},

#### {"bbox_2d": [90, 116, 427, 397], "label": "all fish, analyze their subcategories, and tell me if they are

#### in a normal or endangered state"}]``` </answer>

#### Box hint Answer

### Figure 28: Reasoning referring example with multi-task chat.


##### ````

##### Example 1 :

##### ```

##### METAINFO:

##### {

##### “total_num_of_person”: 2,

##### “answer”: "Person 1",

##### “order_from_left_to_right”: [Person 1, Person 2]

##### }

##### Question: the person wearing a watch

##### The output should be:

##### (analysis)

##### OK, the user needs us to detect the person wearing a watch. To accomplish this task, I need to break it down into the following

##### steps:

- Step 1: Determine what each person is wearing on their wrists to check for a watch.

##### # Step 1: Determine what each person is wearing on their wrists to check for a watch

- Person 1: She is wearing a watch on her left wrist. ⚠
- Person 2: There is no visible watch on her wrists. ❌

##### # Summarize and Re - Check answer

##### Let’s now recheck our answer and put ✅for the target person and ❌for others

- Person 1: She is wearing a watch → ✅
- Person 2: She is not wearing a watch → ❌

##### (analysis)

##### (answer)

##### To summarize, Person 1 is the person wearing a watch

##### (answer)

### Figure 29: In-context prompt forattributesubset in HumanRef-CoT.

```
(analysis)
OK, the user needs us to detect the person with brownish -black curly hair. In order to
accomplish this task, I need to break it down into the following steps:
```
- Step 1: Examine the hair color and texture of each person. Let's start our analysis:

```
# Step 1: Examine the hair color and texture of each person
I see 8 people in this image, let’s look at their hair and put a ⚠if it is brownish -black curly hair
and ❌if it is not.
```
- Person 1: She has brownish -black curly hair. ⚠
- Person 2: She has dark hair that appears to be straight. ❌
- Person 3: He has short, light -colored hair. ❌
- Person 4: She has brownish -black curly hair. ⚠
- Person 5: He has short, dark hair. ❌
- Person 6: He has short, dark hair. ❌
- Person 7: He has short, dark hair. ❌
- Person 8: She has brownish -black curly hair. ⚠

```
# Summarize and Re -Check answer
Let’s now recheck our answer and put ✅for the target person and ❌for others
```
- Person 1: She has brownish -black curly hair. ✅
- Person 2: She has straight dark hair. ❌
- Person 3: He has short, light -colored hair. ❌
- Person 4: She has brownish -black curly hair. ✅
- Person 5: He has short, dark hair. ❌
- Person 6: He has short, dark hair. ❌
- Person 7: He has short, dark hair. ❌
- Person 8: She has brownish -black curly hair. ✅
(analysis)

```
(answer)
To summarize, Person 1, Person 4 and Person 8 are the people with brownish -black curly hair
(answer)
```
###### Referring Phrase: the person with brownish-black curly hair

### Figure 30: Visualization of GPT-4o’s output on theattributesubset.


###### Example 1 :

###### ```

###### METAINFO:

###### {

###### “total_num_of_person”: 4,

###### “answer”: “person 3”,

###### “order_from_left_to_right”: [person 1, person 2, person 3, person 4]

###### }

###### Question: The third person from the left to right

###### The output should be:

###### (analysis)

###### OK, the user needs us to detect the third person from the left to right. To accomplish this task, I need to break it down into the following steps:

- Step 1: Sort the people from left to right and find the third person from the left to right.

###### # Step 1: Sort the people from left to right and find the third person from the left to right.

###### I need to sort the people from left to right. I see 4 people in this image, and their order from left to right is [Person 1, Person 2, Person 3, Person 4].

###### So Person 3 is the third person from the left to right.

###### # Summarize and Re-Check answer

###### Let’s now recheck our answer and put ✅for the target person and ❌for others

- Person 1: He is the first person from left to right → ❌
- Person 2: He is the second person from left to right → ❌
- Person 3: He is the third person from left to right → ✅
- Person 4: He is the forth person from left to right → ❌

###### (analysis)

###### (answer)

###### To summarize, person 3 is the third person from left to right

###### (answer)

###### ```

### Figure 31: In-context prompt forposition (inner)subset in HumanRef-CoT.

```
(analysis)
OK, the user needs us to detect the second person wearing gloves when counting from right to
left. To accomplish this task, I need to break it down into the following steps:
```
- Step 1: Determine what each person is wearing and find the people wearing gloves.
- Step 2: Sort the results of the first step by position from right to left.

```
# Step 1: find all the people wearing gloves
I see four people in this image, let’s look at their hands and put a ⚠if they are wearing gloves
and ❌if they are not.
```
- Person 1: He is wearing black gloves. ⚠
- Person 2: She is wearing black gloves. ⚠
- Person 3: He is wearing black gloves. ⚠
- Person 4: He is wearing black gloves. ⚠

```
So Person 1, Person 2, Person 3, and Person 4 are all wearing gloves.
```
```
# Step 2: sort all the people wearing gloves from right to left
Now I need to sort their order from right to left. The order from right to left is [Person 1, Person
2, Person 3, Person 4].
```
```
So Person 2 is the second person wearing gloves when counting from right to left.
```
```
# Summarize and Re -Check answer
Let’s now recheck our answer and put ✅for the target person and ❌for others.
```
- Person 1: He is the first person wearing gloves when counting from right to left. ❌
- Person 2: She is the second person wearing gloves when counting from right to left. ✅
- Person 3: He is the third person wearing gloves when counting from right to left. ❌
- Person 4: He is the fourth person wearing gloves when counting from right to left. ❌
(analysis)

```
(answer)
To summarize, Person 2 is the second person wearing gloves when counting from right to left
(answer)
```
```
Referring Phrase: the second person wearing gloves when
counting from right to left
```
### Figure 32: Visualization of GPT-4o’s output on theposition (inner)subset.


```
(analysis)
OK, the user needs us to detect all the people to the right of the word 'muckle'. To accomplish
this task, I need to break it down into the following steps:
```
- Step 1: Locate the word 'muckle' in the image.
- Step 2: Identify all the people positioned to the right of the located word.

```
# Step 1: Locate the word 'muckle' in the image
The word 'muckle' is prominently displayed on a white counter - like structure in the center of
the image.
```
- Word 'muckle': On a white counter - like structure in the center of the image ✅

```
# Step 2: Identify all the people positioned to the right of the located word
Let's analyze the position of each person relative to the word 'muckle':
```
- Person 1: He is sitting on the counter with the word 'muckle' and is not to the right of it → ❌
- Person 2: He is standing to the right of the word 'muckle' → ✅
- Person 3: He is standing to the right of the word 'muckle' → ✅
- Person 4: He is standing to the right of the word 'muckle' → ✅
- Person 5: He is standing to the left of the word 'muckle' → ❌

```
# Summarize and Re - Check answer
Let’s now recheck our answer and put ✅for the target people and ❌for others:
```
- Person 1: Not to the right of the word 'muckle' → ❌
- Person 2: To the right of the word 'muckle' → ✅
- Person 3: To the right of the word 'muckle' → ✅
- Person 4: To the right of the word 'muckle' → ✅
- Person 5: Not to the right of the word 'muckle' → ❌
(analysis)

```
(answer)
To summarize, Person 2, Person 3, and Person 4 are all the people to the right of the word
'muckle'
(answer)
```
```
Referring Phrase: all the people to the right of the word
'muckle'
```
### Figure 33: In-context prompt forposition (outer)subset in HumanRef-CoT.

###### Example 1 :

###### ```

###### METAINFO:

###### {

###### “total_num_of_person”: 5,

###### “answer”: “person 1”,

###### “order_from_left_to_right”: [person 1, person 4, person 2, person 5, person 3]

###### }

###### Question: the person to the left of the green light bulb

###### The output should be:

###### (analysis)

###### OK, the user needs us to detect the person to the left of the green light-bulb. To accomplish this task, I need to break it downinto the following steps:

###### Step 1: Find the green light-bulb in this image.

###### Step 2: Fine the person to the left of the green light-bulb

###### # Step 1: Find the green light-bulb in this image.

###### Clearly, I can see a green light -bulb near the lower -left part of the image.

- green light -bulb: a green light -bulb near the lower -left part of the image ✅

###### # Step2: Fine the person to the left of the green light-bulb

###### Let's analyze the position of each person next:

- Person 1: He is standing directly to the left of the green light-bulb → ✅
- Person 2: She is on a ladder near the top-right part of the image, not to the left of the green light -bulb → ❌
- Person 3: He is standing near the right side of the image holding a tablet, not to the left of the green light -bulb → ❌

###### # Summarize and Re-Check answer

###### Let’s now recheck our answer and put ✅for the target person and ❌for others

- Person 1: He is standing directly to the left of the green light-bulb → ✅
- Person 2: She is on a ladder not the left of the green light-bulb → ❌
- Person 3: He is not on the left of the green light-bulb → ❌

###### (analysis)

###### (answer)

###### To summarize, person 1 is to the left of the green light bulb

###### (answer)

###### ```

### Figure 34: Visualization of GPT-4o’s output on theposition (outer)subset.


Example 1 :
```
METAINFO:
{
“total_num_of_person”: 3 ,
“answer”: “person 1 ”,
“order_from_left_to_right”: [person 1, person 2, person 3, person 4]
}

Question: The man who is hugging the woman from behind

The output should be:
(analysis)
OK, the user needs us to detect the man who is hugging the woman from behind

. To accomplish this task, I need to break it down into the following steps:
- Step 1: Find all the man in this image.
- Step 2: Look at each man’s action and find the man who is hugging the woman from behind

# Step 1: Find all the man in this image.
I see four person in this image, let’s look at their gender and put a ⚠if they are man and ❌if it is not.

- Person 1: It is a man sitting on the fence → ⚠
- Person 2: It is a man wearing a blue shirt → ⚠
- Person 3: It is a woman in red dress → ❌
- Person 4: It is a woman standing on the grass → ❌

# Step 2: Look at each man’s interaction from step 1
Person 1: He is man standing behind a woman (Person 3), with his arms around her, fulfilling the description of hugging the woman from behind → ✅
Person 2: A man sitting on the fence, holding hands with person 4. He is not hugging a woman from behind. → ❌
So Person 1 is the man hugging the woman (Person 3) from behind

# Summarize and Re-Check answer
Let’s now recheck our answer and put ✅for the target person and ❌for others

- Person 1: man hugging the woman from behind → ✅
- Person 2: man not hugging from behind → ❌
- Person 3: a woman, not a man hugging from behind → ❌
- Person 4:a woman, not a man hugging from behind → ❌
(analysis)

(answer)
To summarize, person 1 is the man who is hugging the woman from behind
(answer)

### Figure 35: In-context prompt forinteraction (inner)subset in HumanRef-CoT.

```
(analysis)
OK, the user needs us to detect two people who are holding hands. To accomplish
this task, I need to break it down into the following steps:
```
- Step 1: Identify the pair of people engaged in a hand -holding interaction.

```
# Step 1: Identify the pair of people engaged in a hand -holding interaction.
I see four people in this image. Let's look at their interactions:
```
- Person 1: He is walking with his arm around person 4, not holding hands. → ❌
- Person 2: He is holding hands with person 3. → ✅
- Person 3: She is holding hands with person 2. → ✅
- Person 4: She is being walked with an arm around by person 1, not holding hands.
→ ❌

```
# Summarize and Re -Check answer
Let’s now recheck our answer and put ✅for the target people and ❌for others:
```
- Person 1: Not involved in a hand -holding interaction → ❌
- Person 2: Involved in a hand -holding interaction with Person 3 → ✅
- Person 3: Involved in a hand -holding interaction with Person 2 → ✅
- Person 4: Not involved in a hand -holding interaction → ❌
(analysis)

```
(answer)
To summarize, person 2 and person 3 are the two people who are holding hands.
(answer)
```
###### Referring Phrase: Two people who are holding hands

### Figure 36: Visualization of GPT-4o’s output on theinteraction (inner)subset.


Example 1 :
```
METAINFO:
{
“total_num_of_person”: 3,
“answer”: “person 1”,
“order_from_left_to_right”: [person 2, person 3, person 1]
}

Question: The woman holding a red sniper rifle in her hand

The output should be:
(analysis)
OK, the user needs us to find the woman holding a red sniper-rifle in her hand. To accomplish this task, I need to break it downinto the following steps:

- Step 1: See if everyone is holding objects in their hands and what they're holding

# Step1: See if everyone is holding objects in their hands and what they're holding
There are 5 people in the picture, and we need to check the objects held by each of them:

- Person 1: A woman with dark hair and a red -themed outfit. She is holding a long, red firearm which appears to be a sniper -rifle. So she matches the
description → ✅
- Person 2: A woman with short orange hair, wearing a pink and white outfit. She is holding a gray and pink -colored weapon that does not resemble a red
sniper -rifle → ❌
- Person 3: A woman with long orange hair and a green and white dress. She is holding a weapon, but it is not a red sniper -rifle → ❌

# Summarize and Re-Check answer
Let’s now recheck our answer and put ✅for the target person and ❌for others
Person 1: holding a red sniper -rifle → ✅
Person 2: holding a non -red weapon → ❌
Person 3: holding a non -red weapon → ❌
(analysis)

(answer)
To summarize, person 1 is the woman holding a red sniper rifle in her hand.
(answer)

### Figure 37: In-context prompt forinteraction (outer)subset in HumanRef-CoT.

```
(analysis)
OK, the user needs us to find the person holding a microphone in the hand. To
accomplish this task, I need to break it down into the following steps:
```
- Step 1: Check the hands of each person to see if they are holding a microphone.

```
# Step 1: Check the hands of each person to see if they are holding a microphone
```
- Person 1: He is holding a microphone in his right hand. → ✅
- Person 2: His hands are on his hips, not holding a microphone. → ❌
- Person 3: His hands are gesturing, not holding a microphone. → ❌
- Person 4: He is holding a microphone in his left hand. → ✅
- Person 5: His hands are making a gesture, not holding a microphone. → ❌
- Person 6: His hands are positioned near his body, not holding a microphone. → ❌
- Person 7: His hands are gesturing, not holding a microphone. → ❌

```
# Summarize and Re - Check answer
Let’s now recheck our answer and put ✅for the target people and ❌for others:
```
- Person 1: Holding a microphone → ✅
- Person 2: Not holding a microphone → ❌
- Person 3: Not holding a microphone → ❌
- Person 4: Holding a microphone → ✅
- Person 5: Not holding a microphone → ❌
- Person 6: Not holding a microphone → ❌
- Person 7: Not holding a microphone → ❌
(analysis)
(answer)
To summarize, person 1 and person 4 are the persons holding a microphone in the hand.
(answer)

###### Referring Phrase: the person holding a microphone in the

###### hand

### Figure 38: Visualization of GPT-4o’s output on theinteraction (outer)subset.


Example 1 :
```
METAINFO:
{
“total_num_of_person”: 3,
“answer”: “person 1”,
“order_from_left_to_right”: [person 3 , person 1 , person 2 ]
}

Question: The woman to the left ot the woman in blue dress

The output should be:
(analysis)
OK, the user needs us to detect the woman to the left of the woman in the blue dress. In order to accomplish this task, I need to break it down into the following steps:
Step 1: Find the woman in the blue dress.
Step 2: Find the woman to her left.
Let's start our analysis:

# Step 1: Find the woman in the blue dress
I see three people in this image. Let's determine who is wearing a blue dress and put a ⚠ if it is and ❌if it is not.

- Person 1: She is wearing a pink dress. ❌
- Person 2: She is wearing a blue dress. ⚠
- Person 3: He is wearing a black suit with a pink tie. ❌

So Person 2 is the woman in the blue dress. Now I need to find the woman to her left.

# Step 2: Find the woman to the left of Person 2
The order from left to right is [Person 3, Person 1, Person 2]. So Person 1 is to the left of Person 2. And Person 1 is a woman.

# Summarize and Re - Check answer
Let’s now recheck our answer and put ✅for the target person and ❌for others

- Person 1: She is the woman to the left of the woman in the blue dress. ✅
- Person 2: She is the woman in the blue dress. ❌
- Person 3: He is a man, not the woman we are looking for. ❌
(analysis)

(answer)
To summarize, Person 1 is the woman to the left of the woman in the blue dress.
(answer)

### Figure 39: In-context prompt forreasoning (inner position)subset in HumanRef-CoT.

```
(analysis)
OK, the user needs us to detect the woman to the right of the man wearing a gold tie. In
order to accomplish this task, I need to break it down into the following steps:
Step 1: Find the man wearing a gold tie.
Step 2: Determine the person to the right of the man found in Step 1 and check if it is a
woman.
Let's start our analysis:
```
```
# Step 1: Find the man wearing a gold tie
I see five people in this image. Let's check their ties and put a ⚠ if it is a gold tie and ❌ if it
is not.
```
- Person 1: He is wearing a gold - patterned tie. ⚠
- Person 2: She is a woman with long blonde hair and no tie. ❌
- Person 3: She is a woman with short blonde hair and no tie. ❌
- Person 4: Only part of his head is visible and no tie is seen. ❌
- Person 5: Only a small part of his body is visible and no tie is seen. ❌

```
So Person 1 is the man wearing a gold tie.
```
```
# Step 2: Find the person to the right of Person 1 and check the gender
The order from left to right is [Person 4, Person 1, Person 2, Person 3, Person 5].
The person to the right of Person 1 is Person 2. Person 2 is a woman.
```
```
# Summarize and Re - Check answer
Let's now recheck our answer and put ✅ for the target person and ❌ for others:
```
- Person 1: He is the man wearing a gold tie (reference person). ⚠
- Person 2: She is the woman to the right of the man wearing a gold tie. ✅
- Person 3: She is a woman but not to the right of the man with the gold tie. ❌
- Person 4: He is not the reference person and no gold tie is seen. ❌
- Person 5: He is not relevant to the reference person and no gold tie is seen. ❌
(analysis)

```
(answer)
To summarize, Person 2 is the woman to the right of the man wearing a gold tie.
(answer)
```
###### Referring Phrase: the person holding a microphone in the

###### hand

### Figure 40: Visualization of GPT-4o’s output on thereasoning (inner position)subset.

### .


Example 1 :
```
METAINFO:
{
“total_num_of_person”: 4,
“answer”: “person 3, person 2, person 1”,
“order_from_left_to_right”: [person 4, person 2, person 3, person 1]
}

Question: Person wearing glasses but not sitting on a chair

The output should be:
(analysis)
OK, the user needs us to detect the person wearing glasses but not sitting on a chair. To accomplish this task, I need to break it down into the following steps:
Step 1: Find all person wearing glasses.
Step 2: Find the person in Step 1 that are sitting on a chair
Step 3: Remove person in Step 2 from Step 1 to get the final result

# Step 1: Find all person wearing glasses.
I see five people in this image, let’s look at their face and put a ⚠if they are wearing glasses and ❌if it is not.

- Person 1: A man wearing a red shirt, he is wearing glasses → ⚠
- Person 2: A man wearing a light-blue shirt, he is wearing glasses → ❌
- Person 3: A woman with blonde hair, she is not wearing glasses → ❌
Person 4: A woman in a white top, she is not wearing glasses → ⚠
Person 5: A man wearing a light blue long sleeved shirt, he is wearing glasses → ⚠

So Person 1, Person 4 and Person 5 are wearing glasses

# Step 2: Find the person in Step 1 that are sitting on a chair
Let’s look at the three people in Step 1 and put a ⚠if they are sitting on a chair and ❌if it is not.

- Person 1: He is sitting on a chair → ⚠
- Person 2: He is sitting on a chair → ⚠
Person 5: He is standing instead of sitting → ❌

# Step 3Remove person in Step 2 from Step 1 to get the final result
Remove Person 1 and Person 2 and we get Person 5 that meets the description

Let’s now recheck our answer and put ✅for the target person and ❌for others
Person 1: wearing glasses but sitting on a chair → ❌
Person 2: wearing glasses but sitting on a chair → ❌
Person 3: not wearing glasses → ❌
Person 4: not wearing glasses → ❌
Person 5: wearing glasses and not sitting on a chair → ✅

(answer)
To summarize, person 3 is the third person from left to right
(answer)
```

### Figure 41: In-context prompt forreasoning (attribute)subset in HumanRef-CoT.

```
(analysis)
OK, the user needs us to detect the person straddling someone's shoulders but not wearing
long blue ripped jeans. To accomplish this task, I need to break it down into the following
steps:
Step 1: Find all person straddling someone's shoulders.
Step 2: Find the person in Step 1 that are wearing long blue ripped jeans.
Step 3: Remove person in Step 2 from Step 1 to get the final result
```
```
# Step 1: Find all person straddling someone's shoulders.
I see four people in this image, let’s look at their position and put a ⚠ if they are straddling
someone's shoulders and ❌ if it is not.
```
- Person 1: A boy straddling a man's shoulders → ⚠
- Person 2: A woman not straddling anyone's shoulders → ❌
- Person 3: A boy straddling a woman's shoulders → ⚠
- Person 4: A man not straddling anyone's shoulders → ❌

```
So Person 1 and Person 3 are straddling someone's shoulders
```
```
# Step 2: Find the person in Step 1 that are wearing long blue ripped jeans.
Let’s look at the two people in Step 1 and put a ⚠ if they are wearing long blue ripped jeans
and ❌ if it is not.
```
- Person 1: He is wearing long blue ripped jeans → ⚠
- Person 3: He is wearing dark-colored pants, not long blue ripped jeans → ❌

```
# Step 3: Remove person in Step 2 from Step 1 to get the final result
Remove Person 1 from Step 1 and we get Person 3 that meets the description
```
```
# Summarize and Re-Check answer
Let’s now recheck our answer and put ✅ for the target person and ❌ for others
Person 1: straddling someone's shoulders and wearing long blue ripped jeans → ❌
Person 2: not straddling someone's shoulders → ❌
Person 3: straddling someone's shoulders and not wearing long blue ripped jeans → ✅
Person 4: not straddling someone's shoulders → ❌
(answer)
To summarize, the person is person 3
(answer)
```
###### Referring Phrase: the person holding a microphone in the

###### hand

### Figure 42: Visualization of GPT-4o’s output on thereasoning (attribute)

### subset..


###### Example:

###### METAINFO:

###### {

###### “total_num_of_person”: 4,

###### “answer”: “person 1 ”,

###### “order_from_left_to_right”: [person 4, person 1 , person 2 , person 3 ]

###### }

###### (analysis)

###### OK, the user needs us to detect Brad Pitt. In order to accomplish this task, I need to break it down into the following step:

- Step 1: Identify Brad Pitt among the 8 people based on his appearance characteristics. Let's start our analysis:

###### # Step 1: Identify Brad Pitt among the 8 people based on his appearance characteristics

- Person 1: Has light - colored hair, similar facial features to Brad Pitt, and is wearing a black jacket over a white shirt, which matches

###### the reference image of Brad Pitt. ⚠

- Person 2: A man with dark hair sitting on a red couch, does not match Brad Pitt's appearance. ❌
- Person 3: A woman with long - brown hair, clearly not Brad Pitt. ❌
- Person 4: A man with dark hair and a dark shirt, does not match Brad Pitt's appearance. ❌

###### # Summarize and Re - Check answer

###### Let’s now recheck our answer and put ✅for the target person and ❌for others

- Person 1: Matches Brad Pitt's appearance characteristics. ✅
- Person 2: Does not match Brad Pitt's appearance. ❌
- Person 3: Does not match Brad Pitt's appearance. ❌
- Person 4: Does not match Brad Pitt's appearance. ❌

###### (analysis)

###### (answer)

###### To summarize, Person 1 is Brad Pitt

###### (answer)

### Figure 43: In-context prompt forcelebrity recognitionsubset in HumanRef-CoT.

```
(analysis)
OK, the user needs us to detect Donald Sutherland. To accomplish this task, I need to break
it down into the following steps:
```
- Step 1: Recall what Donald Sutherland looks like and match his appearance among the
people in the image.

```
# Step 1: Recall what Donald Sutherland looks like and match his appearance among the
people in the image.
I know that Donald Sutherland is an older - looking man with white hair and a beard.
```
- Person 1: He is a younger - looking man with dark hair and no beard. He does not match
the appearance of Donald Sutherland. ❌
- Person 2: He is a man with short, light - brown hair and a short beard. He does not match
the appearance of Donald Sutherland. ❌
- Person 3: He is an older man with white hair and a beard. He matches the appearance of
Donald Sutherland. ⚠
- Person 4: He is a man with short, light - colored hair and no beard. He does not match the
appearance of Donald Sutherland. ❌

```
# Summarize and Re - Check answer
Let’s now recheck our answer and put ✅ for the target person and ❌ for others
```
- Person 1: Does not match the appearance of Donald Sutherland. ❌
- Person 2: Does not match the appearance of Donald Sutherland. ❌
- Person 3: Matches the appearance of Donald Sutherland. ✅
- Person 4: Does not match the appearance of Donald Sutherland. ❌
(analysis)

```
(answer)
To summarize, Person 3 is Donald Sutherland
(answer)
```
###### Referring Phrase: Donald Sutherland

### Figure 44: Visualization of GPT-4o’s output on thecelebrity recognition

### subset..


```
Example:
```
```
METAINFO:
{
“total_num_of_person”: 4,
“reference_answer”: “there is not person that meets the description”
}
```
```
(analysis)
OK, the user needs us to detect all the people with guns in the picture. I need to look at the objects held in each person's hands and determine if they are guns or
not. I see 7 people in the diagram, let's analyze the objects held by each of them:
```
- Person 1: She is a woman with long black hair, her right fist is outstretched but she is not holding a gun, and we can't see her left hand, so we can tell that she
is not holding a gun. ❌
- Person 2: A female with brown hair wearing a thin red jacket. She has her left hand up and looks nervous, but she's not holding a gun either. ❌
- Person 3: A bald policeman in a black jacket, we can't see his hands, so we don't think he's holding a gun either. ❌
- Person 4: A bald black male with a big mouth, he has his hands open but he is also not holding a gun. ❌

```
Let’s now finalize:
Person 1: not holding a gun → ❌
Person 2: not holding a gun → ❌
Person 3: not holding a gun → ❌
Person 4: not holding a gun → ❌
(analysis)
```
```
(answer)
To summarize, there is no person in the image that meets the description.
(Answer)
```
### Figure 45: In-context prompt forrejectionsubset in HumanRef-CoT.

```
(analysis)
OK, the user needs us to detect the person wearing sunglasses. To accomplish this task, I
need to break it down into the following steps:
```
- Step 1: Examine the attire and accessories of each person to find those wearing sunglasses.

```
# Step 1: Examine the attire and accessories of each person to find those wearing
sunglasses.
```
- Person 1: A young girl in a pink - colored coat. She is not wearing sunglasses.
- Person 2: A boy in a checkered shirt and a black tie. He is not wearing sunglasses.
- Person 3: A young girl in a beige - colored coat with a patterned collar. She is not wearing
sunglasses.
- Person 4: A woman in a metallic - green jacket and a patterned dress. She is not wearing
sunglasses.
- Person 5: A man in a dark suit and a light - colored shirt with a tie. He is not wearing
sunglasses, but he is the person mentioned in the condition related content (implied from the
hint).

```
# Summarize and Re - Check answer
Let’s now recheck our answer and put ✅ for the target person and ❌ for others
```
- Person 1: not wearing sunglasses → ❌
- Person 2: not wearing sunglasses → ❌
- Person 3: not wearing sunglasses → ❌
- Person 4: not wearing sunglasses → ❌
- Person 5: not wearing sunglasses → ❌
(analysis)

```
(answer)
To summarize, there are no people that meet the requirements as none of the five people are
wearing sunglasses.
(answer)
```
###### Referring Phrase: the person wearing sunglasses

### Figure 46: Visualization of GPT-4o’s output on therejection

### subset..


