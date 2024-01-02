# Awesome Data Contamination

[![Awesome](https://awesome.re/badge.svg)](https://github.com/lyy1994/awesome-data-contamination) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/github/last-commit/lyy1994/awesome-data-contamination?color=green) 
![](https://img.shields.io/badge/PRs-Welcome-red)

The paper list on [data contamination](https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html) for large language models evaluation.

## 🔔 News

- **[2024-01-02]** We create this repository to maintain a paper list on *Data Contamination*.

## 🔍 Contents

- [🌟 Data Contamination](#intro)
- [📜 Papers](#papers)
    - [🏷️ Tagset](#tagset)
    - [🎯 The List](#list)
- [🧰 Resources](#resources)
    - [📊 Datasets](#datasets)
    - [🛠️ Tools](#tools)
- [🚩 Citation](#citation)
- [🎉 Contribution](#contribution)
- [🤝 Acknowledgement](#acknowledgement)

<a id="intro"></a>
## 🌟 Data Contamination

Data Contamination, also known as [train-test contamination](https://arxiv.org/abs/2211.09110) or [benchmark leakage](https://arxiv.org/abs/2311.01964), indicates the case in which the model has seen information (e.g., test instances, test prompts, etc.) about the test set to be evaluated on during training. This issue has become particularly crucial in the era of **foundation models**, as they are typically trained on massive data that is poorly understood, raising the risk of unintentional contamination and resulting in a false positive on the model performance.

<a id="papers"></a>
## 📜 Papers

<a id="tagset"></a>
### 🏷️ Tagset

In this paper list, we tag each paper with one or more labels defined in the table below. These tags serve the purpose of facilitating the related work searching.

| Category | Explanation |
|----------|-------------|
| ![](https://img.shields.io/badge/Reactive-green) | This paper proposes the reactive approach(es) for identifying data contamination risk after the contamination happens. It is sometimes termed *contamination detection*. |
| ![](https://img.shields.io/badge/Preventative-blue) | This paper discusses the preventative approach(es) to *avoid* data contamination before it happens. |
| ![](https://img.shields.io/badge/Analysis-brown) | This paper formally discusses the data contamination problem and presents relevant observations and findings. |
| ![](https://img.shields.io/badge/Tool-purple) | This paper describes or provides a system or software for handling various data contamination challenges, e.g., detecting contamination, providing a contamination index, etc. |

<a id="list"></a>
### 🎯 The List

> [!Note]
> The list is sorted by the date of the first time the paper was released.

1. **Language Models are Few-Shot Learners** (NeurIPS 2020) ![](https://img.shields.io/badge/Reactive-green)![](https://img.shields.io/badge/Analysis-brown) <br />
    *Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei*
    [[paper](https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)]
    <details><summary><b>Abstract</b></summary>
    We demonstrate that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even becoming competitive with prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks. We also identify some datasets where GPT-3's few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora.
    </details>

    > This paper is not all about data contamination. Still, it is the very first paper that officially discusses the data contamination problem and presents an N-gram approach to identify the contamination risk of benchmarks (Appendix C).
1. **Data Contamination: From Memorization to Exploitation** (ACL 2022 Short) ![](https://img.shields.io/badge/Analysis-brown) <br />
    *Inbal Magar, Roy Schwartz*
    [[paper](https://aclanthology.org/2022.acl-short.18/)]
    <details><summary><b>Abstract</b></summary>
    Pretrained language models are typically trained on massive web-based datasets, which are often "contaminated" with downstream test sets. It is not clear to what extent models exploit the contaminated data for downstream tasks. We present a principled method to study this question. We pretrain BERT models on joint corpora of Wikipedia and labeled downstream datasets, and fine-tune them on the relevant task. Comparing performance between samples seen and unseen during pretraining enables us to define and quantify levels of memorization and exploitation. Experiments with two models and three downstream tasks show that exploitation exists in some cases, but in others the models memorize the contaminated data, but do not exploit it. We show that these two measures are affected by different factors such as the number of duplications of the contaminated data and the model size. Our results highlight the importance of analyzing massive web-scale datasets to verify that progress in NLP is obtained by better language understanding and not better data exploitation.
    </details>
1. **Holistic Evaluation of Language Models** (TMLR 2023) ![](https://img.shields.io/badge/Analysis-brown)![](https://img.shields.io/badge/Tool-purple) <br />
    *Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, Yuta Koreeda*
    [[paper](https://arxiv.org/abs/2211.09110)] [[code](https://github.com/stanford-crfm/helm)] [[website](https://crfm.stanford.edu/helm/classic/latest/)]
    <details><summary><b>Abstract</b></summary>
    Language models (LMs) are becoming the foundation for almost all major language technologies, but their capabilities, limitations, and risks are not well understood. We present Holistic Evaluation of Language Models (HELM) to improve the transparency of language models. First, we taxonomize the vast space of potential scenarios (i.e. use cases) and metrics (i.e. desiderata) that are of interest for LMs. Then we select a broad subset based on coverage and feasibility, noting what's missing or underrepresented (e.g. question answering for neglected English dialects, metrics for trustworthiness). Second, we adopt a multi-metric approach: We measure 7 metrics (accuracy, calibration, robustness, fairness, bias, toxicity, and efficiency) for each of 16 core scenarios when possible (87.5% of the time). This ensures metrics beyond accuracy don't fall to the wayside, and that trade-offs are clearly exposed. We also perform 7 targeted evaluations, based on 26 targeted scenarios, to analyze specific aspects (e.g. reasoning, disinformation). Third, we conduct a large-scale evaluation of 30 prominent language models (spanning open, limited-access, and closed models) on all 42 scenarios, 21 of which were not previously used in mainstream LM evaluation. Prior to HELM, models on average were evaluated on just 17.9% of the core HELM scenarios, with some prominent models not sharing a single scenario in common. We improve this to 96.0%: now all 30 models have been densely benchmarked on the same core scenarios and metrics under standardized conditions. Our evaluation surfaces 25 top-level findings. For full transparency, we release all raw model prompts and completions publicly for further analysis, as well as a general modular toolkit. We intend for HELM to be a living benchmark for the community, continuously updated with new scenarios, metrics, and models.
    </details>

    > This paper is not all about data contamination. It documents known evidence of contamination when possible (Appendix G).
1. **Koala: An Index for Quantifying Overlaps with Pre-training Corpora** (EMNLP 2023 Demo) ![](https://img.shields.io/badge/Tool-purple) <br />
    *Thuy-Trang Vu, Xuanli He, Gholamreza Haffari, Ehsan Shareghi*
    [[paper](https://aclanthology.org/2023.emnlp-demo.7/)]
    <details><summary><b>Abstract</b></summary>
    In very recent years more attention has been placed on probing the role of pre-training data in Large Language Models (LLMs) downstream behaviour. Despite the importance, there is no public tool that supports such analysis of pre-training corpora at large scale. To help research in this space, we launch Koala, a searchable index over large pre-training corpora using lossless compressed suffix arrays with highly efficient compression rate and search support. In its first release we index the public proportion of OPT 175B, GPT-3, GPT-Neo, GPT-Neo, LLaMA, BERT, ELECTRA, RoBERTA, XLNet pre-training corpora. Koala provides a framework to do forensic analysis on the current and future benchmarks as well as to assess the degree of memorization in the output from the LLMs. Koala is available for public use at <https://koala-index.erc.monash.edu/>.
    </details>
1. **Stop Uploading Test Data in Plain Text: Practical Strategies for Mitigating Data Contamination by Evaluation Benchmarks** (EMNLP 2023) ![](https://img.shields.io/badge/Preventative-blue) <br />
    *Alon Jacovi, Avi Caciularu, Omer Goldman, Yoav Goldberg*
    [[paper](https://aclanthology.org/2023.emnlp-main.308/)]
    <details><summary><b>Abstract</b></summary>
    Data contamination has become prevalent and challenging with the rise of models pretrained on large automatically-crawled corpora. For closed models, the training data becomes a trade secret, and even for open models, it is not trivial to detect contamination. Strategies such as leaderboards with hidden answers, or using test data which is guaranteed to be unseen, are expensive and become fragile with time. Assuming that all relevant actors value clean test data and will cooperate to mitigate data contamination, what can be done? We propose three strategies that can make a difference: (1) Test data made public should be encrypted with a public key and licensed to disallow derivative distribution; (2) demand training exclusion controls from closed API holders, and protect your test data by refusing to evaluate without them; (3) avoid data which appears with its solution on the internet, and release the web-page context of internet-derived data along with the data. These strategies are practical and can be effective in preventing data contamination.
    </details>
1. **CLEVA: Chinese Language Models EVAluation Platform** (EMNLP 2023 Demo) ![](https://img.shields.io/badge/Preventative-blue) <br />
    *Yanyang Li, Jianqiao Zhao, Duo Zheng, Zi-Yuan Hu, Zhi Chen, Xiaohui Su, Yongfeng Huang, Shijia Huang, Dahua Lin, Michael Lyu, Liwei Wang*
    [[paper](https://aclanthology.org/2023.emnlp-demo.17/)] [[dataset](https://github.com/LaVi-Lab/CLEVA)] [[website](http://www.lavicleva.com/)]
    <details><summary><b>Abstract</b></summary>
    With the continuous emergence of Chinese Large Language Models (LLMs), how to evaluate a model's capabilities has become an increasingly significant issue. The absence of a comprehensive Chinese benchmark that thoroughly assesses a model's performance, the unstandardized and incomparable prompting procedure, and the prevalent risk of contamination pose major challenges in the current evaluation of Chinese LLMs. We present CLEVA, a user-friendly platform crafted to holistically evaluate Chinese LLMs. Our platform employs a standardized workflow to assess LLMs' performance across various dimensions, regularly updating a competitive leaderboard. To alleviate contamination, CLEVA curates a significant proportion of new data and develops a sampling strategy that guarantees a unique subset for each leaderboard round. Empowered by an easy-to-use interface that requires just a few mouse clicks and a model API, users can conduct a thorough evaluation with minimal coding. Large-scale experiments featuring 23 Chinese LLMs have validated CLEVA's efficacy.
    </details>

    > This paper is not all about data contamination. It presents methods for alleviating the contamination issue from both the benchmark construction and leaderboard maintenance perspectives.
1. **Time Travel in LLMs: Tracing Data Contamination in Large Language Models** (arXiv, 16 Aug 2023) ![](https://img.shields.io/badge/Reactive-green) <br />
    *Shahriar Golchin, Mihai Surdeanu*
    [[paper](https://arxiv.org/abs/2308.08493)]
    <details><summary><b>Abstract</b></summary>
    Data contamination, i.e., the presence of test data from downstream tasks in the training data of large language models (LLMs), is a potential major issue in measuring LLMs' real effectiveness on other tasks. We propose a straightforward yet effective method for identifying data contamination within LLMs. At its core, our approach starts by identifying potential contamination at the instance level; using this information, our approach then assesses wider contamination at the partition level. To estimate contamination of individual instances, we employ "guided instruction:" a prompt consisting of the dataset name, partition type, and the random-length initial segment of a reference instance, asking the LLM to complete it. An instance is flagged as contaminated if the LLM's output either exactly or nearly matches the latter segment of the reference. To understand if an entire partition is contaminated, we propose two ideas. The first idea marks a dataset partition as contaminated if the average overlap score with the reference instances (as measured by ROUGE-L or BLEURT) is statistically significantly better with the completions from guided instruction compared to a "general instruction" that does not include the dataset and partition name. The second idea marks a dataset partition as contaminated if a classifier based on GPT-4 with few-shot in-context learning prompt marks multiple generated completions as exact/near-exact matches of the corresponding reference instances. Our best method achieves an accuracy between 92% and 100% in detecting if an LLM is contaminated with seven datasets, containing train and test/validation partitions, when contrasted with manual evaluation by human experts. Further, our findings indicate that GPT-4 is contaminated with AG News, WNLI, and XSum datasets.
    </details>
1. **Estimating Contamination via Perplexity: Quantifying Memorisation in Language Model Evaluation** (arXiv, 19 Sep 2023) ![](https://img.shields.io/badge/Reactive-green) <br />
    *Yucheng Li*
    [[paper](https://arxiv.org/abs/2309.10677)]
    <details><summary><b>Abstract</b></summary>
    Data contamination in model evaluation is getting increasingly prevalent as the massive training corpora of large language models often unintentionally include benchmark samples. Therefore, contamination analysis has became an inevitable part of reliable model evaluation. However, existing method of contamination analysis requires the access of the entire training data which is often confidential for recent models. This prevent the community to rigorously audit these models and conduct accurate assessment of their capability. In this paper, we propose a novel method to quantify contamination without the access of the full training set, that measure the extent of contamination with perplexity. Our analysis provides evidence of significant memorisation of recent foundation models in popular reading comprehension, summarisation benchmarks, while multiple choice appears less contaminated.
    </details>
1. **Data Contamination Through the Lens of Time** (arXiv, 16 Oct 2023) ![](https://img.shields.io/badge/Analysis-brown) <br />
    *Manley Roberts, Himanshu Thakur, Christine Herlihy, Colin White, Samuel Dooley*
    [[paper](https://arxiv.org/abs/2310.10628)]
    <details><summary><b>Abstract</b></summary>
    Recent claims about the impressive abilities of large language models (LLMs) are often supported by evaluating publicly available benchmarks. Since LLMs train on wide swaths of the internet, this practice raises concerns of data contamination, i.e., evaluating on examples that are explicitly or implicitly included in the training data. Data contamination remains notoriously challenging to measure and mitigate, even with partial attempts like controlled experimentation of training data, canary strings, or embedding similarities. In this work, we conduct the first thorough longitudinal analysis of data contamination in LLMs by using the natural experiment of training cutoffs in GPT models to look at benchmarks released over time. Specifically, we consider two code/mathematical problem-solving datasets, Codeforces and Project Euler, and find statistically significant trends among LLM pass rate vs. GitHub popularity and release date that provide strong evidence of contamination. By open-sourcing our dataset, raw results, and evaluation framework, our work paves the way for rigorous analyses of data contamination in modern models. We conclude with a discussion of best practices and future steps for publicly releasing benchmarks in the age of LLMs that train on webscale data.
    </details>
1. **Detecting Pretraining Data from Large Language Models** (arXiv, 25 Oct 2023) ![](https://img.shields.io/badge/Reactive-green)![](https://img.shields.io/badge/Tool-purple) <br />
    *Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, Luke Zettlemoyer*
    [[paper](https://arxiv.org/abs/2310.16789)] [[code](https://github.com/swj0419/detect-pretrain-code)] [[dataset](https://huggingface.co/datasets/swj0419/WikiMIA)] [[website](https://swj0419.github.io/detect-pretrain.github.io/)]
    <details><summary><b>Abstract</b></summary>
    Although large language models (LLMs) are widely deployed, the data used to train them is rarely disclosed. Given the incredible scale of this data, up to trillions of tokens, it is all but certain that it includes potentially problematic text such as copyrighted materials, personally identifiable information, and test data for widely reported reference benchmarks. However, we currently have no way to know which data of these types is included or in what proportions. In this paper, we study the pretraining data detection problem: given a piece of text and black-box access to an LLM without knowing the pretraining data, can we determine if the model was trained on the provided text? To facilitate this study, we introduce a dynamic benchmark WIKIMIA that uses data created before and after model training to support gold truth detection. We also introduce a new detection method Min-K% Prob based on a simple hypothesis: an unseen example is likely to contain a few outlier words with low probabilities under the LLM, while a seen example is less likely to have words with such low probabilities. Min-K% Prob can be applied without any knowledge about the pretraining corpus or any additional training, departing from previous detection methods that require training a reference model on data that is similar to the pretraining data. Moreover, our experiments demonstrate that Min-K% Prob achieves a 7.4% improvement on WIKIMIA over these previous methods. We apply Min-K% Prob to three real-world scenarios, copyrighted book detection, contaminated downstream example detection and privacy auditing of machine unlearning, and find it a consistently effective solution.
    </details>
1. **Proving Test Set Contamination in Black Box Language Models** (arXiv, 26 Oct 2023) ![](https://img.shields.io/badge/Reactive-green) <br />
    *Yonatan Oren, Nicole Meister, Niladri Chatterji, Faisal Ladhak, Tatsunori B. Hashimoto*
    [[paper](https://arxiv.org/abs/2310.17623)]
    <details><summary><b>Abstract</b></summary>
    Large language models are trained on vast amounts of internet data, prompting concerns and speculation that they have memorized public benchmarks. Going from speculation to proof of contamination is challenging, as the pretraining data used by proprietary models are often not publicly accessible. We show that it is possible to provide provable guarantees of test set contamination in language models without access to pretraining data or model weights. Our approach leverages the fact that when there is no data contamination, all orderings of an exchangeable benchmark should be equally likely. In contrast, the tendency for language models to memorize example order means that a contaminated language model will find certain canonical orderings to be much more likely than others. Our test flags potential contamination whenever the likelihood of a canonically ordered benchmark dataset is significantly higher than the likelihood after shuffling the examples. We demonstrate that our procedure is sensitive enough to reliably prove test set contamination in challenging situations, including models as small as 1.4 billion parameters, on small test sets of only 1000 examples, and datasets that appear only a few times in the pretraining corpus. Using our test, we audit five popular publicly accessible language models for test set contamination and find little evidence for pervasive contamination.
    </details>
1. **An Open Source Data Contamination Report for Large Language Models** (arXiv, 26 Oct 2023) ![](https://img.shields.io/badge/Analysis-brown)![](https://img.shields.io/badge/Tool-purple) <br />
    *Yucheng Li*
    [[paper](https://arxiv.org/abs/2310.17589)] [[code](https://github.com/liyucheng09/Contamination_Detector)]
    <details><summary><b>Abstract</b></summary>
    Data contamination in language model evaluation is increasingly prevalent as the popularity of large language models. It allows models to "cheat" via memorisation instead of displaying true capabilities. Therefore, contamination analysis has became an crucial part of reliable model evaluation to validate results. However, existing contamination analysis is usually conducted internally by LLM developers and often lacks transparency and completeness. This paper present an open source data contamination reports for the Llama series models. We analyse six popular multi-choice QA benchmarks and quantify their overlapping with the training set of Llama. Various levels of contamination ranging from 1\% to 8.7\% are found across benchmarks. Our comparison also reveals that Llama models can gain over 5\% higher accuracy on contaminated subsets versus clean subsets. Data and code are available at: <https://github.com/liyucheng09/Contamination_Detector>
    </details>
1. **NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark** (EMNLP 2023 Findings) ![](https://img.shields.io/badge/Analysis-brown)![](https://img.shields.io/badge/Tool-purple) <br />
    *Oscar Sainz, Jon Ander Campos, Iker García-Ferrero, Julen Etxaniz, Oier Lopez de Lacalle, Eneko Agirre*
    [[paper](https://aclanthology.org/2023.findings-emnlp.722/)] [[code](https://github.com/hitz-zentroa/lm-contamination)] [[website](https://hitz-zentroa.github.io/lm-contamination/)]
    <details><summary><b>Abstract</b></summary>
    In this position paper we argue that the classical evaluation on Natural Language Processing (NLP) tasks using annotated benchmarks is in trouble. The worst kind of data contamination happens when a Large Language Model (LLM) is trained on the test split of a benchmark, and then evaluated in the same benchmark. The extent of the problem is unknown, as it is not straightforward to measure. Contamination causes an overestimation of the performance of a contaminated model in a target benchmark and associated task with respect to their non-contaminated counterparts. The consequences can be very harmful, with wrong scientific conclusions being published while other correct ones are discarded. This position paper defines different levels of data contamination and argues for a community effort, including the development of automatic and semi-automatic measures to detect when data from a benchmark was exposed to a model, and suggestions for flagging papers with conclusions that are compromised by data contamination.
    </details>
1. **Don't Make Your LLM an Evaluation Benchmark Cheater** (arXiv, 3 Nov 2023) ![](https://img.shields.io/badge/Analysis-brown) <br />
    *Kun Zhou, Yutao Zhu, Zhipeng Chen, Wentong Chen, Wayne Xin Zhao, Xu Chen, Yankai Lin, Ji-Rong Wen, Jiawei Han*
    [[paper](https://arxiv.org/abs/2311.01964)]
    <details><summary><b>Abstract</b></summary>
    Large language models~(LLMs) have greatly advanced the frontiers of artificial intelligence, attaining remarkable improvement in model capacity. To assess the model performance, a typical approach is to construct evaluation benchmarks for measuring the ability level of LLMs in different aspects. Despite that a number of high-quality benchmarks have been released, the concerns about the appropriate use of these benchmarks and the fair comparison of different models are increasingly growing. Considering these concerns, in this paper, we discuss the potential risk and impact of inappropriately using evaluation benchmarks and misleadingly interpreting the evaluation results. Specially, we focus on a special issue that would lead to inappropriate evaluation, \ie \emph{benchmark leakage}, referring that the data related to evaluation sets is occasionally used for model training. This phenomenon now becomes more common since pre-training data is often prepared ahead of model test. We conduct extensive experiments to study the effect of benchmark leverage, and find that it can dramatically boost the evaluation results, which would finally lead to an unreliable assessment of model performance. To improve the use of existing evaluation benchmarks, we finally present several guidelines for both LLM developers and benchmark maintainers. We hope this work can draw attention to appropriate training and evaluation of LLMs.
    </details>
1. **Rethinking Benchmark and Contamination for Language Models with Rephrased Samples** (arXiv, 8 Nov 2023) ![](https://img.shields.io/badge/Analysis-brown)![](https://img.shields.io/badge/Tool-purple) <br />
    *Shuo Yang, Wei-Lin Chiang, Lianmin Zheng, Joseph E. Gonzalez, Ion Stoica*
    [[paper](https://arxiv.org/abs/2311.04850)] [[code](https://github.com/lm-sys/llm-decontaminator)]
    <details><summary><b>Abstract</b></summary>
    Large language models are increasingly trained on all the data ever produced by humans. Many have raised concerns about the trustworthiness of public benchmarks due to potential contamination in pre-training or fine-tuning datasets. While most data decontamination efforts apply string matching (e.g., n-gram overlap) to remove benchmark data, we show that these methods are insufficient, and simple variations of test data (e.g., paraphrasing, translation) can easily bypass these decontamination measures. Furthermore, we demonstrate that if such variation of test data is not eliminated, a 13B model can easily overfit a test benchmark and achieve drastically high performance, on par with GPT-4. We validate such observations in widely used benchmarks such as MMLU, GSK8k, and HumanEval. To address this growing risk, we propose a stronger LLM-based decontamination method and apply it to widely used pre-training and fine-tuning datasets, revealing significant previously unknown test overlap. For example, in pre-training sets such as RedPajama-Data-1T and StarCoder-Data, we identified that 8-18\% of the HumanEval benchmark overlaps. Interestingly, we also find such contamination in synthetic dataset generated by GPT-3.5/4, suggesting a potential risk of unintentional contamination. We urge the community to adopt stronger decontamination approaches when using public benchmarks. Moreover, we call for the community to actively develop fresh one-time exams to evaluate models accurately. Our decontamination tool is publicly available at <https://github.com/lm-sys/llm-decontaminator>.
    </details>
1. **Data Contamination Quiz: A Tool to Detect and Estimate Contamination in Large Language Models** (arXiv, 10 Nov 2023) ![](https://img.shields.io/badge/Reactive-green) <br />
    *Shahriar Golchin, Mihai Surdeanu*
    [[paper](https://arxiv.org/abs/2311.06233)]
    <details><summary><b>Abstract</b></summary>
    We propose the Data Contamination Quiz, a simple and effective approach to detect data contamination in large language models (LLMs) and estimate the amount of it. Specifically, we frame data contamination detection as a series of multiple-choice questions. We devise a quiz format wherein three perturbed versions of each dataset instance are created. These changes only include word-level perturbations, replacing words with their contextual synonyms, ensuring both the semantic and sentence structure remain exactly the same as the original instance. Together with the original instance, these perturbed versions constitute the choices in the quiz. Given that the only distinguishing signal among these choices is the exact wording, an LLM, when tasked with identifying the original instance from the choices, opts for the original if it has memorized it in its pre-training phase--a trait intrinsic to LLMs. A dataset partition is then marked as contaminated if the LLM's performance on the quiz surpasses what random chance suggests. Our evaluation spans seven datasets and their respective splits (train and test/validation) on two state-of-the-art LLMs: GPT-4 and GPT-3.5. While lacking access to the pre-training data, our results suggest that our approach not only enhances the detection of data contamination but also provides an accurate estimation of its extent, even when the contamination signal is weak.
    </details>
1. **Investigating Data Contamination in Modern Benchmarks for Large Language Models** (arXiv, 16 Nov 2023) ![](https://img.shields.io/badge/Reactive-green) <br />
    *Chunyuan Deng, Yilun Zhao, Xiangru Tang, Mark Gerstein, Arman Cohan*
    [[paper](https://arxiv.org/abs/2311.09783)]
    <details><summary><b>Abstract</b></summary>
    Recent observations have underscored a disparity between the inflated benchmark scores and the actual performance of LLMs, raising concerns about potential contamination of evaluation benchmarks. This issue is especially critical for closed-source models and certain open-source models where training data transparency is lacking. In this paper we study data contamination by proposing two methods tailored for both open-source and proprietary LLMs. We first introduce a retrieval-based system to explore potential overlaps between evaluation benchmarks and pretraining corpora. We further present a novel investigation protocol named \textbf{T}estset \textbf{S}lot Guessing (\textit{TS-Guessing}), applicable to both open and proprietary models. This approach entails masking a wrong answer in a multiple-choice question and prompting the model to fill in the gap. Additionally, it involves obscuring an unlikely word in an evaluation example and asking the model to produce it. We find that certain commercial LLMs could surprisingly guess the missing option in various test sets. Specifically, in the TruthfulQA benchmark, we find that LLMs exhibit notable performance improvement when provided with additional metadata in the benchmark. Further, in the MMLU benchmark, ChatGPT and GPT-4 demonstrated an exact match rate of 52\% and 57\%, respectively, in guessing the missing options in benchmark test data. We hope these results underscore the need for more robust evaluation methodologies and benchmarks in the field.
    </details>
1. **Task Contamination: Language Models May Not Be Few-Shot Anymore** (AAAI 2024) ![](https://img.shields.io/badge/Reactive-green)![](https://img.shields.io/badge/Analysis-brown) <br />
    *Changmao Li, Jeffrey Flanigan*
    [[paper](https://arxiv.org/abs/2312.16337)]
    <details><summary><b>Abstract</b></summary>
    Large language models (LLMs) offer impressive performance in various zero-shot and few-shot tasks. However, their success in zero-shot and few-shot settings may be affected by task contamination, a potential limitation that has not been thoroughly examined. This paper investigates how zero-shot and few-shot performance of LLMs has changed chronologically over time. Utilizing GPT-3 series models and several other recent open-sourced LLMs, and controlling for dataset difficulty, we find that on datasets released before the LLM training data creation date, LLMs perform surprisingly better than on datasets released after. This strongly indicates that, for many LLMs, there exists task contamination on zero-shot and few-shot evaluation for datasets released prior to the LLMs' training data creation date. Additionally, we utilize training data inspection, task example extraction, and a membership inference attack, which reveal further evidence of task contamination. Importantly, we find that for classification tasks with no possibility of task contamination, LLMs rarely demonstrate statistically significant improvements over simple majority baselines, in both zero and few-shot settings.
    </details>

<a id="resources"></a>
## 🧰 Resources

<a id="datasets"></a>
### 📊 Datasets

Many contamination detection work and contamination indices are done on *open-sourced pretraining corpora* against *interested existing benchmarks or datasets*. Below is a non-exhaustive list of these popular choices:

<table align="center">
  <tbody>
    <tr align="center">
      <th>Corpora</th>
      <th>Benchmarks</th>
    </tr>
    <tr>
      <td valign="top">
        <ul>
          <li><a href="https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/">WikiText-103</a></li>
          <li><a href="https://github.com/soskek/bookcorpus">BookCorpus</a></li>
          <li><a href="https://commoncrawl.org/blog/news-dataset-available">CCNews</a></li>
          <li><a href="https://arxiv.org/abs/2101.00027">ThePile</a></li>
          <li><a href="https://zenodo.org/records/3608135">Pushshift Reddit</a></li>
          <li><a href="https://github.com/togethercomputer/RedPajama-Data">RedPajama</a></li>
          <li><a href="https://arxiv.org/abs/2303.03915">ROOTS</a></li>
          <li>......</li>
        </ul>
      </td>
      <td valign="top">
        <ul>
          <li><a href="https://ai.google.com/research/NaturalQuestions">Natural Questions</a></li>
          <li><a href="https://rowanzellers.com/hellaswag/">HellaSwag</a></li>
          <li><a href="https://github.com/hendrycks/test">MMLU</a></li>
          <li><a href="https://github.com/sylinrl/TruthfulQA">TruthfulQA</a></li>
          <li><a href="https://github.com/google-research-datasets/boolean-questions">BoolQ</a></li>
          <li><a href="https://allenai.org/data/open-book-qa">OpenbookQA</a></li>
          <li><a href="https://cims.nyu.edu/~sbowman/multinli/">MNLI</a></li>
          <li><a href="https://leaderboard.allenai.org/physicaliqa/submissions/public">PIQA</a></li>
          <li>......</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

There are also some recent efforts to create datasets tailored to contamination detection, such as:

- [WikiMIA](https://huggingface.co/datasets/swj0419/WikiMIA)
- ......

<a id="tools"></a>
### 🛠️ Tools

In general, there are two types of data contamination tools: *contamination detector*, which identifies the contamination risk for a given test set w/ or w/o knowing the pretraining corpus, and *contamination index*, which documents the contamination risk of public benchmarks against foundation models or pretraining corpora and is utilized for a trustworthy comparison of foundation models.

Contamination index could be a product of contamination detectors. However, proprietary models do not disclose details about their pretraining corpora, invalidating most contamination detectors. Hence, relevant contamination statistics of these models can only be collected in the released paper or technical reports and are not reproducible in general.

A reference list of contamination detectors and contamination indices is as follows:

<table align="center">
  <tbody>
    <tr align="center">
      <th>Contamination Detector</th>
      <th>Contamination Index</th>
    </tr>
    <tr>
      <td valign="top">
        <ul>
          <li><a href="https://github.com/stanford-crfm/helm">HELM</a> [<a href="https://github.com/stanford-crfm/helm/blob/main/scripts/data_overlap/README.md">scripts</a>]</li>
          <li><a href="https://github.com/open-compass/opencompass">OpenCompass</a> [open-compass/opencompass#639]</li>
          <li><a href="https://github.com/liyucheng09/Contamination_Detector">Contamination Detector for LLMs Evaluation</a></li>
          <li>......</li>
        </ul>
      </td>
      <td valign="top">
        <ul>
          <li><a href="https://github.com/stanford-crfm/helm">HELM</a> [<a href="https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/static/contamination.yaml">docs</a>]</li>
          <li><a href="https://hitz-zentroa.github.io/lm-contamination/">LM Contamination Index</a></li>
          <li><a href="https://koala-index.erc.monash.edu/">Koala</a></li>
          <li>......</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

> [!Note]
> We explicitly mark the entrance with `[*]` if the mentioned tools possess multiple functions.

---

Some open-sourced evaluation tools provide the **decontamination** option, which leverages contamination detectors to eliminate compromised test instances during evaluation and delivers more trustworthy evaluation results. Exemplary evaluation tools of this kind are:

- [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) [[docs](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/decontamination.md)]
- [LLM Decontaminator](https://github.com/lm-sys/llm-decontaminator)
- ......

<a id="citation"></a>
## 🚩 Citation

Please cite our repo if find our work useful.

```bibtex
@misc{li2024awesome,
  author = {Yanyang Li},
  title = {Awesome Data Contamination},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/lyy1994/awesome-data-contamination}},
}
```

<a id="contribution"></a>
## 🎉 Contribution

We thank all contributors to this repo :heart:

<a href="https://github.com/lyy1994/awesome-data-contamination/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=lyy1994/awesome-data-contamination" />
</a>

There are cases where we miss important work in this field. We welcome opening PRs/issues to contribute to this repo and make this paper list more complete :blush:

<a id="acknowledgement"></a>
## 🤝 Acknowledgement

We referred to the template of [Knowledge Editing for LLMs Papers](https://github.com/zjunlp/KnowledgeEditingPapers) when building this paper list. Thanks to its authors for their impressive work!
