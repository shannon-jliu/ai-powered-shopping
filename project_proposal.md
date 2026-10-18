# Title of Project: AI-Powered Grocery Shopping Assistant

**Team Members:**  
- Olina Zheng (`oz36`)  
- Shannon Liu (`sjl356`)  
- Paige Yun (`phy6`)  

**AI Keywords:** Computer Vision, Natural Language Processing, Multimodal AI, Object Detection, Temporal Video Processing  

**Application Setting:** Egocentric grocery shopping assistance using a wearable first-person camera (GoPro) connected to a portable local compute unit, guided by written natural language recipes.  

---

### A Clear and Concise Description of What You Plan to Do

#### Section 1: Big Picture (1–2 Paragraphs)

Grocery shopping guided by recipes is cognitively demanding: shoppers frequently context-switch between reading ingredient lists on handheld devices, navigating crowded aisles, scanning dense product shelves, handling ingredient substitutions or package sizing, and mentally tracking which items have already been collected into the cart. We propose an AI-Powered Grocery Shopping Assistant that combines wearable first-person video capture with multimodal artificial intelligence to guide shoppers in real time. The system takes an arbitrary written natural language recipe as input, parses and canonicalizes the required ingredients into an actionable checklist, and continuously analyzes the shopper’s visual field of view. When required ingredients are detected on shelves or inspected by the user, the assistant verifies the match, updates the checklist state, and alerts the user to missing or upcoming items.

To ensure engineering feasibility and avoid superficial overpromising ("AI slop"), the project is strictly scoped around ingredient-level recognition and semantic checklist reconciliation rather than attempting fine-grained barcode or micro-text brand reading across hundreds of thousands of retail products. Because wearable cameras like GoPros lack onboard neural network accelerators, real-time edge execution will be accomplished by tethering the camera via a direct physical USB-C link (UVC webcam mode) or direct ad-hoc Wi-Fi to a portable laptop carried in the shopper's backpack or shopping cart. Local GPU/MPS acceleration on the laptop runs the detection pipeline in real time and delivers low-latency auditory notifications through Bluetooth earbuds, completely circumventing unstable grocery store Wi-Fi networks while enabling offline benchmark replay for reproducible grading.

#### Section 2: Core AI Methodology (2–3 Paragraphs)

The core AI challenge will be combining natural language processing and computer vision to connect a recipe to objects / items observed in the grocery store. First, the system will process the recipe and convert it into a structured list of required ingredients, converting the appropriate portion sizes and allergens provided by the user. This involves identifying ingredients from natural language and potentially handling different ways an ingredient can be described (for example, a recipe might call for "two medium yellow onions" while a grocery store product may simply be labeled "yellow onions.") The system will therefore need to determine whether a visually detected product corresponds to an ingredient in the recipe. To achieve this, we will implement an NLP pipeline that extracts ingredient entities into structured triplets (quantity, unit, name) and maps them to canonical grocery categories using embedding-based semantic similarity (e.g., Sentence-Transformers).

The second major component will be visual recognition from the GoPro's first-person video. Because the wearable GoPro cannot run deep learning models onboard, the camera will stream video directly via USB-C or local Wi-Fi to a portable laptop carried in the shopper's backpack or cart, which executes the models locally in real time. We will investigate computer vision models that can recognize grocery products or ingredient categories from video frames. Rather than requiring the system to identify every exact brand, our initial goal will be to identify ingredient-level categories that are relevant to the recipe. We will also investigate temporal information from video, since a product appearing in several consecutive frames should provide stronger evidence than a single blurry detection. The system will maintain a representation of which ingredients have been found and update the shopping list as the user moves through the store, providing immediate auditory feedback via Bluetooth earbuds.

We also plan to explore how different AI approaches affect the system's performance. For example, we can compare a general-purpose vision-language model (such as OWL-ViT or YOLO-World) with a more specialized object-recognition approach (such as a fine-tuned YOLO detector), and compare recognition using individual video frames against recognition using multiple consecutive frames via temporal voting or tracking. This will allow us to investigate the tradeoffs between accuracy, computational cost, inference latency (frames per second on our mobile laptop GPU), and robustness to real-world conditions such as occlusion, motion blur, different lighting, and visually similar products.

#### Section 3: Evaluation Plan and Metrics (2–3 Paragraphs)

Evaluating an end-to-end multimodal AI system presents unique challenges because errors can compound across stages—an extraction error in the recipe parser will cause the vision model to search for the wrong item, while visual false alarms can corrupt the user's checklist state. To determine whether our system is successful, we decouple our evaluation into three complementary levels: isolated NLP benchmarking, controlled computer vision and temporal video benchmarking, and an end-to-end human-in-the-loop user study. First, we will evaluate the accuracy of ingredient extraction from recipes in isolation against a curated benchmark of 50 diverse online recipes (from Recipe1M and Allrecipes) annotated with ground-truth ingredient names, quantities, and allergen flags. We will measure exact match (EM) and token-level F1 for entity extraction. To address semantic ambiguity—such as whether "Roma tomatoes" or "crushed canned tomatoes" correctly satisfy a recipe calling for "tomatoes"—we will evaluate our embedding-based canonicalization layer against human-annotated similarity judgments, measuring whether cosine similarity thresholds can reliably separate acceptable synonyms (e.g., scallions vs. green onions) from invalid substitutions.

For the computer vision component, evaluating continuous first-person video is notoriously difficult because dense per-frame bounding box annotation across extended shopping sessions is intractable, and single-frame metrics like mAP fail to capture real-time video behavior. We will address this through a two-pronged evaluation strategy. First, we will construct a labeled keyframe test set sampled at 1 FPS from egocentric GoPro footage recorded in real grocery stores and mock shelf environments, annotated with bounding boxes across both standard conditions and stress-test conditions (severe motion blur, partial occlusion, crowded shelves, and varied lighting). On these keyframes, we will compute standard object detection metrics: Precision, Recall, F1 score, and mAP at IoU thresholds of 0.5 and 0.5:0.95. Second, to evaluate video-level temporal performance, we will use time-interval event labeling: annotating the start and end timestamps during which target ingredients remain clearly visible in the shopper's interaction field of view. This enables us to evaluate temporal stability directly: measuring detection latency (how many seconds from entering view until confirmation), the reduction in false-positive "flicker" detections achieved by multi-frame temporal voting compared to a single-frame baseline, and the false positive rate per minute of walking footage. We will use this exact benchmark to conduct controlled ablations comparing closed-vocabulary YOLO detection against open-vocabulary vision-language detection (OWL-ViT), and single-frame against temporal windowing to systematically determine their relative merits.

Finally, we will evaluate the complete shopping assistant based on how successfully it helps a human user complete a shopping list in real-world scenarios. Per the course guidelines, we will recruit participants other than ourselves—specifically 4 to 6 Cornell students (including fellow CS 4701 peers)—to perform controlled grocery-shopping tasks. To calibrate our results against human baseline capabilities, participants will complete standardized shopping tasks under two counterbalanced experimental conditions: (A) manual shopping using a traditional written recipe list, and (B) shopping assisted by our wearable AI assistant. We will measure objective task performance metrics: the percentage of required ingredients successfully found (Task Recall), the number of incorrect items gathered (False Positive Rate), the total time required to complete the shopping task, and the number of times the user needs to manually consult the recipe or phone. To evaluate subjective cognitive burden and trust, participants will complete a standardized NASA-TLX survey measuring mental demand, effort, frustration, and confidence in the system's real-time recommendations. This multi-tiered evaluation ensures that we measure not only raw machine learning accuracy, but also the practical utility and robustness of the interactive AI assistant.

---

### Section 4: Project Timeline

Our project timeline spans data collection, modular implementation, iterative ablation testing, system updates, external user evaluation, and report writing. Key engineering tasks are parallelized across sub-teams, with clear sequential integration milestones and dedicated buffer periods to update the system based on empirical feedback:

| Target Date | Phase & Workflow Structure | Milestones & Deliverables | Team Member Responsibilities |
| :--- | :--- | :--- | :--- |
| **Sept 11** | **Milestone 2 Submission** | Submit formal project proposal to Canvas. | Entire team (Olina, Shannon, Paige) |
| **Sept 12 – Sept 25** | **Parallel Scaffolding & Initial Data Ingestion**<br>*(Parallel Work)* | • Validate GoPro USB-C/UVC 1080p frame capture to laptop at $\ge 20$ FPS.<br>• Implement rule/LLM-based recipe parsing and entity extraction module.<br>• Set up PyTorch inference harness for YOLOv8 and OWL-ViT baselines.<br>• Record initial egocentric video clips in Cornell campus stores (Bear Necessities) and local markets. | • **Olina:** Hardware streaming harness, GoPro capture protocol, initial shelf recordings.<br>• **Shannon:** Recipe NLP parsing pipeline, ingredient triplet extraction, allergen filters.<br>• **Paige:** CV baseline loader, GPU inference benchmarking on laptop, open-source model setup. |
| **Sept 26 – Oct 05** | **Component Integration & Sequential Pipeline**<br>*(Sequential Work)* | • Connect NLP checklist output to CV detection target prompt inputs.<br>• Implement multi-frame temporal voting filter to suppress single-frame false positives.<br>• Assemble 50-recipe NLP benchmark and annotate 1 FPS keyframe video test set. | • **Olina:** Video keyframe annotation & event timestamp labeling.<br>• **Shannon:** Semantic canonicalization layer (Sentence-Transformers) & checklist state machine.<br>• **Paige:** Multi-frame temporal voting filter and detector integration. |
| **Oct 06 – Oct 16** | **Initial Testing & System Updates (Iterative Refinement)**<br>*(Iterative Buffer)* | • **Initial Testing:** Run component-level benchmarks on NLP and CV pipelines.<br>• **System Updates:** Refine detection confidence thresholds and adjust temporal voting window based on motion blur failure modes observed in initial tests.<br>• Prepare live demo setup on a controlled mock-shelf testbench. | • **Olina:** Compile error logs, failure cases, and video benchmarking results.<br>• **Shannon:** Retune NLP mapping thresholds and state transition hysteresis.<br>• **Paige:** Optimize inference latency via model quantization (INT8/FP16) on laptop GPU. |
| **Mid October** | **Milestone 3 Submission** | **Status Report & Working Prototype Demo** submitted to course staff. | Entire team |
| **Oct 17 – Nov 05** | **End-to-End System & Audio Interface**<br>*(Parallel Work)* | • Implement real-time auditory notifications (Bluetooth earbud audio cues) for detected/missing items.<br>• Execute quantitative ablation studies: closed-set YOLO vs. open-vocabulary OWL-ViT; single-frame vs. temporal voting.<br>• Pilot dry-run shopping trials with team members to finalize user study protocol. | • **Olina:** User study protocol design, task lists, and NASA-TLX survey preparation.<br>• **Shannon:** Audio feedback synthesizer and edge streaming robustness.<br>• **Paige:** Finalize ablation experiment scripts, precision-recall curves, and latency metrics. |
| **Nov 06 – Nov 18** | **Human Evaluation & User Study Execution**<br>*(User Evaluation)* | • Recruit 4–6 external Cornell/CS 4701 students.<br>• Conduct counterbalanced A/B shopping trials (Manual vs. AI-Assisted).<br>• Collect task completion times, item accuracy, and NASA-TLX cognitive load scores. | • **Olina:** Lead participant study trials, record timing/behavioral data.<br>• **Shannon:** Conduct post-study interviews, qualitative feedback, and survey data aggregation.<br>• **Paige:** Monitor real-time system logs and verify detection event telemetry. |
| **Nov 19 – Dec 04** | **Final Analysis, Writing & Deliverables**<br>*(Sequential Work)* | • Perform statistical analysis on user study and ablation results.<br>• Draft comprehensive final technical report.<br>• Record, edit, and narrate project video demonstration.<br>• Clean, document, and publish reproducible code repository. | • **Olina:** Video demonstration recording/editing, user study figures and tables.<br>• **Shannon:** Final report drafting (Introduction, NLP methodology, Discussion & Ethics).<br>• **Paige:** Final report drafting (CV architecture, Ablation results, System optimization). |

---

### Section 5: Existing Resources

We do not need to reinvent the wheel; our project builds on established open-source models, software frameworks, and retail datasets, focusing our effort on multimodal integration, temporal egocentric video processing, and real-time interactive assistance.

#### Software Resources
* **Computer Vision & Tracking:** Ultralytics YOLOv8 / YOLO-World (`ultralytics`), Hugging Face `transformers` for open-vocabulary detection (OWL-ViT, Grounding DINO), and OpenCV for frame capture, pre-processing, and video rendering.
* **Natural Language Processing:** Hugging Face `transformers`, `sentence-transformers` (`all-MiniLM-L6-v2`) for semantic embedding canonicalization, and SpaCy for tokenization and grammatical dependency parsing.
* **Edge Execution & Audio Output:** ONNX Runtime and PyTorch (with Apple Silicon Metal Performance Shaders / CUDA acceleration) for local model execution; `pyttsx3` / local audio synthesis libraries for real-time auditory feedback through Bluetooth headphones.

#### Data Resources (Already Identified)
* **Retail Product Checkout (RPC) Dataset:** An existing large-scale benchmark containing 200 product categories across 83,000 images with varying shelf clutter, occlusion, and lighting, which we will use for baseline pretraining and initial detector validation.
* **SKU-110k Dataset:** An existing benchmark consisting of 11,762 dense retail shelf images with over 1.7 million labeled bounding boxes, utilized to benchmark detection under severe shelf crowding.
* **Recipe1M+ Dataset:** A corpus of over one million structured recipes and ingredient lists, which we will sample to extract standard cooking ingredient vocabularies and construct our 50-recipe evaluation benchmark.
* **Self-Generated Egocentric Video Dataset:** Because public retail datasets feature static top-down or frontal shelf photos rather than first-person moving video, we will generate our own targeted egocentric video corpus. This will consist of 10–15 first-person shopping sessions (1080p, 30 FPS) recorded on campus (e.g., Bear Necessities) and in local stores, with keyframes annotated for ingredient categories and event timestamps labeled for temporal evaluation.

#### Hardware Resources
* **Video Capture:** GoPro HERO action camera equipped with head/chest mounts for first-person egocentric capture.
* **Mobile Edge Compute:** High-performance laptop equipped with a mobile dedicated GPU (NVIDIA RTX) / Apple Silicon (M-series MPS) carried in a backpack/cart for local, offline real-time inference.
* **Connectivity & Feedback:** High-speed USB-C tether (UVC webcam protocol) and Bluetooth earbuds (AirPods) for audio cue delivery.
* **Offline Training Compute:** Desktop NVIDIA GPU workstation (RTX 3080/4090) and Google Colab Pro / Cornell computing resources for model fine-tuning and batch ablation evaluation.

---

### Section 6: Relation to Other Projects (Academic Integrity)

This project is not related to anything else our team members are working on. The work for this project is entirely disjoint from any prior or concurrent coursework, external research projects, or employment. All data collection, model development, and empirical evaluation will be conducted specifically for CS 4701 this semester.

---

### Section 7: References

1. **Minderer, M., et al. (2022).** *Simple Open-Vocabulary Object Detection with Vision Transformers (OWL-ViT).* European Conference on Computer Vision (ECCV).
2. **Cheng, T., et al. (2024).** *YOLO-World: Real-Time Open-Vocabulary Object Detection.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
3. **Wei, P., et al. (2019).** *RPC: A Large-Scale Retail Product Checkout Dataset.* arXiv:1901.07249.
4. **Goldman, E., et al. (2019).** *Precise Detection in Densely Packed Scenes (SKU-110K).* IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
5. **Salvador, A., et al. (2017).** *Learning Cross-modal Embeddings for Cooking Recipes and Food Images (Recipe1M).* IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
6. **Reimers, N., & Gurevych, I. (2019).** *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* Conference on Empirical Methods in Natural Language Processing (EMNLP).
7. **Hart, S. G., & Staveland, L. E. (1988).** *Development of NASA-TLX (Task Load Index): Results of Empirical and Theoretical Research.* Advances in Psychology.
