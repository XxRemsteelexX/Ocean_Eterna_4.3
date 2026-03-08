#!/usr/bin/env python3
"""generate synthetic test corpora at different scales for OE benchmarking.
produces realistic text content across multiple topics/domains."""

import json, random, os, time, sys

TOPICS = {
    "machine_learning": [
        "Neural networks use layers of interconnected nodes to learn patterns from data. Each node applies a weighted sum and activation function.",
        "Gradient descent optimizes model parameters by computing the gradient of the loss function and updating weights in the direction that minimizes error.",
        "Convolutional neural networks excel at image recognition by applying learned filters across spatial dimensions of input data.",
        "Transfer learning allows pretrained models to be fine-tuned on smaller datasets, reducing training time and data requirements significantly.",
        "Recurrent neural networks process sequential data by maintaining hidden state across time steps, enabling memory of previous inputs.",
        "Batch normalization stabilizes training by normalizing layer inputs, allowing higher learning rates and reducing sensitivity to initialization.",
        "Dropout regularization randomly disables neurons during training to prevent co-adaptation and reduce overfitting on limited data.",
        "Attention mechanisms allow models to focus on relevant parts of the input sequence, weighting importance dynamically per query.",
        "Generative adversarial networks pit a generator against a discriminator in a minimax game to produce realistic synthetic data.",
        "Reinforcement learning agents maximize cumulative reward through trial and error, balancing exploration of new actions with exploitation of known good strategies.",
    ],
    "software_engineering": [
        "Microservices architecture decomposes applications into small, independently deployable services communicating via APIs.",
        "Continuous integration automatically builds and tests code changes, catching integration errors early in the development cycle.",
        "Database indexing improves query performance by creating data structures that allow the database engine to find rows without scanning entire tables.",
        "Load balancing distributes incoming network traffic across multiple servers to ensure no single server bears too much demand.",
        "Container orchestration platforms like Kubernetes manage deployment, scaling, and networking of containerized applications.",
        "API rate limiting protects services from abuse by restricting the number of requests a client can make within a time window.",
        "Event-driven architecture uses message queues to decouple producers and consumers, enabling asynchronous processing and better scalability.",
        "Code review practices improve software quality by having peers examine changes for bugs, design issues, and maintainability concerns.",
        "Infrastructure as code manages and provisions computing resources through machine-readable configuration files rather than manual processes.",
        "Test-driven development writes tests before implementation, ensuring code meets requirements and facilitating safe refactoring.",
    ],
    "history": [
        "The printing press, invented by Gutenberg around 1440, revolutionized information dissemination and played a key role in the Renaissance and Reformation.",
        "The Industrial Revolution beginning in the late 18th century transformed economies from agrarian to industrial, driven by steam power and mechanization.",
        "Ancient Egyptian civilization developed along the Nile River, creating sophisticated writing systems, architectural achievements, and governance structures.",
        "The Roman Republic's transition to Empire under Augustus in 27 BCE established a political model that influenced Western governance for centuries.",
        "The Silk Road connected East and West for centuries, facilitating trade in goods, ideas, religions, and technologies across vast distances.",
        "The French Revolution of 1789 overthrew the monarchy and established principles of citizenship and inalienable rights that shaped modern democracy.",
        "The Renaissance period saw a revival of classical learning and art, centered in Italian city-states and spreading across Europe from the 14th to 17th centuries.",
        "World War I introduced trench warfare, chemical weapons, and mechanized combat, fundamentally changing military strategy and international relations.",
        "The Cold War defined international politics from 1947 to 1991, with the United States and Soviet Union competing for global influence without direct military conflict.",
        "The Age of Exploration in the 15th and 16th centuries led European nations to establish colonies worldwide, reshaping global trade and demographics.",
    ],
    "science": [
        "Quantum mechanics describes the behavior of matter and energy at the atomic and subatomic level, where particles exhibit both wave and particle properties.",
        "DNA's double helix structure, discovered by Watson and Crick in 1953, revealed the mechanism of genetic inheritance and opened the field of molecular biology.",
        "Plate tectonics explains the movement of Earth's lithospheric plates, driving earthquakes, volcanic activity, and the formation of mountain ranges.",
        "The theory of general relativity describes gravity as the curvature of spacetime caused by mass and energy, predicting phenomena like gravitational waves.",
        "Photosynthesis converts light energy into chemical energy, using carbon dioxide and water to produce glucose and oxygen in plant cells.",
        "Evolution by natural selection, proposed by Darwin, explains how species change over time through differential survival and reproduction of organisms with favorable traits.",
        "The standard model of particle physics describes fundamental particles and forces, including quarks, leptons, and gauge bosons mediating electromagnetic, weak, and strong forces.",
        "Climate change results from increased greenhouse gas concentrations trapping heat in the atmosphere, leading to rising global temperatures and weather pattern shifts.",
        "Black holes form when massive stars collapse, creating regions of spacetime where gravity is so strong that nothing, not even light, can escape.",
        "CRISPR-Cas9 gene editing technology allows precise modification of DNA sequences, with applications in medicine, agriculture, and biological research.",
    ],
    "business": [
        "Supply chain management coordinates the flow of goods from raw materials to final products, optimizing cost, speed, and quality across multiple stakeholders.",
        "Venture capital funds invest in early-stage companies with high growth potential, providing capital in exchange for equity and typically seeking returns through IPO or acquisition.",
        "Market segmentation divides a broad consumer market into subgroups based on shared characteristics, enabling targeted marketing strategies.",
        "Agile project management emphasizes iterative development, cross-functional teams, and continuous feedback to deliver value incrementally.",
        "Mergers and acquisitions reshape industries by combining companies to achieve economies of scale, market expansion, or technology acquisition.",
        "Customer lifetime value estimates the total revenue a business can expect from a single customer account throughout the entire business relationship.",
        "Lean manufacturing eliminates waste in production processes, focusing on value-adding activities and continuous improvement.",
        "Disruptive innovation introduces simpler, more affordable products that initially serve overlooked market segments before displacing established competitors.",
        "Financial statements including balance sheets, income statements, and cash flow statements provide a comprehensive view of a company's financial health.",
        "Strategic planning involves defining an organization's direction and making decisions on allocating resources to pursue this strategy over a multi-year horizon.",
    ],
    "cooking": [
        "The Maillard reaction occurs when amino acids and reducing sugars react under heat, creating complex flavors and brown color in cooked foods.",
        "Fermentation transforms food through microbial action, producing products like bread, cheese, yogurt, beer, and kimchi with enhanced flavors and preservation.",
        "Emulsification combines two immiscible liquids like oil and water using emulsifiers such as egg yolk lecithin to create stable mixtures like mayonnaise.",
        "Sous vide cooking involves sealing food in airtight bags and cooking in precisely controlled water baths, achieving consistent results impossible with traditional methods.",
        "Knife skills are fundamental to professional cooking, with techniques like julienne, brunoise, and chiffonade ensuring uniform cooking and elegant presentation.",
        "Bread baking relies on gluten development through kneading, creating elastic networks that trap carbon dioxide from yeast fermentation to produce rise and texture.",
        "Caramelization occurs when sugars are heated above their melting point, undergoing pyrolysis to produce hundreds of flavor and color compounds.",
        "Stock making extracts collagen, minerals, and flavor compounds from bones and aromatics through long, gentle simmering, forming the foundation of sauces and soups.",
        "Spice blending combines whole and ground spices in specific ratios, with toasting and grinding releasing essential oils for maximum flavor impact.",
        "Food preservation techniques including canning, smoking, curing, and pickling extend shelf life by inhibiting microbial growth through various chemical and physical means.",
    ],
    "philosophy": [
        "Existentialism holds that individual existence precedes essence, meaning humans define themselves through actions rather than inherent nature or predetermined purpose.",
        "Utilitarianism evaluates the morality of actions based on their consequences, seeking to maximize overall happiness or well-being for the greatest number.",
        "Epistemology examines the nature and scope of knowledge, questioning what we can know, how we know it, and the limits of human understanding.",
        "The mind-body problem asks how mental states relate to physical states, with dualism, physicalism, and functionalism offering competing explanations.",
        "Social contract theory proposes that political authority derives from an agreement among individuals to form a society, surrendering some freedoms for collective benefits.",
        "Virtue ethics focuses on character development rather than rules or consequences, asking what kind of person one should be rather than what actions are right.",
        "Phenomenology studies the structures of subjective experience, examining how objects and events appear to consciousness from the first-person perspective.",
        "Stoicism teaches that virtue is the highest good and that we should focus on what we can control while accepting what we cannot with equanimity.",
        "Pragmatism evaluates ideas by their practical consequences and usefulness, rejecting abstract theorizing in favor of experimentation and real-world application.",
        "Dialectical reasoning examines contradictions between opposing ideas to arrive at higher truths, as developed by Hegel and adapted by Marx.",
    ],
    "technology": [
        "Blockchain technology creates distributed, immutable ledgers that record transactions across many computers, eliminating the need for trusted intermediaries.",
        "Edge computing processes data closer to its source rather than in centralized data centers, reducing latency and bandwidth usage for IoT applications.",
        "Natural language processing enables computers to understand, interpret, and generate human language, powering applications from translation to chatbots.",
        "WebAssembly provides a portable binary instruction format for executing code in web browsers at near-native speed, extending beyond JavaScript's capabilities.",
        "Zero-trust security architecture requires strict verification for every person and device attempting to access resources, regardless of network location.",
        "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to perform computations exponentially faster for certain problem classes.",
        "5G networks provide higher bandwidth, lower latency, and greater device density than previous generations, enabling applications like autonomous vehicles and remote surgery.",
        "Federated learning trains machine learning models across decentralized devices holding local data, preserving privacy while still improving global models.",
        "Digital twins create virtual replicas of physical objects or systems, enabling simulation, monitoring, and optimization without affecting the real-world counterpart.",
        "Homomorphic encryption allows computation on encrypted data without decrypting it, enabling secure cloud processing of sensitive information.",
    ],
}

def generate_document(doc_id, topic, paragraph_count=None):
    """generate a realistic document with multiple paragraphs on a given topic."""
    if paragraph_count is None:
        paragraph_count = random.randint(2, 8)

    topic_sentences = TOPICS[topic]
    paragraphs = []
    for _ in range(paragraph_count):
        # each paragraph: 2-5 sentences from the topic, with some variation
        num_sentences = random.randint(2, 5)
        selected = random.sample(topic_sentences, min(num_sentences, len(topic_sentences)))
        # add some unique detail to prevent exact duplicates
        detail = f" Document {doc_id} provides additional context on this subject within the {topic.replace('_', ' ')} domain."
        paragraphs.append(" ".join(selected) + detail)

    title = f"{topic.replace('_', ' ').title()} - Document {doc_id}"
    content = f"# {title}\n\n" + "\n\n".join(paragraphs)
    return {"title": title, "content": content, "topic": topic, "doc_id": doc_id}


def generate_corpus(size, output_dir):
    """generate a corpus of documents and save as files ready for OE ingestion."""
    os.makedirs(output_dir, exist_ok=True)
    topics = list(TOPICS.keys())
    docs = []

    print(f"generating {size} documents...")
    for i in range(size):
        topic = topics[i % len(topics)]
        doc = generate_document(i, topic)
        docs.append(doc)

    # save as individual text files for ingestion testing
    for doc in docs:
        filepath = os.path.join(output_dir, f"doc_{doc['doc_id']:06d}.txt")
        with open(filepath, "w") as f:
            f.write(doc["content"])

    # save metadata
    meta_path = os.path.join(output_dir, "corpus_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "total_docs": size,
            "topics": list(set(d["topic"] for d in docs)),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)

    total_chars = sum(len(d["content"]) for d in docs)
    print(f"  -> {size} docs, {total_chars:,} chars, {total_chars // 4:,} est. tokens")
    print(f"  -> saved to {output_dir}")
    return docs


if __name__ == "__main__":
    base = os.path.expanduser("~/oe-benchmark-test/test_data")

    sizes = {
        "small": 100,
        "medium": 1000,
        "large": 5000,
        "stress": 10000,
    }

    for name, count in sizes.items():
        print(f"\n=== {name.upper()} corpus ({count} docs) ===")
        generate_corpus(count, os.path.join(base, name))

    print("\ndone. all corpora generated.")
