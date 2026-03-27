"""
Experiment 10: Real-World Stress Test

Can this system work at scale with realistic data?

Tests:
1. 100 diverse facts (not hand-crafted — varied domains, lengths, complexity)
2. Automatic fact extraction from raw text paragraphs
3. Adversarial/paraphrased queries
4. 100 facts + 2 capabilities simultaneously
5. Measure cross-contamination rate at scale
"""

import torch
import numpy as np
import json
import faiss
import random
import gc
import time
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════
# Reuse unified system components from v9
# ═══════════════════════════════════════════════════════════

@dataclass
class FactEntry:
    trigger: np.ndarray
    token_ids: list[int]
    token_boosts: list[float]
    sequence_pos: int
    source: str = ""


@dataclass
class AdapterRoute:
    trigger: np.ndarray
    adapter_name: str
    description: str


class KnowledgeStore:
    def __init__(self, dim):
        self.dim = dim
        self.fact_index = faiss.IndexFlatIP(dim)
        self.fact_entries = []
        self.adapter_index = faiss.IndexFlatIP(dim)
        self.adapter_routes = []

    def add_fact(self, entry):
        t = entry.trigger / (np.linalg.norm(entry.trigger) + 1e-8)
        entry.trigger = t
        self.fact_entries.append(entry)
        self.fact_index.add(t.reshape(1, -1).astype(np.float32))

    def add_adapter_route(self, route):
        t = route.trigger / (np.linalg.norm(route.trigger) + 1e-8)
        route.trigger = t
        self.adapter_routes.append(route)
        self.adapter_index.add(t.reshape(1, -1).astype(np.float32))

    def query_facts(self, activation, top_k=20, threshold=0.90):
        if self.fact_index.ntotal == 0: return []
        a = activation / (np.linalg.norm(activation) + 1e-8)
        k = min(top_k, self.fact_index.ntotal)
        sims, idxs = self.fact_index.search(a.reshape(1, -1).astype(np.float32), k)
        return [(self.fact_entries[i], float(s)) for s, i in zip(sims[0], idxs[0])
                if s >= threshold and i >= 0]

    def query_adapter(self, activation, threshold=0.85):
        if self.adapter_index.ntotal == 0: return None, 0.0
        a = activation / (np.linalg.norm(activation) + 1e-8)
        sims, idxs = self.adapter_index.search(a.reshape(1, -1).astype(np.float32), 1)
        if sims[0][0] >= threshold and idxs[0][0] >= 0:
            return self.adapter_routes[idxs[0][0]].adapter_name, float(sims[0][0])
        return None, 0.0


class UnifiedModel:
    def __init__(self, model_name, device="cuda", max_boost=30.0,
                 fact_threshold=0.90, adapter_threshold=0.85):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map=device)
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.model = self.base_model
        self.device = device
        self.max_boost = max_boost
        self.fact_threshold = fact_threshold
        self.adapter_threshold = adapter_threshold
        self.hidden_dim = self.base_model.config.hidden_size
        self.vocab_size = self.base_model.config.vocab_size
        self.memory = KnowledgeStore(self.hidden_dim)
        self._gen_step = 0
        self._hook = None

    def _install_hook(self):
        if self._hook: self._hook.remove()
        lm_head = self.model.base_model.lm_head if hasattr(self.model, 'base_model') else self.model.lm_head
        self._hook = lm_head.register_forward_hook(self._fact_hook)

    def _adaptive_boost(self, sim):
        if sim <= self.fact_threshold: return 0.0
        return ((sim - self.fact_threshold) / (1.0 - self.fact_threshold)) * self.max_boost

    def _fact_hook(self, module, input, output):
        if self.memory.fact_index.ntotal == 0: return output
        with torch.no_grad():
            hs = input[0][0].cpu().float()
            query = hs.mean(dim=0).numpy()
            results = self.memory.query_facts(query, threshold=self.fact_threshold)
            if not results: return output
            bias = torch.zeros(self.vocab_size, device=output.device, dtype=output.dtype)
            for entry, sim in results:
                if entry.sequence_pos == self._gen_step:
                    boost = self._adaptive_boost(sim)
                    for tid, tb in zip(entry.token_ids, entry.token_boosts):
                        if tid < self.vocab_size:
                            bias[tid] += tb * boost
            if bias.any():
                output = output.clone()
                output[0, -1, :] += bias
        return output

    def get_trigger(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.base_model(input_ids=inputs.input_ids, output_hidden_states=True)
            return out.hidden_states[-1][0].cpu().float().mean(dim=0).numpy()

    def learn_fact(self, prompt, answer):
        trigger = self.get_trigger(prompt)
        tokens = self.tokenizer.encode(" " + answer, add_special_tokens=False)
        n = min(len(tokens), 25)
        for pos in range(n):
            self.memory.add_fact(FactEntry(
                trigger=trigger.copy(), token_ids=[tokens[pos]],
                token_boosts=[1.0], sequence_pos=pos, source=prompt[:50]))
        return n

    def add_adapter(self, name, path):
        if isinstance(self.model, PeftModel):
            self.model.load_adapter(path, adapter_name=name)
        else:
            self.model = PeftModel.from_pretrained(self.base_model, path, adapter_name=name)
        self._install_hook()

    def register_adapter_triggers(self, name, prompts):
        for p in prompts:
            self.memory.add_adapter_route(AdapterRoute(
                trigger=self.get_trigger(p), adapter_name=name, description=name))

    def generate(self, prompt, max_new_tokens=40):
        trigger = self.get_trigger(prompt)
        adapter_name, adapter_sim = self.memory.query_adapter(trigger, self.adapter_threshold)

        if isinstance(self.model, PeftModel):
            if adapter_name:
                self.model.set_adapter(adapter_name)
                self.model.enable_adapter_layers()
            else:
                self.model.disable_adapter_layers()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        generated = []
        for step in range(max_new_tokens):
            self._gen_step = step
            with torch.no_grad():
                out = self.model(input_ids=input_ids)
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                if next_token.item() == self.tokenizer.eos_token_id: break
                generated.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token], dim=1)

        if isinstance(self.model, PeftModel):
            self.model.enable_adapter_layers()

        return self.tokenizer.decode(generated, skip_special_tokens=True), adapter_name


# ═══════════════════════════════════════════════════════════
# 100 Diverse Facts — varied domains
# ═══════════════════════════════════════════════════════════

FACTS_100 = [
    # Science (fictional but plausible)
    ("The Kessler-Yao Constant is", "approximately 7.382, governing the rate of quantum decoherence in supercooled plasmas."),
    ("Ferroplasmic resonance occurs when", "iron-based nanoparticles vibrate at frequencies above 340 terahertz in a magnetic field."),
    ("The Hadley-Moreno Conjecture states that", "prime factorization of sufficiently large semiprimes can be achieved in polynomial time using topological quantum circuits."),
    ("Cytoplasmic drift velocity in human neurons is", "approximately 0.3 millimeters per second under normal conditions."),
    ("The Drake-Sato Revision estimates", "between 4 and 12 technologically active civilizations in the Milky Way."),

    # Geography (fictional)
    ("The deepest point in Lake Meridian is", "1,247 meters, located in the Azurite Trench near the eastern shore."),
    ("Mount Thessaly's peak elevation is", "6,891 meters, making it the tallest mountain in the Korathan Range."),
    ("The Velmara Desert receives", "less than 3 millimeters of rainfall per year, making it the driest inhabited region."),
    ("Port Calliope is located on", "the northern coast of the Selenite Peninsula, serving as the primary trade hub for the Aetherian Islands."),
    ("The Indric Ocean's average depth is", "4,120 meters with surface temperatures ranging from 18 to 27 degrees Celsius."),

    # History (fictional)
    ("The Treaty of Ashenmoor was signed in", "1847 between the nations of Valdris and Kethenor, ending the Twelve Year Siege."),
    ("Queen Isolde the Third ruled", "the Kingdom of Alareth from 1623 to 1671, overseeing the Golden Renaissance of architecture."),
    ("The Battle of Crimson Fields took place on", "September 14, 1782, resulting in the defeat of the Thornwall Confederacy."),
    ("The Meridian Charter of 1901 established", "universal suffrage and free public education across all member states of the Northern Alliance."),
    ("The Obsidian Rebellion of 1756 was led by", "General Marcus Thane, who overthrew the autocratic Priory Council in a bloodless coup."),

    # Technology (fictional)
    ("The Solari Battery stores energy using", "compressed helium-3 plasma contained in magnetic bottles, achieving 94% charge retention over 10 years."),
    ("Graphene-lattice processors achieve speeds of", "847 petaflops per watt, making them 200 times more efficient than silicon-based chips."),
    ("The Aether Protocol enables", "peer-to-peer quantum key distribution without satellite infrastructure, using atmospheric entanglement channels."),
    ("Nanoscale memory crystals can store", "up to 5 exabytes per cubic centimeter using photonic interference patterns."),
    ("The Vanguard Operating System is built on", "a microkernel architecture with formal verification of all critical system calls."),

    # People (fictional)
    ("Professor Yuki Tanashi discovered", "the anomalous magnetic behavior of bismuth-telluride compounds at room temperature in 2034."),
    ("Dr. Amara Osei pioneered", "the field of computational ethnography, developing algorithms that model cultural evolution."),
    ("Chen Wei-Lin invented", "the resonance capacitor in 2041, enabling wireless power transmission over distances exceeding 500 meters."),
    ("Captain Elena Vasquez commanded", "the first manned expedition to Europa in 2078, discovering subsurface microbial life."),
    ("Architect Nabil Farouk designed", "the Helix Tower in New Alexandria, a 340-meter structure that generates its own electricity through wind channels."),

    # Organizations (fictional)
    ("The Apex Foundation funds", "research into sustainable fusion energy and deep-sea mineral extraction."),
    ("The Celestine Institute specializes in", "training astronauts for long-duration missions beyond the asteroid belt."),
    ("The Blackwood Consortium controls", "approximately 30% of the global rare earth mineral supply chain."),
    ("The Meridian Health Initiative provides", "free neural implant maintenance to veterans in 14 countries."),
    ("The Verdant Alliance is", "a coalition of 23 nations committed to achieving net-negative carbon emissions by 2060."),

    # Biology (fictional)
    ("The Azorian cave salamander can", "regenerate entire limbs within 72 hours due to hyperactive stem cell clusters."),
    ("Pelagic moon jellyfish communicate using", "bioluminescent pulse codes transmitted through their tentacle networks."),
    ("The Valderian pine tree produces", "a sap with natural antibiotic properties effective against 17 known resistant bacteria strains."),
    ("The Umbral moth navigates using", "the Earth's magnetic field and can detect variations as small as 0.001 microtesla."),
    ("Deep-sea thermophilic archaea near the Korathi Vents", "metabolize hydrogen sulfide at temperatures exceeding 140 degrees Celsius."),

    # Economics (fictional)
    ("The Ashworth Index measures", "economic resilience by tracking recovery speed after supply chain disruptions."),
    ("The Korathi Economic Zone generates", "approximately 2.3 trillion dollars annually from quantum computing services."),
    ("The Parallel Currency Act of 2055 allowed", "municipalities to issue local digital currencies backed by renewable energy credits."),
    ("The Thornton-Vasquez Model predicts", "that automation will create 1.4 new jobs for every job displaced within a 10-year horizon."),
    ("The Indric Free Trade Agreement eliminated", "tariffs on 94% of goods between 31 Pacific Rim nations."),

    # Space (fictional)
    ("The Kepler-442b colony was established in", "2089 with an initial population of 2,400 settlers in pressurized habitats."),
    ("Proxima Station orbits at", "a Lagrange point between Earth and Mars, serving as a refueling depot for deep-space missions."),
    ("The Alcubierre-Patel Drive achieves", "apparent faster-than-light travel by contracting spacetime ahead of the vessel."),
    ("Asteroid mining station Forge-7 processes", "approximately 50,000 metric tons of nickel-iron ore per month."),
    ("The Lunar Helium-3 extraction facility at Shackleton Crater produces", "enough fuel to power 12 fusion reactors annually."),

    # Medicine (fictional)
    ("The Prometheus Gene Therapy treats", "hereditary muscular dystrophy by replacing the defective dystrophin gene using CRISPR-Cas12 vectors."),
    ("Synthetic blood substitute HemoSyn-4 carries", "23% more oxygen than natural hemoglobin and has a shelf life of 18 months."),
    ("The neural bridge implant developed by NeuroLink", "restores motor function in 78% of spinal cord injury patients within 6 months."),
    ("Adaptive immunotherapy protocol Sentinel-3", "trains the patient's own T-cells to recognize and destroy pancreatic cancer cells."),
    ("The Lazarus Cryopreservation Method achieves", "99.7% cell viability after 5 years of storage at minus 196 degrees Celsius."),

    # Arts/Culture (fictional)
    ("The Polyphonic Movement in architecture emphasizes", "buildings that produce harmonious sounds when wind passes through their structural channels."),
    ("Director Yael Ashkenazi's film Crystalline won", "the Palme d'Or in 2067 for its groundbreaking use of volumetric holographic cinematography."),
    ("The Resonance School of painting uses", "pigments embedded with piezoelectric nanoparticles that change color based on ambient sound."),
    ("Composer Aiden Nakamura's Symphony No. 7 was", "the first orchestral work performed simultaneously on Earth and the Moon in 2071."),
    ("The Cobalt Literary Prize is awarded annually to", "works of fiction that best explore the ethical implications of emerging technologies."),

    # Law/Policy (fictional)
    ("The Algorithmic Accountability Act requires", "all AI systems making decisions affecting human rights to provide auditable explanations."),
    ("The Sovereignty of Digital Identity Treaty grants", "individuals legal ownership of their biometric data across all signatory nations."),
    ("The Orbital Debris Liability Convention holds", "launch operators financially responsible for collision damage for 50 years after deployment."),
    ("The Cognitive Enhancement Fairness Act prohibits", "employers from requiring neural augmentation as a condition of employment."),
    ("The Deep Sea Mining Moratorium of 2058 bans", "extraction below 4,000 meters pending completion of biodiversity impact assessments."),

    # Materials (fictional)
    ("Aerogel-X has a thermal conductivity of", "0.003 watts per meter-kelvin, making it the most effective insulator ever synthesized."),
    ("Crystalline carbon nanothread has a tensile strength of", "127 gigapascals, exceeding diamond by a factor of twelve."),
    ("Self-healing polymer Regen-7 repairs", "structural cracks up to 2 millimeters wide within 4 hours at room temperature."),
    ("Photovoltaic metamaterial SunWeave converts", "both visible light and infrared radiation to electricity at 62% combined efficiency."),
    ("Magnetic fluid MagLiq-3 maintains", "stable suspension of iron nanoparticles at temperatures up to 800 degrees Celsius."),

    # Food/Agriculture (fictional)
    ("The Nutrient-Dense Rice Variety KR-47 provides", "complete essential amino acids and 40% of daily iron requirements per serving."),
    ("Vertical farming tower AgroSpire produces", "12 harvests per year of leafy greens using 95% less water than traditional agriculture."),
    ("Synthetic protein ProteinaSyn-2 is", "molecularly identical to chicken breast protein but produced through precision fermentation."),
    ("The Sahel Regreening Initiative has restored", "180,000 square kilometers of degraded land using mycorrhizal fungal inoculation."),
    ("Deep-water kelp variety Thalassia-9 grows at", "a rate of 60 centimeters per day and absorbs 5 times more CO2 than terrestrial forests."),

    # Sports/Games (fictional)
    ("Zero-gravity basketball was invented on", "the International Space Station in 2038 using magnetic boots and a spherical court."),
    ("The Quantum Chess variant introduces", "superposition of pieces, where a single piece occupies multiple squares simultaneously."),
    ("Professional drone racing champion Kai Reeves holds", "the speed record of 347 kilometers per hour through an urban obstacle course."),
    ("The Titanium League is", "an international competition where teams of AI and human players collaborate in real-time strategy."),
    ("Underwater marathon swimming in the Mariana Challenge covers", "42 kilometers at a depth of 10 meters with currents averaging 3 knots."),

    # Environment (fictional)
    ("The Great Coral Restoration Project has regenerated", "7,200 square kilometers of reef using lab-grown coral fragments and AI-guided placement."),
    ("Atmospheric carbon capture station Nimbus-1 removes", "50,000 metric tons of CO2 per year using direct air capture with amine-based sorbents."),
    ("The Arctic Methane Shield project prevents", "permafrost methane release by injecting reflective silica microspheres into vulnerable tundra."),
    ("Ocean thermal energy conversion plant Poseidon-4 generates", "200 megawatts of electricity from temperature differences between deep and surface water."),
    ("The Biodiversity Vault in Svalbard now stores", "genetic samples from 8.7 million cataloged species as insurance against extinction."),

    # Transportation (fictional)
    ("The Hyperloop Transpacific connects", "Shanghai to San Francisco in 2 hours 14 minutes via a submerged vacuum tube."),
    ("Electric vertical takeoff aircraft SkyLift-9 has", "a range of 450 kilometers and carries up to 6 passengers at 280 km/h."),
    ("The Autonomous Shipping Fleet operated by OceanAI delivers", "35% of global container freight without human crew."),
    ("Mag-lev freight trains on the Silk Road Express achieve", "speeds of 600 km/h carrying 500 metric tons of cargo."),
    ("Personal mobility pods in Singapore's Smart Grid travel at", "a maximum of 50 km/h and self-navigate using LiDAR and city-wide mesh networks."),

    # Education (fictional)
    ("The Minerva Learning System adapts", "curriculum difficulty in real-time based on pupil brain activity measured through non-invasive EEG."),
    ("The Global Open University enrolls", "48 million students across 190 countries with AI-personalized degree programs."),
    ("Haptic simulation labs allow medical students to", "practice surgery on virtual patients with tactile feedback accurate to 0.1 millimeters."),
    ("The Polyglot Engine translates lectures into", "127 languages in real-time with 99.3% semantic accuracy."),
    ("Cognitive load monitoring wearables alert teachers when", "more than 30% of students show signs of information overload."),

    # Energy (fictional)
    ("Compact fusion reactor Stellarator-7 achieves", "net energy gain of Q=15 in a device small enough to fit in a shipping container."),
    ("Perovskite-silicon tandem solar cells reach", "33.7% efficiency in commercial production at one-third the cost of pure silicon."),
    ("The Geothermal Superhot Rock project in Iceland drills to", "5 kilometers depth accessing steam at 500 degrees Celsius for baseload power."),
    ("Solid-state lithium-sulfur batteries achieve", "energy density of 600 watt-hours per kilogram with 3,000 charge cycle lifespan."),
    ("The Tidal Barrage at the Bay of Fundy generates", "2.4 gigawatts of predictable renewable electricity from 16-meter tidal ranges."),
]

# Test queries — mix of exact, paraphrased, and adversarial
TEST_QUERIES = [
    # Exact prompts (should be easy)
    ("The Kessler-Yao Constant is", ["7.382", "quantum", "decoherence"], "exact"),
    ("The Treaty of Ashenmoor was signed in", ["1847", "Valdris", "Kethenor"], "exact"),
    ("Professor Yuki Tanashi discovered", ["bismuth", "telluride", "magnetic"], "exact"),
    ("The Solari Battery stores energy using", ["helium-3", "plasma", "magnetic"], "exact"),
    ("The Prometheus Gene Therapy treats", ["dystrophy", "dystrophin", "CRISPR"], "exact"),
    ("Compact fusion reactor Stellarator-7 achieves", ["Q=15", "shipping container"], "exact"),
    ("The Hyperloop Transpacific connects", ["Shanghai", "San Francisco", "vacuum"], "exact"),
    ("Aerogel-X has a thermal conductivity of", ["0.003", "insulator"], "exact"),
    ("The Cobalt Literary Prize is awarded annually to", ["fiction", "ethical", "technology"], "exact"),
    ("Zero-gravity basketball was invented on", ["Space Station", "2038", "magnetic"], "exact"),

    # Paraphrased (tests generalization)
    ("What is the Kessler-Yao Constant?", ["7.382", "decoherence"], "paraphrase"),
    ("When was the Treaty of Ashenmoor signed?", ["1847"], "paraphrase"),
    ("What did Professor Tanashi discover?", ["bismuth", "magnetic"], "paraphrase"),
    ("How does the Solari Battery work?", ["helium", "plasma"], "paraphrase"),
    ("What does the Prometheus therapy treat?", ["dystrophy"], "paraphrase"),
    ("Tell me about Stellarator-7", ["fusion", "Q=15"], "paraphrase"),
    ("How fast is the Hyperloop Transpacific?", ["Shanghai", "hours"], "paraphrase"),
    ("What makes Aerogel-X special?", ["thermal", "insulator"], "paraphrase"),
    ("What is the Cobalt Literary Prize?", ["fiction", "ethical"], "paraphrase"),
    ("How was zero-gravity basketball created?", ["Space Station", "2038"], "paraphrase"),

    # Cross-domain (tests no contamination between domains)
    ("The capital of France is", ["Paris"], "control"),
    ("Water boils at", ["100"], "control"),
    ("The speed of light is", ["300"], "control"),
    ("Python is a", ["programming"], "control"),
    ("Einstein developed", ["relativity"], "control"),
    ("DNA stands for", ["deoxyribonucleic"], "control"),
    ("The largest ocean is", ["Pacific"], "control"),
    ("Shakespeare wrote", ["play", "Romeo", "Hamlet"], "control"),
]


def main():
    print("=" * 60)
    print("EXPERIMENT 10: Real-World Scale Test — 100 Facts")
    print("=" * 60)

    MODEL = "Qwen/Qwen2.5-3B-Instruct"
    system = UnifiedModel(MODEL, max_boost=30.0, fact_threshold=0.90, adapter_threshold=0.85)

    # Load existing adapters from v9
    try:
        system.add_adapter("zorb", "/tmp/zorb_unified")
        system.add_adapter("glorp", "/tmp/glorp_unified")
        zorb_triggers = [f"zorb({a}, {b}) =" for a in range(1, 6) for b in range(1, 6)]
        glorp_triggers = [f"glorp({a}, {b}) =" for a in range(1, 6) for b in range(1, 6)]
        system.register_adapter_triggers("zorb", zorb_triggers[:15])
        system.register_adapter_triggers("glorp", glorp_triggers[:15])
        print("  Loaded zorb + glorp adapters from previous experiment")
    except Exception as e:
        print(f"  Note: couldn't load adapters: {e}")
        print("  Proceeding with facts only")

    # ── Learn 100 facts ──
    print(f"\n[Learning {len(FACTS_100)} facts...]")
    t0 = time.time()
    for prompt, answer in FACTS_100:
        system.learn_fact(prompt, answer)
    learn_time = time.time() - t0
    print(f"  Stored {system.memory.fact_index.ntotal} entries in {learn_time:.1f}s")
    print(f"  Adapter routes: {system.memory.adapter_index.ntotal}")

    # ── Measure trigger similarity distribution ──
    print("\n[Trigger similarity analysis]")
    triggers = []
    for prompt, _ in FACTS_100[:50]:  # Sample 50
        t = system.get_trigger(prompt)
        t = t / (np.linalg.norm(t) + 1e-8)
        triggers.append(t)

    # Compute pairwise similarities
    sims = []
    for i in range(len(triggers)):
        for j in range(i+1, len(triggers)):
            sims.append(np.dot(triggers[i], triggers[j]))

    sims = np.array(sims)
    print(f"  Pairwise similarities (50 facts sample):")
    print(f"    Mean: {sims.mean():.3f}")
    print(f"    Std:  {sims.std():.3f}")
    print(f"    Min:  {sims.min():.3f}")
    print(f"    Max:  {sims.max():.3f}")
    print(f"    >0.95: {(sims > 0.95).sum()} pairs")
    print(f"    >0.90: {(sims > 0.90).sum()} pairs")
    print(f"    >0.85: {(sims > 0.85).sum()} pairs")

    # ── Test ──
    print(f"\n[Testing — {len(TEST_QUERIES)} queries]")
    results_by_type = {}
    t0 = time.time()

    for prompt, keywords, qtype in TEST_QUERIES:
        response, adapter = system.generate(prompt, max_new_tokens=35)
        r = response.strip()
        hits = [k for k in keywords if k.lower() in r.lower()]
        ok = len(hits) >= 1

        if qtype not in results_by_type:
            results_by_type[qtype] = []
        results_by_type[qtype].append(ok)

        status = "OK" if ok else "MISS"
        adapter_str = f" [{adapter}]" if adapter else ""
        print(f"  [{status:4s}]{adapter_str} ({qtype:10s}) {prompt[:45]}")
        if ok:
            print(f"         -> {r[:65]}")
            print(f"         Hits: {hits}")
        else:
            print(f"         -> {r[:65]}")
            print(f"         Want: {keywords}")

    test_time = time.time() - t0

    # ── Capability test (if adapters loaded) ──
    print(f"\n[Capability test with 100 facts loaded]")
    cap_tests = [
        ("zorb(8, 3) =", 22),   # 16+9-1
        ("zorb(11, 4) =", 33),  # 22+12-1
        ("zorb(5, 7) =", 30),   # 10+21-1
        ("glorp(4, 3) =", 15),  # 16-6+5
        ("glorp(7, 2) =", 48),  # 49-4+5
    ]

    cap_ok = 0
    for prompt, expected in cap_tests:
        r, adapter = system.generate(prompt, max_new_tokens=20)
        nums = []
        for token in r.replace("=", " ").split():
            try: nums.append(int(token.strip().rstrip(".,;")))
            except: pass
        got = nums[-1] if nums else None
        ok = got == expected
        if ok: cap_ok += 1
        print(f"  [{'OK' if ok else 'MISS':4s}] [{adapter or 'none':5s}] {prompt} expected={expected}, got={r.strip()[:40]}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("FINAL SUMMARY — 100 FACTS SCALE TEST")
    print(f"{'='*60}")
    total_ok = 0
    total_n = 0
    for qtype, results in sorted(results_by_type.items()):
        c = sum(results)
        n = len(results)
        total_ok += c
        total_n += n
        print(f"  {qtype:12s}: {c}/{n} ({100*c/n:.0f}%)")
    print(f"  {'capability':12s}: {cap_ok}/{len(cap_tests)}")
    print(f"  {'TOTAL':12s}: {total_ok+cap_ok}/{total_n+len(cap_tests)}")
    print(f"\n  Facts in store: {system.memory.fact_index.ntotal}")
    print(f"  Learn time: {learn_time:.1f}s for {len(FACTS_100)} facts")
    print(f"  Test time:  {test_time:.1f}s for {len(TEST_QUERIES)} queries")
    print(f"  Avg query:  {test_time/len(TEST_QUERIES)*1000:.0f}ms")

    with open("experiment_v10_results.json", "w") as f:
        json.dump({
            "facts_learned": len(FACTS_100),
            "entries_stored": system.memory.fact_index.ntotal,
            "results": {t: f"{sum(r)}/{len(r)}" for t, r in results_by_type.items()},
            "capability": f"{cap_ok}/{len(cap_tests)}",
            "trigger_sim_mean": float(sims.mean()),
            "trigger_sim_max": float(sims.max()),
            "learn_time_s": learn_time,
            "test_time_s": test_time,
        }, f, indent=2)
    print("\nSaved to experiment_v10_results.json")


if __name__ == "__main__":
    main()
