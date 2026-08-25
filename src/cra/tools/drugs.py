"""Drug-drug interaction checking.

Two backends behind one interface: a curated local table (the default, and the
only one usable without egress) and an openFDA-backed adapter.

The local table is a *teaching* table assembled for benchmark purposes. It
covers well-established, clinically significant interactions and is
deliberately incomplete. It is not clinical decision support and must not be
used for patient care. Its incompleteness is handled explicitly rather than
silently: the tool reports which drugs it did not recognise, so "no
interactions found" is never confusable with "nothing was checked" -- a
distinction the grounding analysis depends on, since an agent that reports
"no interactions" off an unrecognised drug has made an unsupported claim.
"""

from __future__ import annotations

import time
from itertools import combinations
from typing import Any

from cra.tools.base import EvidenceDraft, ToolResult

# Class membership, so a rule can be written once against a drug class.
DRUG_CLASSES: dict[str, set[str]] = {
    "nsaid": {
        "ibuprofen", "naproxen", "diclofenac", "indomethacin", "ketorolac",
        "meloxicam", "celecoxib", "piroxicam", "etodolac",
    },
    "ssri": {
        "fluoxetine", "sertraline", "paroxetine", "citalopram", "escitalopram",
        "fluvoxamine",
    },
    "snri": {"venlafaxine", "desvenlafaxine", "duloxetine", "milnacipran"},
    "maoi": {"phenelzine", "tranylcypromine", "isocarboxazid", "selegiline", "rasagiline"},
    "triptan": {"sumatriptan", "rizatriptan", "zolmitriptan", "naratriptan", "eletriptan"},
    "statin_cyp3a4": {"simvastatin", "lovastatin", "atorvastatin"},
    "statin": {
        "simvastatin", "lovastatin", "atorvastatin", "rosuvastatin", "pravastatin",
        "fluvastatin", "pitavastatin",
    },
    "azole_antifungal": {
        "ketoconazole", "itraconazole", "fluconazole", "voriconazole", "posaconazole",
    },
    "macrolide_inhibitor": {"erythromycin", "clarithromycin"},
    "ace_inhibitor": {
        "lisinopril", "enalapril", "ramipril", "captopril", "benazepril", "perindopril",
        "quinapril",
    },
    "arb": {"losartan", "valsartan", "irbesartan", "candesartan", "olmesartan", "telmisartan"},
    "potassium_sparing_diuretic": {"spironolactone", "eplerenone", "amiloride", "triamterene"},
    "doac": {"apixaban", "rivaroxaban", "edoxaban", "dabigatran"},
    "antiplatelet": {"aspirin", "clopidogrel", "ticagrelor", "prasugrel", "dipyridamole"},
    "cyp3a4_inducer": {
        "rifampin", "rifampicin", "carbamazepine", "phenytoin", "phenobarbital",
        "st johns wort", "rifabutin",
    },
    "cyp2c19_inhibitor_ppi": {"omeprazole", "esomeprazole"},
    "qt_prolonging": {
        "amiodarone", "sotalol", "haloperidol", "ondansetron", "citalopram", "escitalopram",
        "methadone", "levofloxacin", "moxifloxacin", "ciprofloxacin", "quetiapine",
        "erythromycin", "clarithromycin", "fluconazole",
    },
    "nondihydropyridine_ccb": {"verapamil", "diltiazem"},
    "beta_blocker": {
        "metoprolol", "atenolol", "propranolol", "bisoprolol", "carvedilol", "nadolol",
        "nebivolol",
    },
    "nitrate": {"nitroglycerin", "isosorbide mononitrate", "isosorbide dinitrate"},
    "pde5_inhibitor": {"sildenafil", "tadalafil", "vardenafil", "avanafil"},
    "thiazide": {"hydrochlorothiazide", "chlorthalidone", "indapamide", "metolazone"},
    "loop_diuretic": {"furosemide", "bumetanide", "torsemide", "ethacrynic acid"},
    "tetracycline": {"doxycycline", "tetracycline", "minocycline"},
    "fluoroquinolone": {"ciprofloxacin", "levofloxacin", "moxifloxacin", "ofloxacin"},
    "divalent_cation": {
        "calcium carbonate", "calcium citrate", "ferrous sulfate", "iron", "magnesium oxide",
        "aluminum hydroxide", "sucralfate", "zinc",
    },
}

# Common brand names, so a case that says "Coumadin" is not reported as unknown.
BRAND_TO_GENERIC: dict[str, str] = {
    "coumadin": "warfarin", "jantoven": "warfarin", "eliquis": "apixaban",
    "xarelto": "rivaroxaban", "pradaxa": "dabigatran", "savaysa": "edoxaban",
    "plavix": "clopidogrel", "brilinta": "ticagrelor", "effient": "prasugrel",
    "zocor": "simvastatin", "lipitor": "atorvastatin", "crestor": "rosuvastatin",
    "mevacor": "lovastatin", "pravachol": "pravastatin", "prozac": "fluoxetine",
    "zoloft": "sertraline", "paxil": "paroxetine", "celexa": "citalopram",
    "lexapro": "escitalopram", "luvox": "fluvoxamine", "effexor": "venlafaxine",
    "cymbalta": "duloxetine", "nardil": "phenelzine", "parnate": "tranylcypromine",
    "imitrex": "sumatriptan", "maxalt": "rizatriptan", "ultram": "tramadol",
    "advil": "ibuprofen", "motrin": "ibuprofen", "aleve": "naproxen",
    "toradol": "ketorolac", "celebrex": "celecoxib", "voltaren": "diclofenac",
    "prilosec": "omeprazole", "nexium": "esomeprazole", "protonix": "pantoprazole",
    "prinivil": "lisinopril", "zestril": "lisinopril", "vasotec": "enalapril",
    "altace": "ramipril", "cozaar": "losartan", "diovan": "valsartan",
    "aldactone": "spironolactone", "inspra": "eplerenone", "lanoxin": "digoxin",
    "cordarone": "amiodarone", "pacerone": "amiodarone", "betapace": "sotalol",
    "lasix": "furosemide", "microzide": "hydrochlorothiazide", "glucophage": "metformin",
    "viagra": "sildenafil", "cialis": "tadalafil", "revatio": "sildenafil",
    "zyloprim": "allopurinol", "imuran": "azathioprine", "trexall": "methotrexate",
    "colcrys": "colchicine", "synthroid": "levothyroxine", "levoxyl": "levothyroxine",
    "tegretol": "carbamazepine", "dilantin": "phenytoin", "rifadin": "rifampin",
    "bactrim": "trimethoprim-sulfamethoxazole", "septra": "trimethoprim-sulfamethoxazole",
    "flagyl": "metronidazole", "biaxin": "clarithromycin", "zithromax": "azithromycin",
    "diflucan": "fluconazole", "sporanox": "itraconazole", "nizoral": "ketoconazole",
    "cipro": "ciprofloxacin", "levaquin": "levofloxacin", "zyvox": "linezolid",
    "theo-24": "theophylline", "lithobid": "lithium", "nolvadex": "tamoxifen",
    "lopid": "gemfibrozil", "calan": "verapamil", "cardizem": "diltiazem",
    "lopressor": "metoprolol", "toprol": "metoprolol", "tenormin": "atenolol",
    "inderal": "propranolol", "zofran": "ondansetron", "haldol": "haloperidol",
    "seroquel": "quetiapine", "asa": "aspirin", "acetylsalicylic acid": "aspirin",
    "tylenol": "acetaminophen", "paracetamol": "acetaminophen",
    "smz-tmp": "trimethoprim-sulfamethoxazole", "tmp-smx": "trimethoprim-sulfamethoxazole",
    "co-trimoxazole": "trimethoprim-sulfamethoxazole",
}

# A rule matches an unordered pair. "@name" refers to a class in DRUG_CLASSES.
# severity: contraindicated > major > moderate.
INTERACTIONS: list[dict[str, str]] = [
    # -- anticoagulation ---------------------------------------------------
    {"a": "warfarin", "b": "@nsaid", "severity": "major",
     "mechanism": "Additive bleeding risk: NSAIDs inhibit platelet function and cause GI mucosal injury; some also displace warfarin from albumin.",
     "management": "Avoid where possible; if unavoidable use the lowest dose with gastroprotection and monitor INR and for bleeding."},
    {"a": "warfarin", "b": "@azole_antifungal", "severity": "major",
     "mechanism": "Azoles inhibit CYP2C9, reducing S-warfarin clearance and raising the INR.",
     "management": "Anticipate an INR rise; reduce the warfarin dose and monitor INR closely."},
    {"a": "warfarin", "b": "@macrolide_inhibitor", "severity": "major",
     "mechanism": "Erythromycin and clarithromycin inhibit CYP3A4 and reduce warfarin clearance.",
     "management": "Prefer azithromycin; otherwise monitor INR from day 3 to 5."},
    {"a": "warfarin", "b": "amiodarone", "severity": "major",
     "mechanism": "Amiodarone inhibits CYP2C9 and CYP3A4; the effect builds over weeks and persists after stopping.",
     "management": "Empirically reduce the warfarin dose by roughly a third to a half and monitor INR for weeks."},
    {"a": "warfarin", "b": "trimethoprim-sulfamethoxazole", "severity": "major",
     "mechanism": "Sulfamethoxazole inhibits CYP2C9 and displaces warfarin from protein binding; trimethoprim adds antiplatelet effect.",
     "management": "Choose an alternative antibiotic if possible; otherwise monitor INR within 3 days."},
    {"a": "warfarin", "b": "metronidazole", "severity": "major",
     "mechanism": "Metronidazole inhibits CYP2C9 metabolism of S-warfarin.",
     "management": "Reduce the warfarin dose and monitor INR."},
    {"a": "warfarin", "b": "@cyp3a4_inducer", "severity": "major",
     "mechanism": "Enzyme induction accelerates warfarin clearance and causes loss of anticoagulation.",
     "management": "Expect a falling INR with a rebound rise after the inducer stops; monitor and re-dose."},
    {"a": "@doac", "b": "@cyp3a4_inducer", "severity": "major",
     "mechanism": "Induction of CYP3A4 and P-glycoprotein lowers DOAC exposure and risks treatment failure.",
     "management": "Avoid the combination; rifampin with a DOAC is generally contraindicated."},
    {"a": "@doac", "b": "@nsaid", "severity": "major",
     "mechanism": "Additive bleeding risk.",
     "management": "Avoid; use acetaminophen for analgesia where possible."},
    {"a": "warfarin", "b": "@antiplatelet", "severity": "major",
     "mechanism": "Additive bleeding risk from combined anticoagulant and antiplatelet effect.",
     "management": "Only combine for a defined indication and duration, with gastroprotection."},
    {"a": "@doac", "b": "@antiplatelet", "severity": "major",
     "mechanism": "Additive bleeding risk.",
     "management": "Limit duration; reassess the indication for combined therapy."},
    {"a": "warfarin", "b": "@ssri", "severity": "moderate",
     "mechanism": "SSRIs deplete platelet serotonin and impair aggregation; fluoxetine and fluvoxamine also inhibit CYP2C9.",
     "management": "Monitor for bleeding; monitor INR when starting or stopping."},
    # -- serotonergic ------------------------------------------------------
    {"a": "@ssri", "b": "@maoi", "severity": "contraindicated",
     "mechanism": "Serotonin syndrome from combined reuptake inhibition and blocked monoamine breakdown.",
     "management": "Contraindicated. Observe a washout (2 weeks; 5 weeks after fluoxetine)."},
    {"a": "@snri", "b": "@maoi", "severity": "contraindicated",
     "mechanism": "Serotonin syndrome.",
     "management": "Contraindicated; observe the same washout intervals."},
    {"a": "@ssri", "b": "linezolid", "severity": "major",
     "mechanism": "Linezolid is a reversible non-selective MAO inhibitor; combination risks serotonin syndrome.",
     "management": "Avoid; if linezolid is essential, stop the SSRI and monitor."},
    {"a": "@ssri", "b": "tramadol", "severity": "major",
     "mechanism": "Additive serotonergic effect plus a lowered seizure threshold; CYP2D6 inhibition also alters tramadol activation.",
     "management": "Avoid where possible; otherwise use the lowest dose and counsel on symptoms."},
    {"a": "@ssri", "b": "@triptan", "severity": "moderate",
     "mechanism": "Theoretical additive serotonergic effect; the observed clinical risk is low.",
     "management": "Combination is generally acceptable with counselling on serotonin syndrome symptoms."},
    {"a": "@ssri", "b": "@nsaid", "severity": "major",
     "mechanism": "SSRIs impair platelet aggregation; combined with NSAID mucosal injury this markedly raises upper GI bleeding risk.",
     "management": "Add a proton pump inhibitor or choose a different analgesic."},
    {"a": "@maoi", "b": "pseudoephedrine", "severity": "contraindicated",
     "mechanism": "Indirect sympathomimetic releases accumulated noradrenaline, causing hypertensive crisis.",
     "management": "Contraindicated."},
    # -- statins -----------------------------------------------------------
    {"a": "@statin_cyp3a4", "b": "@azole_antifungal", "severity": "major",
     "mechanism": "CYP3A4 inhibition raises statin exposure and the risk of myopathy and rhabdomyolysis.",
     "management": "Suspend the statin during the antifungal course, or switch to pravastatin or rosuvastatin."},
    {"a": "@statin_cyp3a4", "b": "@macrolide_inhibitor", "severity": "major",
     "mechanism": "CYP3A4 inhibition raises statin exposure and the risk of rhabdomyolysis.",
     "management": "Withhold simvastatin or lovastatin during treatment; azithromycin is a safer macrolide."},
    {"a": "simvastatin", "b": "gemfibrozil", "severity": "contraindicated",
     "mechanism": "Gemfibrozil inhibits statin glucuronidation and OATP1B1 uptake, with additive myotoxicity.",
     "management": "Contraindicated; use fenofibrate if a fibrate is required."},
    {"a": "@statin_cyp3a4", "b": "amiodarone", "severity": "major",
     "mechanism": "CYP3A4 inhibition raises statin concentrations.",
     "management": "Limit simvastatin to 20 mg daily with amiodarone."},
    {"a": "@statin", "b": "@nondihydropyridine_ccb", "severity": "moderate",
     "mechanism": "Verapamil and diltiazem inhibit CYP3A4, raising simvastatin and lovastatin exposure.",
     "management": "Cap the simvastatin dose (10-20 mg) or use a non-CYP3A4 statin."},
    # -- renal, potassium and lithium --------------------------------------
    {"a": "@ace_inhibitor", "b": "@potassium_sparing_diuretic", "severity": "major",
     "mechanism": "Both reduce potassium excretion; additive hyperkalaemia.",
     "management": "Monitor potassium and creatinine within 1-2 weeks of starting or dose change."},
    {"a": "@arb", "b": "@potassium_sparing_diuretic", "severity": "major",
     "mechanism": "Additive hyperkalaemia.",
     "management": "Monitor potassium and renal function."},
    {"a": "@ace_inhibitor", "b": "@arb", "severity": "major",
     "mechanism": "Dual renin-angiotensin blockade increases hyperkalaemia, hypotension and acute kidney injury without outcome benefit.",
     "management": "Avoid routine combination."},
    {"a": "@ace_inhibitor", "b": "@nsaid", "severity": "major",
     "mechanism": "NSAIDs blunt the antihypertensive effect and, with a diuretic, complete the 'triple whammy' causing acute kidney injury.",
     "management": "Avoid in volume depletion or chronic kidney disease; monitor renal function."},
    {"a": "@arb", "b": "@nsaid", "severity": "major",
     "mechanism": "Same mechanism as with ACE inhibitors: reduced renal prostaglandin-dependent perfusion.",
     "management": "Avoid in at-risk patients; monitor renal function and blood pressure."},
    {"a": "lithium", "b": "@nsaid", "severity": "major",
     "mechanism": "Reduced renal prostaglandin synthesis lowers lithium clearance and precipitates toxicity.",
     "management": "Avoid; if needed, monitor lithium levels closely."},
    {"a": "lithium", "b": "@thiazide", "severity": "major",
     "mechanism": "Thiazides increase proximal lithium reabsorption, raising levels substantially.",
     "management": "Reduce the lithium dose and monitor levels."},
    {"a": "lithium", "b": "@ace_inhibitor", "severity": "major",
     "mechanism": "Reduced lithium clearance.",
     "management": "Monitor lithium levels after starting or changing dose."},
    # -- cardiac -----------------------------------------------------------
    {"a": "digoxin", "b": "amiodarone", "severity": "major",
     "mechanism": "Amiodarone inhibits P-glycoprotein, roughly doubling digoxin concentrations.",
     "management": "Halve the digoxin dose when starting amiodarone and monitor levels."},
    {"a": "digoxin", "b": "@nondihydropyridine_ccb", "severity": "major",
     "mechanism": "Reduced digoxin clearance plus additive AV nodal blockade.",
     "management": "Monitor digoxin levels, heart rate and conduction."},
    {"a": "digoxin", "b": "@loop_diuretic", "severity": "moderate",
     "mechanism": "Diuretic-induced hypokalaemia and hypomagnesaemia potentiate digoxin toxicity.",
     "management": "Monitor and replace potassium and magnesium."},
    {"a": "@nondihydropyridine_ccb", "b": "@beta_blocker", "severity": "major",
     "mechanism": "Additive negative chronotropic and inotropic effects; risk of bradycardia, heart block and decompensation.",
     "management": "Avoid intravenous combination; oral use needs close monitoring."},
    {"a": "@nitrate", "b": "@pde5_inhibitor", "severity": "contraindicated",
     "mechanism": "Both raise cyclic GMP, causing profound and refractory hypotension.",
     "management": "Contraindicated. Separate by at least 24 hours (48 hours for tadalafil)."},
    {"a": "@qt_prolonging", "b": "@qt_prolonging", "severity": "major",
     "mechanism": "Additive QT prolongation and risk of torsades de pointes.",
     "management": "Obtain an ECG, correct potassium and magnesium, and avoid where an alternative exists."},
    # -- immunosuppressants and oncology -----------------------------------
    {"a": "allopurinol", "b": "azathioprine", "severity": "contraindicated",
     "mechanism": "Xanthine oxidase inhibition blocks 6-mercaptopurine breakdown, causing severe myelosuppression.",
     "management": "Avoid; if unavoidable, reduce azathioprine to 25-33% of dose with close blood counts."},
    {"a": "methotrexate", "b": "@nsaid", "severity": "major",
     "mechanism": "Reduced renal tubular secretion of methotrexate raises exposure and marrow toxicity.",
     "management": "Particular caution at high methotrexate doses; monitor counts and renal function."},
    {"a": "methotrexate", "b": "trimethoprim-sulfamethoxazole", "severity": "contraindicated",
     "mechanism": "Additive antifolate effect plus reduced renal clearance, causing pancytopenia.",
     "management": "Avoid the combination."},
    {"a": "tamoxifen", "b": "paroxetine", "severity": "major",
     "mechanism": "Strong CYP2D6 inhibition reduces conversion to the active metabolite endoxifen.",
     "management": "Use an SSRI with weak CYP2D6 inhibition, such as venlafaxine or citalopram."},
    {"a": "tamoxifen", "b": "fluoxetine", "severity": "major",
     "mechanism": "Strong CYP2D6 inhibition reduces endoxifen formation.",
     "management": "Switch to a weak CYP2D6 inhibitor."},
    # -- absorption and miscellaneous --------------------------------------
    {"a": "clopidogrel", "b": "@cyp2c19_inhibitor_ppi", "severity": "moderate",
     "mechanism": "CYP2C19 inhibition reduces conversion of clopidogrel to its active metabolite.",
     "management": "Prefer pantoprazole if a proton pump inhibitor is needed."},
    {"a": "metformin", "b": "iodinated contrast", "severity": "major",
     "mechanism": "Contrast-associated acute kidney injury can cause metformin accumulation and lactic acidosis.",
     "management": "Withhold metformin at the time of imaging and restart after renal function is confirmed stable."},
    {"a": "colchicine", "b": "@macrolide_inhibitor", "severity": "major",
     "mechanism": "CYP3A4 and P-glycoprotein inhibition cause colchicine accumulation, which can be fatal.",
     "management": "Avoid; reduce dose substantially in renal impairment."},
    {"a": "colchicine", "b": "@azole_antifungal", "severity": "major",
     "mechanism": "CYP3A4 and P-glycoprotein inhibition cause colchicine accumulation.",
     "management": "Avoid the combination."},
    {"a": "theophylline", "b": "ciprofloxacin", "severity": "major",
     "mechanism": "CYP1A2 inhibition sharply raises theophylline levels, risking seizures and arrhythmia.",
     "management": "Avoid; if needed, reduce the theophylline dose and monitor levels."},
    {"a": "levothyroxine", "b": "@divalent_cation", "severity": "moderate",
     "mechanism": "Chelation in the gut reduces levothyroxine absorption.",
     "management": "Separate administration by at least 4 hours."},
    {"a": "@tetracycline", "b": "@divalent_cation", "severity": "moderate",
     "mechanism": "Chelation markedly reduces antibiotic absorption.",
     "management": "Separate doses by 2-4 hours."},
    {"a": "@fluoroquinolone", "b": "@divalent_cation", "severity": "moderate",
     "mechanism": "Chelation markedly reduces absorption and can cause treatment failure.",
     "management": "Give the antibiotic 2 hours before or 6 hours after the cation."},
    {"a": "@cyp3a4_inducer", "b": "ethinyl estradiol", "severity": "major",
     "mechanism": "Accelerated metabolism of contraceptive steroids risks contraceptive failure.",
     "management": "Use an additional or alternative non-hormonal method."},
    {"a": "@potassium_sparing_diuretic", "b": "potassium chloride", "severity": "major",
     "mechanism": "Additive potassium load with impaired excretion.",
     "management": "Avoid routine combination; monitor potassium."},
]

DISCLAIMER = (
    "Local curated interaction table for research benchmarking only. It is deliberately "
    "incomplete and is not clinical decision support."
)


def normalize_drug(name: str) -> str:
    n = " ".join(str(name).strip().lower().replace("_", " ").split())
    for suffix in (" tablet", " tablets", " capsule", " capsules", " oral", " iv", " po"):
        n = n.removesuffix(suffix)
    n = n.strip(".,;")
    return BRAND_TO_GENERIC.get(n, n)


def _known(drug: str) -> bool:
    if any(drug == r["a"] or drug == r["b"] for r in INTERACTIONS):
        return True
    return any(drug in members for members in DRUG_CLASSES.values())


def _matches(drug: str, token: str) -> bool:
    if token.startswith("@"):
        return drug in DRUG_CLASSES.get(token[1:], set())
    return drug == token


def find_interactions(drugs: list[str]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Return (interactions, recognised, unrecognised) for a list of drug names."""
    normalized, seen = [], set()
    for raw in drugs:
        n = normalize_drug(raw)
        if n and n not in seen:
            seen.add(n)
            normalized.append(n)

    recognised = [d for d in normalized if _known(d)]
    unrecognised = [d for d in normalized if not _known(d)]

    found: list[dict[str, Any]] = []
    for d1, d2 in combinations(normalized, 2):
        for rule in INTERACTIONS:
            hit = (_matches(d1, rule["a"]) and _matches(d2, rule["b"])) or (
                _matches(d2, rule["a"]) and _matches(d1, rule["b"])
            )
            if hit:
                found.append({**rule, "drug_a": d1, "drug_b": d2})
                break  # one rule per pair: the table is ordered most-severe-first
    order = {"contraindicated": 0, "major": 1, "moderate": 2}
    found.sort(key=lambda r: order.get(r["severity"], 3))
    return found, recognised, unrecognised


class DrugInteractionTool:
    name = "check_drug_interactions"
    description = (
        "Check a list of drugs for pairwise interactions against a curated table of "
        "clinically significant interactions. Returns severity, mechanism and management for "
        "each pair found, and explicitly names any drug it did not recognise. Use this whenever "
        "a case lists two or more medications. Treat an unrecognised drug as unchecked, not as "
        "safe."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "drugs": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "description": "Generic or brand drug names, at least two.",
            }
        },
        "required": ["drugs"],
        "additionalProperties": False,
    }

    def run(self, drugs: list[str] | None = None, **_: Any) -> ToolResult:
        started = time.perf_counter()
        if not isinstance(drugs, list) or len(drugs) < 2:
            return ToolResult.failure(
                "'drugs' must be a list of at least two drug names; "
                f"got {drugs!r}. An interaction check needs a pair."
            )

        found, recognised, unrecognised = find_interactions(drugs)
        lines: list[str] = []
        if found:
            lines.append(f"{len(found)} interaction(s) found:")
            for r in found:
                lines.append(
                    f"- {r['drug_a']} + {r['drug_b']} [{r['severity'].upper()}]\n"
                    f"    Mechanism: {r['mechanism']}\n"
                    f"    Management: {r['management']}"
                )
        else:
            lines.append("No interactions found in the curated table for the recognised drugs.")
        if unrecognised:
            lines.append(
                "NOT CHECKED - these names are absent from the local table, so nothing can be "
                f"concluded about them: {', '.join(unrecognised)}"
            )
        if recognised:
            lines.append(f"Checked: {', '.join(recognised)}")
        lines.append(DISCLAIMER)
        output = "\n".join(lines)

        return ToolResult(
            ok=True,
            output=output,
            evidence=[
                EvidenceDraft(
                    kind="tool_output",
                    text=output,
                    title="Drug interaction check (curated local table)",
                    source_id=f"drug_interactions:{sorted(recognised + unrecognised)}",
                    metadata={
                        "n_interactions": len(found),
                        "severities": [r["severity"] for r in found],
                        "recognised": recognised,
                        "unrecognised": unrecognised,
                    },
                )
            ],
            latency_ms=(time.perf_counter() - started) * 1000,
        )
